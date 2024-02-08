import cv2
import os
import torch
import matching
import numpy as np
import torch.nn.functional as F
from log import logger
from collections import deque
from kalman_filter import KalmanFilter
from basetrack import BaseTrack, TrackState
from detectron2.engine.defaults import DefaultPredictor



# from tracking_utils.utils import *
# from utils.post_process import ctdet_post_process



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score,mask,temp_feat, buffer_size=30):
        # detection 坐标 得分 embedding feature
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.mask = mask
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        #  frame_id 1
        self.kalman_filter = kalman_filter # tracking_utils.kalman_filter.KalmanFilter object at 0x7f94de5efed0
        self.track_id = self.next_id() #1
        # self._tlwh [x,y,w,h] self.tlwh_to_xyah(self._tlwh) [centerx,centery,ratio,h]
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked # 1
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.mask = new_track.mask
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, cfg, frame_rate=30):
        self.cfg = cfg.clone()
        print('Creating model...')
        self.predictor = DefaultPredictor(cfg)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = 0.5
        # self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = 30
        # 待调
        mean = [0.408, 0.447, 0.470]
        std = [0.289, 0.274, 0.278]
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.kalman_filter = KalmanFilter()

    def update(self,path, im_blob):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        ''' Step 1: Network forward, get detections & embeddings'''
        predictions = self.predictor(im_blob)
        instances = predictions["instances"].to(torch.device("cpu"))
        locations = instances.locations.numpy()
        fpn_levels  = instances.fpn_levels.numpy()
        det_bboxes = instances.pred_boxes.tensor.numpy()
        det_masks = np.asarray(instances.pred_masks)    
        det_embeddings = instances.embeddings.numpy() # id_feature
        det_scores = instances.scores.numpy()
        # stride = [8,16,32]
        # print('locations',locations)
        # print('det_bboxes',det_bboxes)
        # print('fpn_levels',fpn_levels)

        # exit()
        # if self.frame_id == 6:
        #     exit()
        # img = cv2.imread(path)
        # h = np.shape(img)[0]
        # w = np.shape(img)[1]
        # img_0 = np.zeros((h,w,3),dtype=np.float32)
        # color = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),(150,150,255),(150,0,255)] 
        # for i in range(0, det_bboxes.shape[0]):
        #     mask = det_masks[i]
        #     coef = 255 if np.max(img) < 3 else 1
        #     img = (img * coef).astype(np.float32)
        #     mask = mask.astype(np.uint8)*255
        #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(img_0, contours, -1, color[i], -1)
        #     cv2.rectangle(img, (int(det_bboxes[i][0]), int(det_bboxes[i][1])),(int(det_bboxes[i][2]), int(det_bboxes[i][3])),color[i], 2)
        #     cv2.circle(img, (locations[i][0],locations[i][1]), 1, (255,255,255), 4)
        #     cv2.circle(img_0, (locations[i][0],locations[i][1]), 1, (255,255,255), 4)
        # dst=cv2.addWeighted(img,0.7,img_0,0.3,0)
        #     # # cv2.rectangle(img, (bbox[0], bbox[1]),
        #     #                 (bbox[2], bbox[3]),
        #     #                 (0, 255, 0), 2)
        # save_path = '/data/public/Transfer/models/y50012820/code/AdelaiDet-master/videos_out/debug/0002/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(save_path+str(self.frame_id)+'a.jpg', dst)
        # except:
        #     pass
        # exit()
        ''' visiual
        img = cv2.imread(path)
        for i in range(0, det_bboxes.shape[0]):
            bbox = det_bboxes[i][0:4]
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imwrite(str(self.frame_id)'.jpg', img)
        '''

        if len(det_bboxes) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(box), score,mask,f, 30) for
                (box,score,mask,f) in zip(det_bboxes,det_scores,det_masks,det_embeddings)] #将特征保存在每一个跟踪目标中    
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # print('unconfirmed',unconfirmed)
        # print('tracked_stracks',tracked_stracks)
        ''' Step 2: First association, with embedding
            第一次数据关联，计算特征距离    
        '''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # print('matches',matches)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        ''' Step 3: Second association, with IOU
            将未匹配上的检测框与跟踪框再通过IOU进行一次匹配，防止遗漏的目标
        '''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        """ Step 4: Init new stracks
            初始化新增目标轨迹
        """
        for inew in u_detection:
            track = detections[inew]
            mask = track.mask
            if track.score < self.det_thresh:
                continue
            # print('detections',mask)
            # coef = 255 if np.max(img) < 3 else 1
            # img = (img * coef).astype(np.float32)
            # mask = mask.astype(np.uint8)*255
            # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img_0, contours, -1,(255,0,0) , -1)
            # dst=cv2.addWeighted(img,0.7,img_0,0.3,0)
            # cv2.imwrite(str(self.frame_id)+'a.jpg', dst)
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        # exit()
        """ Step 5: Update state
            更新所有轨迹状态
        """
        # print('activated_starcks',activated_starcks)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks,det_bboxes,det_masks,locations
    




def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
