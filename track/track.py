from multitracker import JDETracker,STrack
import os.path as osp
from timer import Timer
from detectron2.data.detection_utils import read_image
import numpy as np
import copy
import cv2
import os
from basetrack import BaseTrack, TrackState
def IOU( box1, box2 ):

    """

    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标

    :param box2:[x1,y1,x2,y2]

    :return: iou_ratio--交并比

    """

    width1 = abs(box1[2] - box1[0])

    height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值

    width2 = abs(box2[2] - box2[0])

    height2 = abs(box2[1] - box2[3])

    x_max = max(box1[0],box1[2],box2[0],box2[2])

    y_max = max(box1[1],box1[3],box2[1],box2[3])

    x_min = min(box1[0],box1[2],box2[0],box2[2])

    y_min = min(box1[1],box1[3],box2[1],box2[3])

    iou_width = x_min + width1 + width2 - x_max

    iou_height = y_min + height1 + height2 - y_max

    if iou_width <= 0 or iou_height <= 0:

        iou_ratio = 0

    else:

        iou_area = iou_width * iou_height # 交集的面积

        box1_area = width1 * height1

        box2_area = width2 * height2

        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积

    return iou_ratio

def eval_seq(cfg, data,savename,save_dir=None, show_image=True, frame_rate=30):
    BaseTrack._count = 0
    tracker = JDETracker(cfg, frame_rate=frame_rate)
    # timer = Timer()
    results = []
    frame_id = 0
    min_box_area = 200
    for i,path in enumerate(data):        
        # print('path:',path)
        # start = timer.tic()
        # print(start)
        image = read_image(path, format="BGR")
        online_targets,det_bboxes,det_masks,locations = tracker.update(path,image)
        online_tlwhs = []
        online_ids = []
        online_masks = []
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        iou_matrix = np.zeros([len(online_targets),len(det_bboxes)])
        # exit()
        for i,t in enumerate(online_targets):
            tlwh = t.tlwh
            tid = t.track_id
            mask = t.mask
            tlwh_ = copy.deepcopy(tlwh)
            tlwh_[2] = tlwh_[0]+tlwh_[2]
            tlwh_[3] = tlwh_[1]+tlwh_[3]
            for j,bbox in enumerate(det_bboxes):
                iou_matrix[i][j] = round(IOU(bbox,tlwh_))            
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_masks.append(mask)
        # end=timer.toc()
        # print(iou_matrix)

        try:
            max_index = np.argmax(iou_matrix,axis=1)
            temp = copy.deepcopy(det_masks)
            temp_location = copy.deepcopy(locations)
            for i,index in enumerate(max_index):
                if i!= index:
                    det_masks[i] = temp[index] 
                    locations[i] = temp_location[index] 
        except:
            pass
        # print('det_bboxes____',det_bboxes)
        # print()
        results.append(online_ids)
        # print('online_ids',online_ids)
        img = cv2.imread(path)
        h = np.shape(img)[0]
        w = np.shape(img)[1]
        img_0 = np.zeros((h,w,3),dtype=np.float32)
        color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(150,150,255),(150,0,255),(255,50,50),(255,100,100),(255,150,150),(0,150,150),(150,150,150),
        (200,200,150),(100,100,0),(100,0,100),(50,0,50),(50,50,50),(200,0,0),(0,200,0),(0,0,200),(200,200,0),(200,0,200),(0,200,200),(0,150,0),(150,0,0),(0,0,150),(150,150,0),(150,150,0)] 
        # for 
        for c,bbox,masks,mask,location in zip(online_ids,online_tlwhs,online_masks,det_masks,locations):
            # iou = ious(bbox,box)
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),color[i], 2)
            # cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color[i], 2)
        #     exit()
            coef = 255 if np.max(img) < 3 else 1
            img = (img * coef).astype(np.float32)
            mask = mask.astype(np.uint8)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_0, contours, -1, color[c], -1)
            # cv2.circle(img, (int(location[0]*1.25),int(location[1]*1.25)), 1, color[i], 4)
            # cv2.circle(img_0, (int(location[0]*1.25),int(location[1]*1.25)), 1, color[i], 4)
        try:
            dst=cv2.addWeighted(img,0.7,img_0,0.3,0)
        except:
            dst = img
        file_name = path.split('/')[-1]
        name = file_name.split('.')[0]
        # save_path = '/data/public/Transfer/models/y50012820/code/MOTS/results_folder/314999/'
        # save_path = '/home/yr/code/PVIS/Human_Video/output/valid/vis/'
        save_path = '/home/yr/code/PVIS/results/mask_bbox_bbox/'
        save_path = os.path.join(save_path,savename)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+str(name)+'.png', img_0) 