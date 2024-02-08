import os
import cv2
import json
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
import multiprocessing as mp
import detectron2.data.transforms as T
from adet.config import get_cfg
from pycocotools.ytvos import YTVOS
from matching import ious
from detectron2.utils.logger import setup_logger
from track import eval_seq
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "9"  
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default="/data/public/Transfer/models/y50012820/code/MaskTrackRCNN-master/yrtest/data/smalldata/simple/annotations/C0050.json",nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    print('args.input',args.input)
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
    start = time.time()
    for sequence in args.input:
        savename = sequence.split('/')[-1]
        sequence_paths = os.listdir(sequence)  
        frames=[] 
        for img in sequence_paths:
            img = os.path.join(sequence,img)
            file_name_ = img.split('/')[-1]
            if file_name_.endswith('jpg')|file_name_.endswith('png'):
                file_name = file_name_.split('.')[0]
                frame = int(file_name)
                # frame = int(file_name)
                frame = (img,frame)    
                frames.append(frame)
        frames = sorted(frames,key = lambda frames: frames[1])
        img_list_ = []
        for frame in frames :
            img_list_ .append(frame[0]) 
        # print('img_list_:',img_list_)
        a = eval_seq(cfg,img_list_,savename)
    end = time.time()
    # print('time:',end-start)
        # exit()