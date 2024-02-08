import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import argparse
import math
import random
from scipy.spatial import distance
from inner_center.utils import *

class TOsmallError(Exception):
    pass
def inner_dot(instance_mask,point):
    xp, yp = point
    h,w = instance_mask.shape
    bool_inst_mask = instance_mask.astype(bool)
    neg_bool_inst_mask = 1 - bool_inst_mask
    dot_mask = np.zeros(instance_mask.shape)
    insth,instw = instance_mask.shape
    dot_mask[yp][xp] = 1
    if yp + 1 >= h or yp -1 < 0 or xp + 1 >= w or xp -1 < 0:
        return False
    fill_mask = np.zeros((3,3))
    fill_mask.fill(1)
    dot_mask[yp-1:yp+2, xp-1:xp+2] = fill_mask
    not_inner = (neg_bool_inst_mask * dot_mask).any()
    # print(np.sum(neg_bool_inst_mask),np.sum(dot_mask))
    # print('neg_bool',np.unique(dot_mask))
    return not not_inner

def centerdot(instance_mask,gt_box):
    # boundingorder x, y
    # instance_mask 
    im_h,im_w = instance_mask.shape
    bool_inst_mask = instance_mask.astype(bool)
    bool_inst_mask_ = bool_inst_mask.astype(int)
    x,y,x_,y_ = gt_box
    w = x_ - x
    h = y_ - y
    # avg_center_float = np.array([x + w/2, y + h/2]) # w,h
    # avg_center = (int(avg_center_float[0]),int(avg_center_float[1])) # w,h

    # x, y, w, h = cv.boundingRect(instance_mask)
    # avg_center_float = (x + w/2, y + h/2) # w,h
    # avg_center = (int(avg_center_float[0]), int(avg_center_float[1]))
    ys = np.arange(0,im_h,dtype=np.float32)
    xs = np.arange(0,im_w,dtype=np.float32)
    m00 = np.sum(bool_inst_mask_)
    m10 = np.sum(bool_inst_mask_*xs)
    m01 = np.sum(bool_inst_mask_*ys[:,None])
    center_x = m10 / m00
    center_y = m01 / m00
    avg_center_float = np.array([center_x, center_y]) # w,h
    avg_center = (int(avg_center_float[0]),int(avg_center_float[1])) 

    temp = np.zeros(instance_mask.shape)
    temp[int(avg_center[1])][int(avg_center[0])] = 1
    if (bool_inst_mask == temp).any() and inner_dot(instance_mask,avg_center):
        return avg_center_float
    else:
        
        inst_mask_h, inst_mask_w = np.where(instance_mask)
        
        # get gradient_map
        gradient_map = get_gradient(instance_mask)
        grad_h,grad_w = np.where(gradient_map == 1)

        # inst_points
        inst_points = np.array([[inst_mask_w[i], inst_mask_h[i]] for i in range(len(inst_mask_h))])
        '''
        sample
        '''
        sample_int = np.arange(0,len(inst_points),100,dtype=int)
        inst_points = inst_points[sample_int]
        # edge_points
        bounding_order = np.array([[grad_w[i], grad_h[i]] for i in range(len(grad_h))])
        sample = np.random.randint(0,len(bounding_order),size=int(len(bounding_order)/100),dtype = int)
        bounding_order = bounding_order[sample]
        if (len(inst_points)>0)&(len(bounding_order)>0):
            distance_result = distance.cdist(inst_points,bounding_order,'euclidean')
            sum_distance = np.sum(distance_result,1)
            center_index = np.argmin(sum_distance)

            center_distance = (inst_points[center_index][0],inst_points[center_index][1])
            times_num = 0
            len_points = len(inst_points)
            while not inner_dot(instance_mask,center_distance):
                times_num += 1
                sum_distance = np.delete(sum_distance,center_index)
                if len(sum_distance) == 0:
                    # print('no center')
                    # raise TOsmallError
                    return avg_center_float

                center_index = np.argmin(sum_distance)
                center_distance = (inst_points[center_index][0],inst_points[center_index][1])
                if times_num >len_points:
                    break
            
            return center_distance     
        else:
            return avg_center_float
