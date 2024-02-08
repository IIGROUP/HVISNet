import json 
import cv2
import numpy as np
import os

from numpy.core.defchararray import center
mask_instance = '/mnt/data/datasets/yr/OCHuman/ours/clean/instance'
mask_paths = os.listdir(mask_instance)
def extract_mask(mask):
    mask = mask.astype(np.uint8)
    masks = []
    masks_cat = []
    for i in range(1, 256):
        sub_mask = (mask==i)
        if np.sum(sub_mask) !=0:
            masks.append(sub_mask)
            masks_cat.append(i)
    return masks, masks_cat


#extract bbox from mask of one object
def bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    width = cmax - cmin + 1
    height = rmax - rmin + 1
    maskInt = mask.astype(int)
    area = np.sum(maskInt)
    return area, [int(cmin), int(rmin), int(width), int(height)]
nums = 0
for mask_path in mask_paths:
    mask_path = os.path.join(mask_instance, mask_path)
    mask = cv2.imread(mask_path,0)
    # print(mask_path,mask.shape)
    masks, masks_cat = extract_mask(mask)
    for sub_mask,sub_mask_cat in zip(masks,masks_cat):
        area,box = bbox(sub_mask)
        # print(box)
        # center_x,center_y=box[1]+int(box[2]/2),box[1]+int(box[3]/2)
        # cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), [255,0,0], 2)
        ys = np.arange(0,mask.shape[0],dtype=np.float32)
        xs = np.arange(0,mask.shape[1],dtype=np.float32)
        m00 = np.sum(sub_mask)
        m10 = np.sum(sub_mask*xs)
        m01 = np.sum(sub_mask*ys[:,None])
        center_x = int(m10 / m00)
        center_y = int(m01 / m00)
        try:
            if sub_mask[center_x][center_y]!=True:
                nums+=1
        except:
            if sub_mask[center_y][center_x]!=True:
                nums+=1
print('nums:',nums)