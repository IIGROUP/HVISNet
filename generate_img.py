import matplotlib.pyplot as plt
import os
import numpy as np
from imantics import Polygons, Mask
import json
import cv2
root_folder = '/home/yr/code/PVIS/ours'
root_image = '/home/yr/code/PVIS/Human_Image/coco/val/images'
root_mask = '/home/yr/code/PVIS/Human_Image/coco/val/instance'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#从png的mask里提取mask per object, 
#input: mask(0-1)
#output:masks: list, each of them is numpy array(bool)
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

 
def mask_to_polygons(mask):
    polygons = Mask(mask).polygons().points

    # filter out invalid polygons (< 3 points)
    polygons_filtered = []
    for polygon in polygons:
        polygon = polygon.reshape(-1)
        polygon = polygon.tolist()
        if  len(polygon) % 2 == 0 and len(polygon) >= 6:
            polygons_filtered.append(polygon)
    return polygons_filtered
def generate_standert_dataset_dict(root_image,root_mask):
    img_id = 0
    ann_id = 0
    standert_dataset_dicts = {'categories':[], 'images':[], 'annotations':[]}
    standert_dataset_dicts['categories'].append({'id':1, 'name':'person'})
    image_paths = os.listdir(root_image)
    mask_paths = os.listdir(root_mask)
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)
    for image_path,mask_path in zip(image_paths,mask_paths):
        img_id +=1 
        image_file = root_image.split('/')[-4:]
        image_file = '/'.join(image_file) 
        image_file_name = os.path.join(image_file,image_path)
        image_path_r = os.path.join(root_image,image_path)
        mask_path_r = os.path.join(root_mask,mask_path)
        image = cv2.imread(image_path_r)
        height = image.shape[0]
        width = image.shape[1]
        # print(image.shape)
        image_ann = {'file_name':image_file_name,'id':img_id,'height':height,'width':width}
        standert_dataset_dicts['images'].append(image_ann)
        mask = cv2.imread(mask_path_r,0)
        masks, masks_cat = extract_mask(mask)
        for sub_mask,sub_mask_cat in zip(masks,masks_cat):
            ann_id +=1 
            # obj = {}
            area,box = bbox(sub_mask)
            segmentation = mask_to_polygons(sub_mask)
            if len(segmentation)==0:
                continue
            
            assert len(segmentation)!=0
            annotation = {'area': int(area),
                'bbox': box,
                'category_id': 1,
                'id': ann_id,
                'image_id': img_id,
                'iscrowd': 0,
                'sub_mask_cat': sub_mask_cat,
                'segmentation': segmentation
                } 
            standert_dataset_dicts['annotations'].append(annotation)
    return standert_dataset_dicts

def save_all(root_image,root_mask):
    ann_folder = os.path.join(root_folder, 'annotations')
    saved_file_name = 'coco_val.json'
    if not os.path.exists(ann_folder):
        os.makedirs(ann_folder)
    file_path = os.path.join(ann_folder, saved_file_name)
    print('start generate...')
    standert_dataset_dicts = generate_standert_dataset_dict(root_image,root_mask)
    # json_name = os.path.join(anno_path, 'annotations/train_FBMS.json')
    with open(file_path, 'w') as f:
        json.dump(standert_dataset_dicts,f,cls=NpEncoder) 
    print('save data file in ', file_path)

if __name__=='__main__':
    save_all(root_image,root_mask)
