import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from imantics import Polygons, Mask

root_image = '/home/yr/code/PVIS/Human_Video/MOTSchallenge/image'
root_mask = '/home/yr/code/PVIS/Human_Video/MOTSchallenge/instance'

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

def generate_standert_dataset_dict(root_image,root_mask):
    video_paths = sorted(os.listdir(root_image))
    print('video_paths:',len(video_paths))
    instance_paths = sorted(os.listdir(root_mask))
    standert_dataset_dicts = {'categories':[], 'videos':[], 'annotations':[]}
    standert_dataset_dicts['categories'].append({'id':1, 'name':'person'})
    video_id = 0
    ann_id = 0
    category_id= 1
    for image_sequence,mask_sequence in zip(video_paths,instance_paths):
        print('image_sequence:',image_sequence)
        video_id +=1
        image_root = os.path.join(root_image,image_sequence)
        mask_root = os.path.join(root_mask,image_sequence)
        image_paths_ = os.listdir(image_root)
        # print('image_paths_:',image_paths_)

        image_paths = []
        for image_path in image_paths_ :
            if image_path[-3:]=='png' or image_path[-3:]=='jpg' :
                image_paths.append(image_path)
        # print('image_paths:',image_paths)
        mask_paths = os.listdir(mask_root)
        image_paths = sorted(image_paths)
        mask_paths = sorted(mask_paths)
        len_video = len(image_paths)
        image_names = []
        video_maskcat = []
        image_path_r = os.path.join(image_root,image_paths[0])
        image = cv2.imread(image_path_r)
        height = image.shape[0]
        width = image.shape[1]

        for image_path,mask_path in zip(image_paths,mask_paths):
            image_path = os.path.join(image_sequence,image_path)
            mask_path = os.path.join(image_sequence,mask_path)
            image_file = root_image.split('/')[-2:]
            image_file = '/'.join(image_file) 
            image_path = os.path.join(image_file,image_path)
            image_names.append(image_path)        
            mask_path_r = os.path.join(root_mask,mask_path)
            mask = cv2.imread(mask_path_r,0)    
            try:    
                masks, masks_cat = extract_mask(mask)
            except:
                print('path:',mask_path_r)
                exit()
            for mask_cat in masks_cat:
                mask_cat = np.array(mask_cat)
                video_maskcat.append(mask_cat)
            # masks_cat = np.array(masks_cat)
            # video_maskcat.append(masks_cat)
        video_dict = {'file_names':image_names,
        'id':video_id,
        'height':height,
        'width':width,
        'length':len_video
        }
        standert_dataset_dicts['videos'].append(video_dict)
        video_maskcat = np.array(video_maskcat)
        video_maskcat = np.reshape(video_maskcat,[-1])
        video_maskcat = np.unique(video_maskcat)
        for gray in video_maskcat:
            ann_id +=1
            video_bbox =[]
            video_segmentation=[]
            video_area=[]
                # id = id +1          
            for mask_path_an in mask_paths:
                mask_path_an = os.path.join(mask_sequence,mask_path_an)
                mask_path_r_an = os.path.join(root_mask,mask_path_an)
                mask_p =  cv2.imread(mask_path_r_an,0)
                mask_1=np.reshape(mask_p,(1,-1))
                if gray in mask_1:
                    sub_mask = (mask_p==gray)
                    area,box = bbox(sub_mask)
                    segmentation = mask_to_polygons(sub_mask)
                else:
                    box = None
                    area = None
                    segmentation =None
                video_bbox.append(box)
                video_area.append(area)
                video_segmentation.append(segmentation)
            anno_dict={'height':height,
                'width':width,
                'length':len_video,
                'category_id':category_id,
                'video_id':video_id,
                'iscrowd':0,
                'bboxes':video_bbox,
                'areas':video_area,
                'segmentations':video_segmentation,
                'id':ann_id
                }
            standert_dataset_dicts['annotations'].append(anno_dict)
    return standert_dataset_dicts

def save_all(root_image,root_mask):
    anno_path = '/home/yr/code/PVIS/Human_Video/annotations'

    folder = os.path.join(anno_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('start generate...')
    standert_dataset_dicts = generate_standert_dataset_dict(root_image,root_mask)
    json_name = os.path.join(anno_path, 'hvis_motschallenge.json')
    with open(json_name, 'w') as f:
        json.dump(standert_dataset_dicts,f,cls=NpEncoder) 
    print('save data file in ', json_name)

if __name__=='__main__':
    save_all(root_image,root_mask)
