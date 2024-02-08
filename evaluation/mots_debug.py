import  os
import glob
import numpy as np
import PIL.Image as Image
import pycocotools.mask as rletools
import cv2
from mots_metric import compute_MOTS_metrics

IGNORE_CLASS = 10

class SegmentedObject:
  def __init__(self, mask, class_id, track_id):
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id

def load_seqmap(seqmap_filename):
    print("Loading seqmap...")
    seqmap = []
    max_frames = {}
    with open(seqmap_filename, "r") as fh:
        for i, l in enumerate(fh):
            fields = l.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)
            max_frames[seq] = int(fields[3])
    return seqmap, max_frames

def load_image(filename, id_divisor=1000): # 输入图片不是
    #img = np.array(Image.open(filename))
    img = np.array(cv2.imread(filename,cv2.IMREAD_GRAYSCALE))
    obj_ids = np.unique(img) # 像素点
    objects = []
    mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
    for idx, obj_id in enumerate(obj_ids):
      if obj_id == 0:  # background
          continue
      mask.fill(0)
      pixels_of_elem = np.where(img == obj_id)
      mask[pixels_of_elem] = 1  # mask 分别对应每一个instance
      objects.append(SegmentedObject(
          rletools.encode(mask),
          obj_id // id_divisor,
          obj_id
      )) # obj_id 是以像素点编码的
    return objects

def filename_to_frame_nr(filename):
  frame = filename.split('_')[-2]
  # assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
  return int(frame)

def load_images_for_folder(path):
  '''
  递增序列
  '''
  img_list = glob.glob(os.path.join(path, "*.png"))
  # print(img_list)
  # exit()
  frames=[] 
  for img in img_list:
      img_ = img.split('/')[-1]
      frame = int(img_.split('.')[0])
      frame = (img,frame)    
      frames.append(frame)

  #files = sorted(glob.glob(os.path.join(path, "*.png"))) #gt 的png文件
  frames = sorted(frames,key = lambda frames: frames[1])
  files = []
  for frame in frames :
      files .append(frame[0]) 
  objects_per_frame = {}
  print('files:',files)

  for file in files:
      objects = load_image(file)
      # print('file:',file)
      # image = file.split('/')[-1]
      # # print(image)
      # frame = int(image.split('.')[0])
      # print(frame)
      # frame = filename_to_frame_nr(os.path.basename(file))
      file = img.split('/')[-1]
      frame = int(img_.split('.')[0])
      objects_per_frame[frame] = objects
  return objects_per_frame # 对于每个序列的结果保存在每一帧中，key 帧号 
 
def load_sequences(path, seqmap):
  objects_per_frame_per_sequence = {}
  for seq in seqmap:
    print("Loading sequence", seq)
    seq_path_folder = os.path.join(path, seq)
    seq_path_txt = os.path.join(path, seq + ".txt")
    if os.path.isdir(seq_path_folder):
      objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
    elif os.path.exists(seq_path_txt):
      objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
    else:
      assert False, "Can't find data in directory " + path
  return objects_per_frame_per_sequence

def mask_iou(a, b, criterion="union"):
  is_crowd = criterion != "union"
  return rletools.iou([a.mask], [b.mask], [is_crowd])[0][0]

def evaluate_class(gt, results, max_frames, class_id):
  _, results_obj = compute_MOTS_metrics(gt, results, max_frames, class_id, IGNORE_CLASS, mask_iou)
  return results_obj

gt_folder = '/home/yr/code/PVIS/Human_Video/VIS/instance'
# results_folder = '/home/yr/code/PVIS/Human_Video/output/valid/bbox_bbox_mask'
results_folder = '/home/yr/code/PVIS/results/vis'
# results_folder = '/home/yr/code/PVIS/reslut_hvis'
# results_folder = '/home/yr/code/PVIS/results/mask_bbox_bbox'


# results_folder = '//data/public/Transfer/models/y50012820/code/MaskTrackRCNN-master/img_result_pre/'
seqmap_filename = '/home/yr/code/PVIS/ours/blendtrack/evaluation/hvis.seqmap'
seqmap, max_frames = load_seqmap(seqmap_filename)
gt = load_sequences(gt_folder, seqmap)
results = load_sequences(results_folder, seqmap)
print("Evaluate class: Pedestrians")
results_ped = evaluate_class(gt, results, max_frames, 0)