# AdelaiDet

AdelaiDet is an open source toolbox for multiple instance-level recognition tasks on top of [Detectron2](https://github.com/facebookresearch/detectron2).
All instance-level recognition works from our group are open-sourced here.

To date, AdelaiDet implements the following algorithms:

* [FCOS](configs/FCOS-Detection/README.md)
* [BlendMask](configs/BlendMask/README.md)
* [MEInst](configs/MEInst-InstanceSegmentation/README.md)
* [ABCNet](configs/BAText/README.md)
* [CondInst](configs/CondInst/README.md)
* [SOLO](https://arxiv.org/abs/1912.04488) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [SOLOv2](https://arxiv.org/abs/2003.10152) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [DirectPose](https://arxiv.org/abs/1911.07451) _to be released_




## Models
### COCO Object Detecton Baselines with [FCOS](https://arxiv.org/abs/1904.01355)
Name | inf. time | box AP | download
--- |:---:|:---:|:---:
[FCOS_R_50_1x](configs/FCOS-Detection/R_50_1x.yaml) | 16 FPS | 38.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download)
[FCOS_MS_R_101_2x](configs/FCOS-Detection/MS_R_101_2x.yaml) | 12 FPS | 43.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/M3UOT6JcyHy2QW1/download)
[FCOS_MS_X_101_32x8d_2x](configs/FCOS-Detection/MS_X_101_32x8d_2x.yaml) | 6.6 FPS | 43.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/R7H00WeWKZG45pP/download)
[FCOS_MS_X_101_32x8d_dcnv2_2x](configs/FCOS-Detection/MS_X_101_32x8d_2x_dcnv2.yaml) | 4.6 FPS | 46.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/TDsnYK8OXDTrafF/download)
[FCOS_RT_MS_DLA_34_4x_shtw](configs/FCOS-Detection/FCOS_RT/MS_DLA_34_4x_syncbn_shared_towers.yaml) | 52 FPS | 39.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/4vc3XwQezyhNvnB/download)

More models can be found in FCOS [README.md](configs/FCOS-Detection/README.md).

### COCO Instance Segmentation Baselines with [BlendMask](https://arxiv.org/abs/2001.00309)

Model | Name |inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:
Mask R-CNN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) | 10 FPS | 42.9 | 38.6 |
BlendMask | [R_101_3x](configs/BlendMask/R_101_3x.yaml) | 11 FPS | 44.8 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/e4fXrliAcMtyEBy/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/R_101_dcni3_5x.yaml) | 10 FPS | 46.8 | 41.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download)

For more models and information, please refer to BlendMask [README.md](configs/BlendMask/README.md).

### COCO Instance Segmentation Baselines with [MEInst](https://arxiv.org/abs/2003.11712)

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[MEInst_R_50_3x](https://github.com/aim-uofa/AdelaiDet/configs/MEInst-InstanceSegmentation/MEInst_R_50_3x.yaml) | 12 FPS | 43.6 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/1ID0DeuI9JsFQoG/download)

For more models and information, please refer to MEInst [README.md](configs/MEInst-InstanceSegmentation/README.md).

### Total_Text results with [ABCNet](https://arxiv.org/abs/2002.10200)

Name | inf. time | e2e-hmean | det-hmean | download
---  |:---------:|:---------:|:---------:|:---:
[attn_R_50](configs/BAText/TotalText/attn_R_50.yaml) | 11 FPS | 67.1 | 86.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download)

For more models and information, please refer to ABCNet [README.md](configs/BAText/README.md).

### COCO Instance Segmentation Baselines with [CondInst](https://arxiv.org/abs/2003.05664)

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[CondInst_MS_R_50_1x](configs/CondInst/MS_R_50_1x.yaml) | 14 FPS | 39.7 | 35.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/Trx1r4tLJja7sLT/download)
[CondInst_MS_R_50_BiFPN_3x_sem](configs/CondInst/MS_R_50_BiFPN_3x_sem.yaml) | 13 FPS | 44.7 | 39.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/9cAHjZtdaAGnb2Q/download)
[CondInst_MS_R_101_3x](configs/CondInst/MS_R_101_3x.yaml) | 11 FPS | 43.3 | 38.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/vWLiYm8OnrTSUD2/download)
[CondInst_MS_R_101_BiFPN_3x_sem](configs/CondInst/MS_R_101_BiFPN_3x_sem.yaml) | 10 FPS | 45.7 | 40.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/2p1ashxl54Su8vv/download)

For more models and information, please refer to CondInst [README.md](configs/CondInst/README.md).

Note that:
- Inference time for all projects is measured on a NVIDIA 1080Ti with batch size 1.
- APs are evaluated on COCO2017 val split unless specified.


## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) for now. The incompatibility with the latest one will be fixed soon.

Then build AdelaiDet with:

```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```

If you are using docker, a pre-built image can be pulled with:

```
docker pull tianzhi0549/adet:latest
```

Some projects may require special setup, please follow their own `README.md` in [configs](configs).

## Quick Start

### Inference with Pre-trained Models

1. Pick a model and its config file, for example, `fcos_R_50_1x.yaml`.
2. Download the model `wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth`
3. Run the demo with
```
python demo/demo.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS fcos_R_50_1x.pth
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/fcos_R_50_1x
```
To evaluate the model after training, run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/fcos_R_50_1x \
    MODEL.WEIGHTS training_dir/fcos_R_50_1x/model_final.pth
```
Note that:
- The configs are made for 8-GPU training. To train on another number of GPUs, change the `--num-gpus`.
- If you want to measure the inference time, please change `--num-gpus` to 1.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
- This quick start is made for FCOS. If you are using other projects, please check the projects' own `README.md` in [configs](configs). 

## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

```BibTeX
@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}

@inproceedings{chen2020blendmask,
  title     =  {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{zhang2020MEInst,
  title     =  {Mask Encoding for Single Shot Instance Segmentation},
  author    =  {Zhang, Rufeng and Tian, Zhi and Shen, Chunhua and You, Mingyu and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}

@article{wang2020solov2,
  title   =  {{SOLOv2}: Dynamic, Faster and Stronger},
  author  =  {Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:2003.10152},
  year    =  {2020}
}

@article{tian2019directpose,
  title   =  {{DirectPose}: Direct End-to-End Multi-Person Pose Estimation},
  author  =  {Tian, Zhi and Chen, Hao and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1911.07451},
  year    =  {2019}
}

@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/).
