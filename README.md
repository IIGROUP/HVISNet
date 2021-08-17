# HVISNet

Official PyTorch implementation and data for **Real-time Human-Centric Segmentation for Complex Video Scenes**.

[[Paper on arXiv](https://arxiv.org/abs/2108.07199)]

The code will be released soon.

<p align="center">
<img src="/docs/pipeline.jpg"/> 
</p>

## Abstract

Most existing video tasks related to "human" focus on the segmentation of salient humans, ignoring the unspecified others in the video. Few studies have focused on segmenting and tracking all humans in a complex video, including pedestrians and humans of other states (e.g., seated, riding, or occluded). In this paper, we propose a novel framework, abbreviated as HVISNet, that segments and tracks all presented people in given videos based on a one-stage detector. To better evaluate complex scenes, we offer a new benchmark called HVIS (Human Video Instance Segmentation), which comprises 1447 human instance masks in 805 high-resolution videos in diverse scenes. Extensive experiments show that our proposed HVISNet outperforms the state-of-the-art methods in terms of accuracy at a real-time inference speed (30 FPS), especially on complex video scenes. We also notice that using the center of the bounding box to distinguish different individuals severely deteriorates the segmentation accuracy, especially in heavily occluded conditions. This common phenomenon is referred to as the ambiguous positive samples problem. To alleviate this problem, we propose a mechanism named Inner Center Sampling to improve the accuracy of instance segmentation. Such a plug-and-play inner center sampling mechanism can be incorporated in any instance segmentation models based on a one-stage detector to improve the performance. In particular, it gains 4.1 mAP improvement on the state-of-the-art method in the case of occluded humans.

<p align="center">
<img src="/docs/results.jpg"/> 
</p>

## HVIS Dataset

We propose a new benchmark called **H**uman **V**ideo **I**nstance **S**egmentation (HVIS), which focuses on complex real-world scenarios with sufficient human instance masks and identities. Our dataset contains 805 videos with 1447 detailedly annotated human instances. It also includes various overlapping scenes, which integrates into the most challenging video dataset related to humans.

The dataset will be released soon.

## Citation

If you find our work helpful for your research, please consider to cite:

```bibtex
@article{yu2021hvis,
  title={Real-time Human-Centric Segmentation for Complex Video Scenes},
  author={Yu, Ran and Tian, Chenyu and Xia, Weihao and Zhao, Xinyuan and Wang, Haoqian and Yang, Yujiu},
  journal={arxiv preprint arxiv:2108.07199},
  year={2021}
}
```