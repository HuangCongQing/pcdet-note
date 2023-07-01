<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-08-05 22:17:19
 * @LastEditTime: 2022-09-10 11:19:34
 * @FilePath: /PCDet/README.md
-->
<!-- <img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet( v0.5.0)

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection.

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192).

--- -->


## pcdet-Learning( based v0.5.0)

Docs:[https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2](https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2)

自动驾驶相关交流群，欢迎扫码加入：[自动驾驶感知(PCL/ROS+DL)：技术交流群汇总(新版)](https://mp.weixin.qq.com/s?__biz=MzI4OTY1MjA3Mg==&mid=2247486575&idx=1&sn=3145b7a5e9dda45595e1b51aa7e45171&chksm=ec2aa068db5d297efec6ba982d6a73d2170ef09a01130b7f44819b01de46b30f13644347dbf2#rd)

创建一个知识星球 **【自动驾驶感知(PCL/ROS+DL)】** 专注于自动驾驶感知领域，包括传统方法(PCL点云库,ROS)和深度学习（目标检测+语义分割）方法。同时涉及Apollo，Autoware(基于ros2)，BEV感知，三维重建，SLAM(视觉+激光雷达) ，模型压缩（蒸馏+剪枝+量化等），自动驾驶模拟仿真，自动驾驶数据集标注&数据闭环等自动驾驶全栈技术，欢迎扫码二维码加入，一起登顶自动驾驶的高峰！
![image](https://github.com/HuangCongQing/HuangCongQing/assets/20675770/304e0c4d-89d2-4cee-a2a9-3c690611c9d9)


**TODO:**

  - [x] [【202212done】目标检测最新论文实时更新](https://zhuanlan.zhihu.com/p/591349104)
  - [ ] [【202304done】语义分割最新论文实时更新](https://zhuanlan.zhihu.com/p/591349481)
  - [x] [【202209done】目标检测框架(pcdet+mmdetection3d+det3d+paddle3d)文章撰写](https://zhuanlan.zhihu.com/p/569189196?)
  - [ ] 数据集详细剖析：kitti&waymo&nuScenes
  - [ ] Apollo学习https://github.com/HuangCongQing/apollo_note



代码注解

* config yaml配置文件注释(eg.pointpillar.yaml):[tools/cfgs/kitti_models/pointpillar.yaml](tools/cfgs/kitti_models/pointpillar.yaml)

* kitti评测详细介绍（可适配自己的数据集评测):[pcdet/datasets/kitti/kitti_object_eval_python](pcdet/datasets/kitti/kitti_object_eval_python)

* 模型配置注释：[tools/cfgs/kitti_models/pointpillar.yaml](tools/cfgs/kitti_models/pointpillar.yaml)

* 数据集配置注释：[tools/cfgs/dataset_configs/kitti_dataset.yaml](tools/cfgs/dataset_configs/kitti_dataset.yaml)

* [...](./)


**其他目标检测框架(pcdet+mmdetection3d+det3d+paddle3d)代码注解笔记：**

1. pcdet:https://github.com/HuangCongQing/pcdet-note
2. mmdetection3d:https://github.com/HuangCongQing/mmdetection3d-note
3. det3d: TODO
4. paddle3dL TODO


### 运行

```
# pointpillars
python train.py --cfg_file=cfgs/kitti_models/pointpillar.yaml --batch_size=4 --epochs=10

```

tensorrt部署参考：
* https://github.com/hova88/OpenPCDet
* https://github.com/hova88/PointPillars_MultiHead_40FPS

### 
Install `pcdet` toolbox.
```shell
pip install -r requirements.txt
python setup.py develop
```


```
# pointpillars
python train.py --cfg_file=cfgs/kitti_models/pointpillar.yaml --batch_size=4 --epochs=10

# centerpoint
train.py --cfg_file cfgs/kitti_models/centerpoint.yaml --batch_size 4 --epoch 100
## 报错 RuntimeError: CUDA error: out of memory
train.py --cfg_file cfgs/kitti_models/centerpoint_pillar.yaml --batch_size 4 --epoch 100
python demo.py --cfg_file cfgs/kitti_models/centerpoints.yaml --ckpt ../checkpoints/centerpoint_kitti_80.pth --data_path ../testing/velodyne/000003.bin

```

<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection.

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192).


Docs:https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2



## Overview

- [Changelog](#changelog)
- [Design Pattern](#openpcdet-design-pattern)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)

## Changelog

[2020-11-27] **Bugfixed:** Please re-prepare the validation infos of Waymo dataset (version 1.2) if you would like to
use our provided Waymo evaluation tool (see [PR](https://github.com/open-mmlab/OpenPCDet/pull/383)).
Note that you do not need to re-prepare the training data and ground-truth database.

[2020-11-10] **NEW:** The [Waymo Open Dataset](#waymo-open-dataset-baselines) has been supported with state-of-the-art results. Currently we provide the
configs and results of `SECOND`, `PartA2` and `PV-RCNN` on the Waymo Open Dataset, and more models could be easily supported by modifying their dataset configs.

[2020-08-10] Bugfixed: The provided NuScenes models have been updated to fix the loading bugs. Please redownload it if you need to use the pretrained NuScenes models.

[2020-07-30] `OpenPCDet` v0.3.0 is released with the following features:

* The Point-based and Anchor-Free models ([`PointRCNN`](#KITTI-3D-Object-Detection-Baselines), [`PartA2-Free`](#KITTI-3D-Object-Detection-Baselines)) are supported now.
* The NuScenes dataset is supported with strong baseline results ([`SECOND-MultiHead (CBGS)`](#NuScenes-3D-Object-Detection-Baselines) and [`PointPillar-MultiHead`](#NuScenes-3D-Object-Detection-Baselines)).
* High efficiency than last version, support **PyTorch 1.1~1.7** and **spconv 1.0~1.2** simultaneously.

[2020-07-17]  Add simple visualization codes and a quick demo to test with custom data.

[2020-06-24] `OpenPCDet` v0.2.0 is released with pretty new structures to support more models and datasets.

[2020-03-16] `OpenPCDet` v0.1.0 is released.

## Introduction

### What does `OpenPCDet` toolbox do?

Note that we have upgrated `PCDet` from `v0.1` to `v0.2` with pretty new structures to support various datasets and models.

`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud.
It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

Based on `OpenPCDet` toolbox, we win the Waymo Open Dataset challenge in [3D Detection](https://waymo.com/open/challenges/3d-detection/),
[3D Tracking](https://waymo.com/open/challenges/3d-tracking/), [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/)
three tracks among all LiDAR-only methods, and the Waymo related models will be released to `OpenPCDet` soon.

We are actively updating this repo currently, and more datasets and models will be supported soon.
Contributions are also welcomed.

### `OpenPCDet` design pattern

* Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:

<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* Unified 3D box definition: (x, y, z, dx, dy, dz, heading).
* Flexible and clear model structure to easily support various 3D detection models:

<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* Support various models within one framework as:

<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>

### Currently Supported Features

- [X] Support both one-stage and two-stage 3D object detection frameworks
- [X] Support distributed training & testing with multiple GPUs and multiple machines
- [X] Support multiple heads on different scales to detect different classes
- [X] Support stacked version set abstraction to encode various number of points in different scenes
- [X] Support Adaptive Training Sample Selection (ATSS) for target assignment
- [X] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [X] Support GPU version 3D IoU calculation and rotated NMS

## Model Zoo

### KITTI 3D Object Detection Baselines

Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.

* All models are trained with 8 GTX 1080Ti GPUs and are available for download.
* The training time is measured with 8 TITAN XP GPUs and PyTorch 1.5.


|   | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11 | download |
| - | -: | :-: | :-: | :-: | :-: |
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) | ~1.2 hours | 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) |
| [SECOND](tools/cfgs/kitti_models/second.yaml) | ~1.7 hours | 78.62 | 52.98 | 67.15 | [model-20M](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing) |
| [PointRCNN](tools/cfgs/kitti_models/pointrcnn.yaml) | ~3 hours | 78.70 | 54.41 | 72.11 | [model-16M](https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing) |
| [PointRCNN-IoU](tools/cfgs/kitti_models/pointrcnn_iou.yaml) | ~3 hours | 78.75 | 58.32 | 71.34 | [model-16M](https://drive.google.com/file/d/1V0vNZ3lAHpEEt0MlT80eL2f41K2tHm_D/view?usp=sharing) |
| [Part-A^2-Free](tools/cfgs/kitti_models/PartA2_free.yaml) | ~3.8 hours | 78.72 | 65.99 | 74.29 | [model-226M](https://drive.google.com/file/d/1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr/view?usp=sharing) |
| [Part-A^2-Anchor](tools/cfgs/kitti_models/PartA2.yaml) | ~4.3 hours | 79.40 | 60.05 | 69.90 | [model-244M](https://drive.google.com/file/d/10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | ~5 hours | 83.61 | 57.90 | 70.47 | [model-50M](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |

### NuScenes 3D Object Detection Baselines

All models are trained with 8 GTX 1080Ti GPUs and are available for download.


|   | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | download |
| - | -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [PointPillar-MultiHead](tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml) | 33.87 | 26.00 | 32.07 | 28.74 | 20.15 | 44.63 | 58.23 | [model-23M](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing) |
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml) | 31.15 | 25.51 | 26.64 | 26.26 | 20.46 | 50.59 | 62.29 | [model-35M](https://drive.google.com/file/d/1bNzcOnE3u9iooBFMk2xK7HqhdeQ_nwTq/view?usp=sharing) |

### Waymo Open Dataset Baselines

We provide the setting of [`DATA_CONFIG.SAMPLED_INTERVAL`](tools/cfgs/dataset_configs/waymo_dataset.yaml) on the Waymo Open Dataset (WOD) to subsample partial samples for training and evaluation,
so you could also play with WOD by setting a smaller `DATA_CONFIG.SAMPLED_INTERVAL` even if you only have limited GPU resources.

By default, all models are trained with **20% data (~32k frames)** of all the training samples on 8 GTX 1080Ti GPUs, and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).


|   | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
| - | -: | :-: | :-: | :-: | :-: | :-: |
| [SECOND](tools/cfgs/waymo_models/second.yaml) | 68.03/67.44 | 59.57/59.04 | 61.14/50.33 | 53.00/43.56 | 54.66/53.31 | 52.67/51.37 |
| [Part-A^2-Anchor](tools/cfgs/waymo_models/PartA2.yaml) | 71.82/71.29 | 64.33/63.82 | 63.15/54.96 | 54.24/47.11 | 65.23/63.92 | 62.61/61.35 |
| [PV-RCNN](tools/cfgs/waymo_models/pv_rcnn.yaml) | 74.06/73.38 | 64.99/64.38 | 62.66/52.68 | 53.80/45.14 | 63.32/61.71 | 60.72/59.18 |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/),
but you could easily achieve similar performance by training with the default configs.

### Other datasets

More datasets are on the way.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Quick Demo

Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods.
We would like to thank for their proposed methods and the official implementation.

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.

## Citation

If you find this project useful in your research, please consider cite:

```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## Contribution

Welcome to be a member of the OpenPCDet development team by contributing to this repo, and feel free to contact us for any potential contributions.
