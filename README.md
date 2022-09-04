<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-08-05 22:17:19
 * @LastEditTime: 2022-09-04 15:05:38
 * @FilePath: /PCDet/README.md
-->
<!-- <img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection.

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192).

--- -->


## Learning

Docs:[https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2](https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2)

TODO：

*  目标检测框架(pcdet+mmdetection3d+det3d+paddle3d)文章撰写

代码注解

* [config yaml配置文件注释(eg.pointpillar.yaml)](tools/cfgs/kitti_models/pointpillar.yaml)

* [kitti评测详细介绍（可适配自己的数据集评测）](pcdet/datasets/kitti/kitti_object_eval_python)

* [...]()


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


### Branch

* master：no change
* hcq：learning【当前分支】
* test: trpd

### Tips
* 模型配置注释：[tools/cfgs/kitti_models/pointpillar.yaml](tools/cfgs/kitti_models/pointpillar.yaml)
* 数据集配置注释：[tools/cfgs/dataset_configs/kitti_dataset.yaml](tools/cfgs/dataset_configs/kitti_dataset.yaml)