<!-- <img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection.

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192).

--- -->


## Learning

Docs:[https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2](https://www.yuque.com/huangzhongqing/hre6tf/vy6gd2)

```
# pointpillars
python train.py --cfg_file=cfgs/kitti_models/pointpillar.yaml --batch_size=4 --epochs=10

```

tensorrt部署参考：
* https://github.com/hova88/OpenPCDet
* https://github.com/hova88/PointPillars_MultiHead_40FPS

### 


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