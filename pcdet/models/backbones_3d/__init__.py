'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-03-17 17:38:26
LastEditTime: 2022-07-23 22:48:54
FilePath: /PCDet/pcdet/models/backbones_3d/__init__.py
'''
from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG, PointNet2FSMSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'PointNet2FSMSG': PointNet2FSMSG, # 3DSSD
    'VoxelResBackBone8x': VoxelResBackBone8x,
}
