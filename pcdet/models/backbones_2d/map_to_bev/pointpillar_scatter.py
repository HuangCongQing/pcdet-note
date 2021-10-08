'''
Description: 将p转化为(h,w)此模块生成伪造图像，图像维度为(1,64,496,432)  https://blog.csdn.net/cg129054036/article/details/107371831#t5
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-08-03 09:26:49
LastEditTime: 2021-09-06 18:52:27
FilePath: /PCDet/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py
'''
import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # NUM_BEV_FEATURES: 64 #BEV特征数
        self.nx, self.ny, self.nz = grid_size # 网格  torch.Size([8932, 32, 10])
        print("grid_size ", grid_size)
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        # 获得pillar_features
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            #  Create the spatial_feature for this sample
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # # Now scatter the blob back to the canvas.
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature) # # Append to a list for later stacking.
        # 维度处理
        # # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # Undo the column stacking to final 4-dim tensor
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz,  self.ny, self.nx) # (batch_size，C，H，W)  torch.Size([3, 64, 496, 432])
        batch_dict['spatial_features'] = batch_spatial_features #=============================================================
        # print('batch_dict-------------------','\n',batch_dict) #===================================
        return batch_dict
""" 

一个 batch_dict：
'batch_size'：        3  # 3帧点云
'points'              torch.Size([69375, 5])  # 点数目可变 猜测(? x y z r)
'frame_id'            (3,)  # 帧编号
'gt_boxes'            torch.Size([3, 40, 8])  # 3帧点，好像每帧最多40个ground truth，[x, y, z, 长dx, 宽dy, 高dz, 角度heading, 标签？]
'use_lead_xyz'        torch.Size([3])  # ？
'voxels'              torch.Size([17842, 32, 4]) # 体素
'voxel_coords'        torch.Size([17842, 4]) # 体素坐标
'voxel_num_points'    torch.Size([17842])  # 体素点数
'image_shape'         (3, 2) # 图像尺寸
'pillar_features'     torch.Size([17842, 64])  # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个
'spatial_features'    torch.Size([3, 64, 496, 432])  # (batch_size，C，H，W)，其实C H W是固定的    此模块生成伪造图像，图像维度为(1,64,496,432)

 """