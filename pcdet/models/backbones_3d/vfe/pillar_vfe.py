import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer. # 这是一个简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络，只有一层，代码如下：
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results
        only used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.      
        :param out_channels: <int>. Number of output channels.    
        :param use_norm: <bool>. Whether to include BatchNorm.    
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        # 这是一个简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络，只有一层，代码如下：
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2 # " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。

        if self.use_norm: # se_norm=True,
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

# pillars==========================================
# 下面我们将实现PillarFeatureNetOld2类，这里的作用是生成一个个Pillar，并将点云原来的4维特征( x , y , z , r ) 扩充为10维特征 (x,y,z,r, x_c,y_c,z_c,x_p,y_p,z_p)
class PillarVFE(VFETemplate): # 继承：vfe_template.py
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        # Create PillarFeatureNetOld layers
        self.num_filters = self.model_cfg.NUM_FILTERS # NUM_FILTERS: [64]   #滤波器个数
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1): # 40层滤波器
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)) # 简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) # 得到pfn_layers

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # 64

    # https://blog.csdn.net/cg129054036/article/details/107371831#t4
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Create boolean mask by actually number of a padded tensor.
        Args:
            actual_num ([type]): [description]
            max_num ([type]): [description]
        Returns:
            [type]: [description]
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # print('actual_num shape is: ', actual_num.shape)   
        # tiled_actual_num: [N, M, 1]

        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        return paddings_indicator

    # 开始==============================================================================
    def forward(self, batch_dict, **kwargs):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param coors: (z ,y, x)
        :return:
        """
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # 算pillar算术均值   Find distance of x, y, and z from cluster center (x, y, z mean)   Pillar中所有点的几何中心；
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center  点与几何中心的相对位置。
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            # # Combine together feature decorations
            features = [voxel_features, f_cluster, f_center]  # 扩充为10维特征 (x,y,z,r, x_c,y_c,z_c,x_p,y_p,z_p)
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:    # False
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty.  特征装饰的计算与pillar是否空无关
        # Need to ensure that empty pillars remain set to zeros. 需要确保空pillar保持为零。
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        # print('161 features shape is: ', features.shape)  
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features #  torch.Size([17842, 64])  # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个
        # print('batch_dict-------------------','\n',batch_dict) #=====================================================================================
        return batch_dict
''' 
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
'pillar_features'     torch.Size([17842, 64])  # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个  【C是原来D(9)维处理后的维度64】   【对N进行Max Pooling操作,去掉N】=====================================================

 '''