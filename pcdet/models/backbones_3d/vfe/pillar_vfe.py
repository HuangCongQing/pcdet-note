import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

# 被下面调用（简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络）
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
            # 根据论文中，这是是简化版pointnet网络层的初始化
            # 论文中使用的是 1x1 的卷积层完成这里的升维操作（理论上使用卷积的计算速度会更快）
            # 输入的通道数是刚刚经过数据增强过后的点云特征，每个点云有10个特征，
            # 输出的通道数是64
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # 一维BN层
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
            # x的维度由（M, 32, 10）升维成了（M, 32, 64）
            x = self.linear(inputs) # 线性层====================================================================
        torch.backends.cudnn.enabled = False
        # BatchNorm1d层:(M, 64, 32) --> (M, 32, 64)
        # （pillars,num_point,channel）->(pillars,channel,num_points)
        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x  # BN层
        torch.backends.cudnn.enabled = True
        x = F.relu(x) # 激活函数
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        # x_max shape ：（M, 1, 64） 32个点变成1个点
        x_max = torch.max(x, dim=1, keepdim=True)[0] # 最大值

        if self.last_vfe:
            # 返回经过简化版pointnet处理pillar的结果
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

# pillars==========================================
# 下面我们将实现PillarFeatureNetOld2类，这里的作用是生成一个个Pillar，并将点云原来的4维特征( x , y , z , r ) 扩充为10维特征 (x,y,z,r, x_c,y_c,z_c,x_p,y_p,z_p)
class PillarVFE(VFETemplate): # 继承：vfe_template.py
    """
    model_cfg:NAME: PillarVFE
                    WITH_DISTANCE: False
                    USE_ABSLOTE_XYZ: True
                    USE_NORM: True
                    NUM_FILTERS: [64]
    num_point_features:4
    voxel_size:[0.16 0.16 4]
    POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]
    """
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
                # class PFNLayer(nn.Module):具体实现在上面
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)) # 简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络
            )
        # 加入线性层，将10维特征变为64维特征
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
    # 表明一个pillar中哪些是真实数据，哪些是填充的0数据
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        计算padding的指示 Create boolean mask by actually number of a padded tensor.
        Args:
            actual_num ([type]): [description] actual_num:每个voxel实际点的数量（M，）
            max_num ([type]): [description] max_num:voxel最大点的数量（32，）
        Returns:
            [type]: [description] paddings_indicator:表明一个pillar中哪些是真实数据，哪些是填充的0数据
        """
        # 扩展一个维度，使变为（M，1） torch.Size([25584, 1])
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # print('actual_num shape is: ', actual_num.shape)   
        # tiled_actual_num: [N, M, 1]
        # [1, 1]
        max_num_shape = [1] * len(actual_num.shape)
        # [1, -1]
        max_num_shape[axis + 1] = -1
        # (1,32)
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        # (M, 32)  True or False
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
        """
        batch_dict:
            points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
            frame_id:(4,) --> (003877,001908,006616,005355) 帧ID
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels:(M,32,4) --> (x,y,z,r)
            voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index============================
            voxel_num_points:(M,)
            image_shape:(4,2) 每份点云数据对应的2号相机图片分辨率
            batch_size:4    batch_size大小
        """
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # 求每个pillar中所有点云的和 (M, 32, 3)->(M, 1, 3) 设置keepdim=True的，则保留原来的维度信息
        # 然后在使用求和信息除以每个点云中有多少个点来求每个pillar中所有点云的平均值 points_mean shape：(M, 1, 3)

        # 算pillar算术均值   Find distance of x, y, and z from cluster center (x, y, z mean)   Pillar中所有点的几何中心；
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        # 每个点云数据减去该点对应pillar的平均值得到差值 xc,yc,zc===============================================================
        f_cluster = voxel_features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center  点与几何中心的相对位置。
        #  创建每个点云到该pillar的坐标中心点偏移量空数据 xp,yp,zp
        f_center = torch.zeros_like(voxel_features[:, :, :3])

        #  coords是每个网格点的坐标，即[432, 496, 1]，需要乘以每个pillar的长宽得到点云数据中实际的长宽（单位米）
        #  同时为了获得每个pillar的中心点坐标，还需要加上每个pillar长宽的一半得到中心点坐标
        #  每个点的x、y、z减去对应pillar的坐标中心点，得到每个点到该点中心点的偏移量===================================================
        # coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset) # 获得每个pillar的中心点坐标，还需要加上每个pillar长宽的一半得到中心点坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        # 此处偏移多了z轴偏移  论文中没有z轴偏移
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # 如果使用绝对坐标，直接组合
        if self.use_absolute_xyz:
            # Combine together feature decorations
            features = [voxel_features, f_cluster, f_center]  # 扩充为10维特征 (x,y,z,r, x_c,y_c,z_c,x_p,y_p,z_p)
        # 否则，取voxel_features的3维之后，在组合
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        # 如果使用距离信息
        if self.with_distance:    # False
            # torch.norm的第一个2指的是求2范数，第二个2是在第三维度求范数
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        # 将特征在最后一维度拼接 得到维度为（M，32,10）的张量
        features = torch.cat(features, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty.  特征装饰的计算与pillar是否空无关
        # Need to ensure that empty pillars remain set to zeros. 需要确保空pillar保持为零。
        # 每个pillar中点云的最大数量
        voxel_count = features.shape[1]

        """
        由于在生成每个pillar中，不满足最大32个点的pillar会存在由0填充的数据，
        而刚才上面的计算中，会导致这些
        由0填充的数据在计算出现xc,yc,zc和xp,yp,zp出现数值，
        所以需要将这个被填充的数据的这些数值清0,
        因此使用get_paddings_indicator计算features中哪些是需要被保留真实数据和需要被置0的填充数据
        """
        # 得到mask维度是（M， 32）
        # mask中指名了每个pillar中哪些是需要被保留的数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # （M， 32）->(M, 32, 1)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        #  将feature中被填充为0的数据的所有特征置0
        features *= mask

        # print('161 features shape is: ', features.shape)  
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        # (M, 64), 每个pillar抽象出一个64维特征
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