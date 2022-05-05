'''
Description:  SSD 的目标检测头？？
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-08-03 09:26:49
LastEditTime: 2022-05-05 12:58:16
FilePath: /PCDet/pcdet/models/dense_heads/anchor_head_single.py
'''
import numpy as np
import torch.nn as nn
#  分类
from .anchor_head_template import AnchorHeadTemplate

# input_channels：384 ， 
class AnchorHeadSingle(AnchorHeadTemplate):
    """
    Args:
        model_cfg: AnchorHeadSingle的配置
        input_channels: 384 输入通道数
        num_class: 3
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (432, 496, 1)
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1)
        predict_boxes_when_training: False
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        # 每个点有3个尺度的个先验框  每个先验框都有两个方向（0度，90度） num_anchors_per_location:[2, 2, 2]
        self.num_anchors_per_location = sum(self.num_anchors_per_location) # sum([2, 2, 2]) = 6
        # 分类卷积nn.Conv2d(384, 6*3=18)
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # 回归卷积nn.Conv2d(384, 6*7=42)
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, # 6*7=42  其中 box_coder.code_size,=7
            kernel_size=1
        )
        # 朝向卷积nn.Conv2d(384, 6*2=14)
        # 如果存在方向损失，则添加方向卷积层Conv2d(512,12,kernel_size=(1,1),stride=(1,1))====================
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,  #  NUM_DIR_BINS: 2 #BINS的方向数
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    # 初始化参数
    def init_weights(self):
        pi = 0.01
        # 初始化分类卷积偏置
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # 初始化分类卷积权重
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # 2D卷积backbone后 (batch_size，6C，H/2，W/2)
        spatial_features_2d = data_dict['spatial_features_2d'] # torch.Size([3, 384, 248, 216]) # 2D卷积，上下采样连接后的 (batch_size，6C，H/2，W/2)
        # 分类2D卷积：# 每个坐标点上面6个先验框的类别预测 --> (batch_size, 18, 200, 176)
        cls_preds = self.conv_cls(spatial_features_2d)  # 2D卷积，进行类别预测   (batch_size，6C，H/2，W/2)
        # print("cls_preds.shape:", cls_preds.shape) # torch.Size([1, 18(6*3), 248, 216])
        # 回归2D卷积:每个坐标点上面6个先验框的参数预测 --> (batch_size, 42, 200, 176)  其中每个先验框需要预测7个参数，分别是（x, y, z, w, l, h, θ）
        box_preds = self.conv_box(spatial_features_2d) # 2D卷积， 进行位置预测  torch.Size([3, 248, 216, 42]) # 位置预测结果 [N, H, W, C2]   3帧点云，每一帧每个位置有14个预测值，14 = 7 x 2 ，2个anchor，7个回归坐标。
        # 42：3帧点云，每一帧每个位置有14个预测值，14 = 7 x 2 ，2个anchor，7个回归坐标。

        # 维度调整，将类别放置在最后一维度   [N, H, W, C] --> (batch_size, 200, 176, 18
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] # C放在最后面
        # 维度调整，将先验框调整参数放置在最后一维度   [N, H, W, C] --> (batch_size ,200, 176, 42)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] 3帧点云，每一帧每个位置有14个预测值，14 = 7 x 2 ，2个anchor，7个回归坐标。
        # 42：3帧点云，每一帧每个座标点有14个预测值，14 = 7 x 2 ，2个anchor，7个回归坐标。
        
        # 将类别和先验框调整预测结果放入前向传播字典中
        self.forward_ret_dict['cls_preds'] = cls_preds  # 类别预测结果  pcdet/models/dense_heads/anchor_head_template.py会用到 cls_preds = self.forward_ret_dict['cls_preds']  
        self.forward_ret_dict['box_preds'] = box_preds # 位置预测结果  

        # 进行方向分类预测
        if self.conv_dir_cls is not None:
            # 每个先验框都要预测为两个方向中的其中一个方向 --> (batch_size, 12, 200, 176)
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) #   # torch.Size([1, 12(6*2), 248, 216])
            # 将类别和先验框方向预测结果放到最后一个维度中   [N, H, W, C] --> (batch_size, 248, 216, 12)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds # torch.Size([3, 248, 216, 12]) # 方向预测结果 [N, H, W, C3]   12：3帧点云，每一帧每个位置预测2个anchor，2个方向？
        else:
            dir_cls_preds = None

        """
        如果是在训练模式的时候，需要对每个先验框分配GT来计算loss
        """
        if self.training: # 训练=============================================================
            # anchor生成和GT匹配（编码操作！！！！！！）
            # pcdet/models/dense_heads/target_assigner/axis_aligned_target_assigner.py
            targets_dict = self.assign_targets(  # 计算IOU===============anchor_head_template.py》》pcdet/models/dense_heads/target_assigner/axis_aligned_target_assigner.py
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict) #  update() 方法向字典插入指定的项目。
            # targets_dict = {
            #     'box_cls_labels': cls_labels, # (4，211200）
            #     'box_reg_targets': bbox_targets, # (4，211200, 7）  已经编码后的结果！！！！！！！
            #     'reg_weights': reg_weights # (4，211200）
            # }
            
        
        # 如果不是训练模式，则直接生成进行box的预测
        if not self.training or self.predict_boxes_when_training: # 预测=======================================================================
            # 根据预测结果解码生成最终结果
            # 计算出每一个先验框的偏差和中心点的偏移量。通过解码得到真实框的中心和和宽高【解码操作】
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes( # 生成预测框【解码操作】
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds # torch.Size([3, 321408, 3])  # 类别预测
            data_dict['batch_box_preds'] = batch_box_preds #   torch.Size([3, 321408, 7])  # box预测
            data_dict['cls_preds_normalized'] = False

        return data_dict
''' 

    定位：pcdet/models/dense_heads/anchor_head_single.py   # 预测类别、box
    一个 data_dict：
    'batch_size'：        3  # 3帧点云
    'points'              torch.Size([69375, 5])  # 点数目可变 猜测(? x y z r)
    'frame_id'            (3,)  # 帧编号
    'gt_boxes'            torch.Size([3, 40, 8])  # 3帧点，好像每帧最多40个ground truth，[x, y, z, 长dx, 宽dy, 高dz, 角度heading, 标签？]
    'use_lead_xyz'        torch.Size([3])  # ？
    'voxels'              torch.Size([17842, 32, 4]) # 体素
    'voxel_coords'        torch.Size([17842, 4]) # 体素坐标
    'voxel_num_points'    torch.Size([17842])  # 体素点数
    'image_shape'         (3, 2) # 图像尺寸

    'pillar_features'     torch.Size([17842, 64])  # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个 【C是原来D(9)维处理后的维度】   【对N进行Max Pooling操作,去掉N】
    'spatial_features'    torch.Size([3, 64, 496, 432])  # (batch_size，C，H，W)，其实C H W是固定的

    'spatial_features_2d' torch.Size([3, 384, 248, 216]) # 2D卷积，上下采样连接后的 (batch_size，6C，H/2，W/2) 
    
    'calib'               (3,)  # 相机、雷达、惯导等传感器的矫正数据
    'road_plane'          torch.Size([3, 4])  # planes文件夹里002188/002193/002196.txt 标签数据
    'batch_cls_preds'     torch.Size([3, 321408, 3])  # 类别预测========================================================
    'batch_box_preds'     torch.Size([3, 321408, 7])  # box预测======================================================
    'cls_preds_normalized'  False  # 类别预测是否归一化
 '''