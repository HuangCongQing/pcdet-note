import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict
    # #------------------------------------ 计算 分类 loss ----------------------------—----
    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']  # # 类别预测结果 [N, H, W, C1] = [3, 248, 216, 18]                                                       来源: pcdet/models/dense_heads/anchor_head_single.py
        box_cls_labels = self.forward_ret_dict['box_cls_labels']  # box类别标签 [N, num_anchors]= [3, 321408]
        """ 
        torch.Size([3, 248, 216, 18]) # 类别预测结果
        torch.Size([3, 321408]) # box类别标签

        数字分析：
        C1通道：18 = 3 x 3 x 2，3帧点云，每一帧点云在backbon得到的feature map 上每一个位置生成2个anchor，每个anchor预测Car、Pedestrian、Cyclists 这 3个类别。
        num_anchors ： 321408 = 107136 x 3，3帧点云，每一帧点云生成107136 = 248 x 216 x 2个anchor。
        ————————————————
        版权声明：本文为CSDN博主「THE@JOKER」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/W1995S/article/details/115400741

         """
        batch_size = int(cls_preds.shape[0]) # 批数 3
        cared = box_cls_labels >= 0  # [N, num_anchors]  ，我们只关心>=0标签的box
        positives = box_cls_labels > 0 # 标签>0 正样本
        negatives = box_cls_labels == 0 # 标签=0 负样本
        negative_cls_weights = negatives * 1.0 # 负样本类别权重，0变1
        cls_weights = (negative_cls_weights + 1.0 * positives).float()  # 正负样本全部变1、2，3
        reg_weights = positives.float() # 只保留标签>0的值
        if self.num_class == 1: # # 如果只预测一类，如Car，实际上==3
            # class agnostic  方式只回归2类bounding box，即前景和背景，仅检测“前景”物体
            box_cls_labels[positives] = 1  # 检测一类，故将正样本标签全部置1

        pos_normalizer = positives.sum(1, keepdim=True).float()  # 行和，维度保留[3,1]，根据统计，每行都是100以上的值=1x?+2x?+3x?
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) # 最小1.0 ,很像归一化, 仅是正样本
        cls_weights /= torch.clamp(pos_normalizer, min=1.0) # 正负样本归一化？

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)  # 只保留标签>0的类别
        cls_targets = cls_targets.unsqueeze(dim=-1) # 增加1个维度 [3, 321408, 1]
        cls_targets = cls_targets.squeeze(dim=-1) # 又减少一个维度???!!!  [3, 321408]

        one_hot_targets = torch.zeros( # [3, 321408, 4]
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)    # 转为one-hot类型
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)  # [3, 321408, 3]
        one_hot_targets = one_hot_targets[..., 1:]  # [3, 321408, 3]  只有0和1，1大约有150±30个左右
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']  # 损失权重，cls_weight：1.0
        # tensorboard显示
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets
    # ----------------------------------- 计算box 位置 loss  https://blog.csdn.net/W1995S/article/details/115399145 --------------------------------
    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']   # 位置预测结果 [N, H, W, C2] = [3, 248, 216, 42] # 来源:     self.forward_ret_dict['cls_preds'] = cls_preds  pcdet/models/dense_heads/anchor_head_single.py
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)  # 方向、类别预测结果 [N, H, W, C3] = [3,248,216,12]  12??????????????
        box_reg_targets = self.forward_ret_dict['box_reg_targets'] # box回归目标 [3, 321408, 7]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']  # box类别标签 [3, 321408]   (num_anchors ： 321408 = 107136 x 3，3帧点云(batch_size)，每一帧点云生成107136 = 248 x 216 x 2个anchor。)
        """ 
        torch.Size([3, 248, 216, 42]) # 位置预测结果 [N, H, W, C2]
        torch.Size([3, 248, 216, 12]) # 方向预测结果 [N, H, W, C3]
        torch.Size([3, 321408, 7]) # box回归目标
        torch.Size([3, 321408]) # box类别标签

        数字解释：
            42：3帧点云，每一帧每个位置有14个预测值，14 = 7 x 2 ，2个anchor，7个回归坐标。
            12：3帧点云，每一帧每个位置预测2个anchor，2个方向？
        ————————————————
        版权声明：本文为CSDN博主「THE@JOKER」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/W1995S/article/details/115399145
        
         """
        batch_size = int(box_preds.shape[0]) # 

        positives = box_cls_labels > 0 # 正样本标签
        reg_weights = positives.float() # 只保留标签>0的值
        pos_normalizer = positives.sum(1, keepdim=True).float()  # 行和，[3,1]，根据统计，每行都是100以上的值=1x?+2x?+3x?
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) # 最小1.0 ,很像归一化, 仅是正样本

        if isinstance(self.anchors, list):
            if self.use_multihead:   # False
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors   # [1, 248, 216, 3, 2, 7]
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)  # [3, 321408, 7]==========================================================
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])   # [3, 321408, 7]
        # sin(a - b) = sinacosb-cosasinb    仅计算最后一个维度
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)  # 均 [3, 321408, 7]

        # 调用 WeightedSmoothL1Loss 计算位置损失函数
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']   # 损失权重，loc_weight: 2.0
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }
        # 获取方向目标。=================================================
        if box_dir_cls_preds is not None:  # [3,248,216,12]  方向、类别预测结果 [N, H, W, C3] = [3,248,216,12] 
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,   # 均 [3, 321408, 7]
                dir_offset=self.model_cfg.DIR_OFFSET,   # 方向偏移量 0.78539 = π/4
                num_bins=self.model_cfg.NUM_DIR_BINS # BINS的方向数 = 2
            )  # [3, 321408, 2]
            # 方向损失函数：计算方向损失
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)   # 方向预测值 [3, 321408, 2]
            weights = positives.type_as(dir_logits)   # 只要正样本的方向预测值
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)  # [3, 321408]
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']  # 损失权重，dir_weight: 0.2
            # 总的位置损失==============================
            box_loss += dir_loss    # 位置损失 + 方向(c朝向角)损失
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict
    # -------------------------------------- 获得损失 --------------------------------------
    def get_loss(self): #  pv_rcnn.py引用   loss_rpn, tb_dict = self.dense_head.get_loss() #
        cls_loss, tb_dict = self.get_cls_layer_loss()   # 计算classification loss    函数来源于下面   def get_cls_layer_loss(self):
        box_loss, tb_dict_box = self.get_box_reg_layer_loss() # 计算box regression loss
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss #  区域提案RPN region proposal loss 是以上两个loss的和=============================
        print("rpn_loss:", rpn_loss)

        tb_dict['rpn_loss'] = rpn_loss.item()# rpn损失
        return rpn_loss, tb_dict # 

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
