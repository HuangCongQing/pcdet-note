import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils

# 多标签分类损失函数使用的是focal loss  ref: https://blog.csdn.net/W1995S/article/details/114687437
# https://blog.csdn.net/W1995S/article/details/115400741
class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples. 用于平衡简单、困难样本
            alpha: Weighting parameter to balance loss for positive and negative examples. 用于平衡正、负样本
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod  # 代码自定义了tf.nn.sigmoid_cross_entropy_with_logits()交叉熵损失，这里以sigmoid函数做为logistic函数，所有输入sigmoid之前的函数都可以叫logit，这里是多输入，所以叫logits
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:输入
            input: (B, #anchors, #classes) float tensor.   [3, 321408, 3]
                Predicted logits for each class        为每一个类别预测的logits
            target: (B, #anchors, #classes) float tensor.   [3, 321408, 3]
                One-hot encoded classification targets   One-hot编码的类别标签

        Returns:输出
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction   Sigmoid交叉熵损失
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))  # 交叉熵公式计算===========================
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.  [3, 321408, 3]
                Predicted logits for each class           为每一个类别预测的logits
            target: (B, #anchors, #classes) float tensor. [3, 321408, 3]
                One-hot encoded classification targets    One-hot编码的类别标签，只有0和1，1大约有150±30个左右
            weights: (B, #anchors) float tensor.          [3, 321408]
                Anchor-wise weights.                      每一个anchor的权重？

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.  加权损失
        """
        pred_sigmoid = torch.sigmoid(input)    # [3, 321408, 3]

        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)  # 正样本(里面的1)->0.25，负样本(0)->0.75 【1:3】
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid    # 前半部分是正样本，后半部分是负样本
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)   # α * (1-pt)^2

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)  # 交叉熵 # 代码自定义了tf.nn.sigmoid_cross_entropy_with_logits()交叉熵损失，这里以sigmoid函数做为logistic函数，所有输入sigmoid之前的函数都可以叫logit，这里是多输入，所以叫logits

        loss = focal_weight * bce_loss # 

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1) # 权重  传入的weights参数对每个正anchor和负anchor进行了一个平均，使得每个样本的损失与样本中目标的数量无关。

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights # 

# smoothL1 loss  # 位置损失函数：加权 Smooth L1 Loss
class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta # 0.11111
        if code_weights is not None:   # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  7维
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:   # beta=0.11111 > 0.00001
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.     [3, 321408, 7]
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.    [3, 321408, 7]
                Regression targets.
            weights: (B, #anchors) float tensor if not None.  [3, 321408]

        Returns:
            loss: (B, #anchors) float tensor.              [3, 321408, 7]
                加权的 smooth l1 loss without reduction.
————————————————
版权声明：本文为CSDN博主「THE@JOKER」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/W1995S/article/details/115399145
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets   # 忽略 nan targets

        diff = input - target   # 差值(前6个值做差，最后一个角度编码值就差做差了)
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)   # [3, 321408, 7]

        # anchor-wise weighting
        if weights is not None:  # [3, 321408]
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)   # [3, 321408, 7] * [3, 321408, 1]

        return loss  # # [3, 321408, 7]


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

# 交叉熵损失
class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.   方向预测值 [3, 321408, 2]
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.  方向目标 [3, 321408, 2]
                One-hot classification targets.
            weights: (B, #anchors) float tensor.           权重参数 [3, 321408]
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)  # [3, 2, 321408]
        target = target.argmax(dim=-1) # [3, 321408]
        loss = F.cross_entropy(input, target, reduction='none') * weights    # 计算交叉熵损失
        return loss

# 角corner损失
def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)
