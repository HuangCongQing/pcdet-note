import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
# 3dssd
from ..ops.roiaware_pool3d import roiaware_pool3d_utils

# 多标签分类损失函数使用的是focal loss  ref: https://blog.csdn.net/W1995S/article/details/114687437
# https://blog.csdn.net/W1995S/article/details/115400741
# focal loss 分类损失
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

# smoothL1 loss  # 位置回归损失函数：加权 Smooth L1 Loss
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
        self.code_weights = code_weights # 3dssd修改
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

# 权重交叉熵损失(朝向dir损失)
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


# 3dssd
class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').mean(dim=-1) * weights
        return loss



# SASA loss计算
class PointSASALoss(nn.Module):
    """
    Layer-wise point segmentation loss, used for SASA.
    """
    def __init__(self,
                 func: str = 'BCE',
                 layer_weights: list = None,
                 extra_width: list = None,
                 set_ignore_flag: bool = False):
        super(PointSASALoss, self).__init__()

        self.layer_weights = layer_weights
        if func == 'BCE':
            self.loss_func = WeightedBinaryCrossEntropyLoss()
        elif func == 'Focal':
            self.loss_func = SigmoidFocalClassificationLoss()
        else:
            raise NotImplementedError

        assert not set_ignore_flag or (set_ignore_flag and extra_width is not None)
        self.extra_width = extra_width
        self.set_ignore_flag = set_ignore_flag
    
    def assign_target(self, points, gt_boxes):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, \
            'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if self.extra_width is not None else gt_boxes
        
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())

            if not self.set_ignore_flag:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

            else:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                ignore_flag = box_fg_flag ^ (extend_box_idx_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            point_cls_labels_single[box_fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single
        
        return point_cls_labels # (N, ) 0: bg, 1: fg, -1: ignore

    def forward(self, l_points, l_scores, gt_boxes):
        """
        Args:
            l_points: List of points, [(N, 4): bs_idx, x, y, z]
            l_scores: List of points, [(N, 1): predicted point scores]
            gt_boxes: (B, M, 8)
        Returns:
            l_labels: List of labels: [(N, 1): assigned segmentation labels]
        """
        l_labels = []
        for i in range(len(self.layer_weights)):
            li_scores = l_scores[i]
            if li_scores is None or self.layer_weights[i] == 0:
                l_labels.append(None)
                continue
            # binary segmentation labels: 0: bg, 1: fg, -1: ignore
            li_labels = self.assign_target(l_points[i], gt_boxes)
            l_labels.append(li_labels)

        return l_labels

    def loss_forward(self, l_scores, l_labels):
        """
        Args:
            l_scores: List of points, [(N, 1): predicted point scores]
            l_labels: List of points, [(N, 1): assigned segmentation labels]
        Returns:
            l_loss: List of segmentation loss
        """
        l_loss = []
        for i in range(len(self.layer_weights)):
            li_scores, li_labels = l_scores[i], l_labels[i]
            if li_scores is None or li_labels is None:
                l_loss.append(None)
                continue

            positives, negatives = li_labels > 0, li_labels == 0
            cls_weights = positives * 1.0 + negatives * 1.0 # (N, 1)
            pos_normalizer = cls_weights.sum(dim=0).float()

            one_hot_targets = li_scores.new_zeros(
                *list(li_labels.shape), 2
            )
            one_hot_targets.scatter_(-1, (li_labels > 0).long().unsqueeze(-1), 1.0)
            one_hot_targets = one_hot_targets[:, 1:] # (N, 1)

            li_loss = self.loss_func(li_scores[None],
                                     one_hot_targets[None],
                                     cls_weights.reshape(1, -1))
            li_loss = self.layer_weights[i] * li_loss.sum() / torch.clamp(
                pos_normalizer, min=1.0)
            l_loss.append(li_loss)

        return l_loss


# centerpoint



def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss