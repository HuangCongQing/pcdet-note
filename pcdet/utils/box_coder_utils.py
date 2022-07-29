import numpy as np
import torch

# 参考:https://blog.csdn.net/W1995S/article/details/115136348
# https://blog.csdn.net/qq_41366026/article/details/123006401#t13
# 残差编码器
class ResidualCoder(object):
    """
    loss中anchor和gt的编码与解码
    7个参数的编码的方式为
        ∆x = (x^gt − xa^da)/d^a , ∆y = (y^gt − ya^da)/d^a , ∆z = (z^gt − za^ha)/h^a
        ∆w = log (w^gt / w^a) ∆l = log (l^gt / l^a) , ∆h = log (h^gt / h^a)
        ∆θ = sin(θ^gt - θ^a)
    """
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    # 编码
    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        # 截断anchors的[dx,dy,dz]，每个anchor_box的l, w, h数值如果小于1e-5则为1e-5
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        # 截断boxes的[dx,dy,dz] 每个GT_box的l, w, h数值如果小于1e-5则为1e-5
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)  split_size是切分后每块的大小，不是切分为多少块！，多余的参数使用*cags接收
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        # 计算anchor对角线长度
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        # 计算loss的公式，Δx,Δy,Δz,Δw,Δl,Δh,Δθ
        # ∆x = x ^ gt − xa ^ da
        xt = (xg - xa) / diagonal
        # ∆y = (y^gt − ya^da)/d^a
        yt = (yg - ya) / diagonal
        # ∆z = (z^gt − za^ha)/h^a
        zt = (zg - za) / dza
        # ∆l = log(l ^ gt / l ^ a)
        dxt = torch.log(dxg / dxa)
        # ∆w = log(w ^ gt / w ^ a)
        dyt = torch.log(dyg / dya)
        # ∆h = log(h ^ gt / h ^ a)
        dzt = torch.log(dzg / dza)
        # False
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra] # Δθ 角度

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    # 解码（编码的逆操作）
    # 7回归参数 (x, y, z, w, l, h, θ)；x, y,  z预测了目标中心点到该anchor左上顶点的偏移数值， w，l，h预测了基于该anchor长宽高的调整系数，
    # θ预测了box的旋转角度，
    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)  split_size是切分后每块的大小，不是切分为多少块！，多余的参数使用*cags接收
        # 预测的结果
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        # 分割编码后的box PointPillar为False
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        
        # 计算anchor对角线长度
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)  # (B, N, 1)-->(1, 321408, 1)
        # loss计算中anchor与GT编码的运算:g表示gt，a表示anchor
        # ∆x = (x^gt − xa^da)/diagonal --> x^gt = ∆x * diagonal + x^da
        # 下同
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            # rg = torch.atan2(rg_sin, rg_cos)
            rg = torch.atan(rg_sin / (rg_cos + 1e-6))
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

# 以前的残差解码器
class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

# 以前的残差RoI解码器
class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

# 点残差编码器
class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    # anchor_free解码
    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


# 3dssd用的这个编码解码   在yaml有参数配置
class PointBinResidualCoder(object):
    def __init__(self, code_size=30, use_mean_size=True, angle_bin_num=12, pred_velo=False, **kwargs):
        super().__init__()
        self.code_size = 6 + 2 * angle_bin_num
        self.angle_bin_num = angle_bin_num
        self.pred_velo = pred_velo
        if pred_velo:
            self.code_size += 2
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_angle_torch(self, angle):
        """
        Args:
            angle: (N)
        Returns:
            angle_cls: (N, angle_bin_num)
            angle_res: (N, angle_bin_num)
        """
        angle = torch.remainder(angle, np.pi * 2.0)
        angle_per_class = np.pi * 2.0 / float(self.angle_bin_num)
        shifted_angle = torch.remainder(angle + angle_per_class / 2.0, np.pi * 2.0)

        angle_cls_f = (shifted_angle / angle_per_class).floor()
        angle_cls = angle_cls_f.new_zeros(*list(angle_cls_f.shape), self.angle_bin_num)
        angle_cls.scatter_(-1, angle_cls_f.unsqueeze(-1).long(), 1.0)

        angle_res = shifted_angle - (angle_cls_f * angle_per_class + angle_per_class / 2.0)
        angle_res = angle_res / angle_per_class  # normalize residual angle to [0, 1]
        angle_res = angle_cls * angle_res.unsqueeze(-1)
        return angle_cls, angle_res

    def decode_angle_torch(self, angle_cls, angle_res):
        """
        Args:
            angle_cls: (N, angle_bin_num)
            angle_res: (N, angle_bin_num)
        Returns:
            angle: (N)
        """
        angle_cls_idx = angle_cls.argmax(dim=-1)
        angle_cls_onehot = angle_cls.new_zeros(angle_cls.shape)
        angle_cls_onehot.scatter_(-1, angle_cls_idx.unsqueeze(-1), 1.0)

        angle_res = (angle_cls_onehot * angle_res).sum(dim=-1)
        angle = (angle_cls_idx.float() + angle_res) * (np.pi * 2.0 / float(self.angle_bin_num))
        return angle
    # 编码
    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 6 + 2 * B + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        rg_cls, rg_reg = self.encode_angle_torch(rg.squeeze(-1))
        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, rg_cls, rg_reg, *cts], dim=-1)

    def decode_torch_kernel(self, box_offsets, box_angle_cls, box_angle_reg, points, pred_classes=None):
        """
        Args:
            box_offsets: (N, 6) [x, y, z, dx, dy, dz]
            box_angle_cls: (N, angle_bin_num)
            box_angle_reg: (N, angle_bin_num)
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
            boxes3d: (N, 7)
        """
        xt, yt, zt, dxt, dyt, dzt = torch.split(box_offsets, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg = torch.exp(dxt)
            dyg = torch.exp(dyt)
            dzg = torch.exp(dzt)

        rg = self.decode_angle_torch(box_angle_cls, box_angle_reg).unsqueeze(-1)
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)

    # anchor_free解码  实际输入: box_encodings(256,30) 输入角度是cos, sin
    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]  实际输入: box_encodings(256,30)
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
            boxes3d: (N, 7)
        """
        box_offsets = box_encodings[:, :6] # 偏移x, y, z, dx, dy, dz
        box_angle_cls = box_encodings[:, 6:6 + self.angle_bin_num] # 角度
        box_angle_reg = box_encodings[:, 6 + self.angle_bin_num:6 + self.angle_bin_num * 2]
        cgs = box_encodings[:, 6 + self.angle_bin_num * 2:]

        boxes3d = self.decode_torch_kernel(box_offsets, box_angle_cls, box_angle_reg, points, pred_classes) # 解码decode_torch_kernel
        return torch.cat([boxes3d, cgs], dim=-1)
