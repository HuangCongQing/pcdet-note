from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz: (B, npoint, 3) tensor of the xyz coordinates of the grouping centers if specified
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint) # PFS最远点采样
            ).transpose(1, 2).contiguous() if self.npoint is not None else None
        
        # 修改
        for i in range(len(self.groupers)):

            # new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float()
            idx_cnt_mask = idx_cnt_mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            new_features *= idx_cnt_mask
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )

# 下面被for循环执行3次：for i in range(len(self.SA_modules))
# 3dssd Base  ( ['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']参数中的一个)
class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None # 需要初始化处理
        self.mlps = None # k=0: [16, 16, 32], [16, 16, 32], [32, 32, 64]]
        self.npoint_list = [] # NPOINT_LIST: [[4096], [512, 512], [256, 256]]
        self.sample_range_list = [[0, -1]] # SAMPLE_RANGE_LIST: [[[0, 16384]], [[0, 4096], [0, 4096]], [[0, 512], [512, 1024]]]
        self.sample_method_list = ['d-fps'] # 方法列表 # [['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']]
        self.radii = [] # [[0.2, 0.4, 0.8], [0.4, 0.8, 1.6], [1.6, 3.2, 4.8]] # 感受野？？？？

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None # AGGREGATION_MLPS: [[64], [128], [256]] # 聚合MLPS
        self.confidence_mlp = None # CONFIDENCE_MLPS: [[32], [64], []] # 基于3dssd新添加的 # 置信度？？？

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps====================================================
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        sample_sfps = None # 初始化

        xyz_flipped = xyz.transpose(1, 2).contiguous() # 
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            # 遍历方法列表参数中的一个 ： ['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']参数中的一个
            for i in range(len(self.sample_method_list)):
                #  SAMPLE_RANGE_LIST:    [[[0, 16384]],  [[0, 4096], [0, 4096]], [[0, 512], [512, 1024]]]
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous() #
                # print(xyz_slice.shape) #(1,16384,3),   [(1,4096,3),(1,4096,3)], [(1,512,3),(1,512,3)]
                if self.sample_method_list[i] == 'd-fps':
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i]) # NPOINT_LIST: [[4096],   [512, 512],   [256, 256]]
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]] # 得到范围
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i]) # 得到采样下标
                # ============================================s-fps实现
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma # 权重(1, 4096),(1, 512)
                    sample_idx = pointnet2_utils.furthest_point_sample_weights( # (1, 512)(1, 256),pcdet/ops/pointnet2/pointnet2_batch/pointnet2_utils.py
                        xyz_slice,   # (1, 4096,3),(1, 512,3)
                        scores_slice, # (1, 4096),(1, 512)
                        self.npoint_list[i] # NPOINT_LIST: [[4096], [512, 512], [256, 256]]
                    )
                    # add vis sfps
                    sample_sfps = torch.cat([xyz_slice.reshape(-1, 3), scores_slice.reshape(-1,1)],dim=-1 ) # torch.Size([4096, 4])
                    # print(sample_idx.shape)
                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0]) # 得到下标列表    SAMPLE_RANGE_LIST: [[[0, 16384]], [[0, 4096], [0, 4096]], [[0, 512], [512, 1024]]]
                # print(sample_idx_list) # 一共输出3次  第一次：list[1:tensor(1,4096)]到下面  第二次：list[2:tensor(1,512)，tensor(1,512)] 第三次：list[2:tensor(1,256)，tensor(1,256)]
            # for  end

            sample_idx = torch.cat(sample_idx_list, dim=-1) # list 一共3次
            # 第一次：list[1:tensor(1,4096)]
            # 第二次：(1,1024) = list[2:tensor(1,512)+tensor(1,512)]
            # 第三次：(1,512) = list[2:tensor(1,256)+tensor(1,256)]

            # 根据下标得到新的xyz点云
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3) #
            # 第一次：(1,4096,3)
            # 第二次：(1,1024,3)
            # 第三次：(1,512,3)
            
            if self.skip_connection: 
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        # 1groupers+2mlp
        for i in range(len(self.groupers)):
            #  (1 groupers): ModuleList
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            #  (2 mlps): ModuleList
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)  k=0: [16, 16, 32], [16, 16, 32], [32, 32, 64]]
            idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            new_features *= idx_cnt_mask

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        # False
        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        #  (3 aggregation_mlp): Sequential
        new_features = torch.cat(new_features_list, dim=1)
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features) # 输入是new_features

        #   (4 confidence_mlp): Sequential CONFIDENCE_MLPS: [[32], [64], []] # 基于3dssd新添加的 # 置信度？？？
        if self.confidence_mlp is not None:
            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_scores, sample_sfps  #  在这返回值=============================================================================

        return new_xyz, new_features, None, sample_sfps # (1,512,3), (1,256,512) 因为CONFIDENCE_MLPS参数最后一次为None 在这运行

# 被调用pcdet/models/dense_heads/point_head_vote.py
# SAMPLE_METHOD_LIST: [['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']]
# 流程：  (0): PointnetSAModuleFSMSG(   groupers， mlps，aggregation_mlp，confidence_mlp
# 被调用3次 forward没有变，只是修改了初始化__init__ 包括：step1-4！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,# [k] NPOINT_LIST: [[4096], [512, 512], [256, 256]]
                 sample_range_list: List[List[int]] = None,   #[k]  #SAMPLE_RANGE_LIST: [[[0, 16384]], [[0, 4096], [0, 4096]], [[0, 512], [512, 1024]]]
                 sample_method_list: List[str] = None, # [k] 方法参数配置 [['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']]
                 radii: List[float],#  [[0.2, 0.4, 0.8], [0.4, 0.8, 1.6], [1.6, 3.2, 4.8]] # 感受野？？？？
                 nsamples: List[int],# NSAMPLE: [[32, 32, 64], [32, 32, 64], [32, 32, 64]] # 采样点数？？
                 mlps: List[List[int]], # 其中一层的[16, 16, 32], [16, 16, 32], [32, 32, 64],  [[64, 64, 128], [64, 64, 128], [64, 96, 128]]   [[128, 128, 256], [128, 196, 256], [128, 256, 256]]]
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None, # 传参 AGGREGATION_MLPS: [[64], [128], [256]] # 聚合MLPS
                 confidence_mlp: List[int] = None): # CONFIDENCE_MLPS: [[32], [64], []] # 基于3dssd新添加的 # 置信度？？？
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList() # step1 =======
        self.mlps = nn.ModuleList() # step2 =========

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        # loop3次 step1 ： self.groupers
        for i in range(len(radii)):
            radius = radii[i] # RADIUS: [[0.2, 0.4, 0.8], [0.4, 0.8, 1.6], [1.6, 3.2, 4.8]] # 感受野？？？？
            nsample = nsamples[i] # NSAMPLE: [[32, 32, 64], [32, 32, 64], [32, 32, 64]] # 采样点数？？
            if dilated_radius_group: # True
                self.groupers.append( #   (1 groupers): ModuleList
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz) # pcdet/ops/pointnet2/pointnet2_batch/pointnet2_utils.py
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                )
            former_radius = radius
            # step2 ： self.mlps
            mlp_spec = mlps[i] #  k=0: [[1, 16, 16, 32], [1, 16, 16, 32], [1, 32, 32, 64]] # 添加一维输入channel_in
            if use_xyz:
                mlp_spec[0] += 3 # 4 [[4, 16, 16, 32], [4, 16, 16, 32], [4, 32, 32, 64]]

            shared_mlp = []
            for k in range(len(mlp_spec) - 1): # [4, 16, 16, 32]
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), # 参数
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlp)) #  (2 mlps): ModuleList
            in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        # step3 AGGREGATION_MLPS: [[64], [128], [256]] # 聚合MLPS
        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        # step4 CONFIDENCE_MLPS: [[32], [64], []] # 基于3dssd新添加的 # 置信度？？？
        if confidence_mlp is not None:
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            # 添加nn.Conv1d
            shared_mlp.append(
                nn.Conv1d(out_channels, 1, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
        else:
            self.confidence_mlp = None

# Base 上面class    pcdet/models/dense_heads/point_head_vote.py
class PointnetSAModuleFS(PointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=[radius], nsamples=[nsample],
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp
        )

# 最简单的FPS？？？
class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another将一个集合的特征传递给另一个集合"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

if __name__ == "__main__":
    pass
