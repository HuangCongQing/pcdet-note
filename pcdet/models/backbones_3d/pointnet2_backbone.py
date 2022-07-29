import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

# 当前只有pointRCNN使用了
class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList() # PointnetSAModuleMSG
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList() # PointnetFPModule pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py

        # for k in range(self.model_cfg.FP_MLPS.__len__()):
        #     pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
        #     self.FP_modules.append(
        #         pointnet2_modules.PointnetFPModule(
        #             mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
        #         )
        #     )

        skip_sa_block = self.model_cfg.SA_CONFIG.NPOINTS.__len__() - self.model_cfg.FP_MLPS.__len__()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule( # # PointnetFPModule pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py
                    mlp=[pre_channel + skip_channel_list[k + skip_sa_block]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        # features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict

# 3dssd的BACKBONE_3D》》》》》》》》》》》》》》》》》》》main
class PointNet2FSMSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        use_xyz = self.model_cfg.SA_CONFIG.get('USE_XYZ', True)
        dilated_group = self.model_cfg.SA_CONFIG.get('DILATED_RADIUS_GROUP', False)
        skip_connection = self.model_cfg.SA_CONFIG.get('SKIP_CONNECTION', False)
        weight_gamma = self.model_cfg.SA_CONFIG.get('WEIGHT_GAMMA', 1.0)

        self.aggregation_mlps = self.model_cfg.SA_CONFIG.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = self.model_cfg.SA_CONFIG.get('CONFIDENCE_MLPS', None)

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        # for循环参数3：NPOINT_LIST: [[4096], [512, 512], [256, 256]]
        for k in range(self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            if skip_connection:
                channel_out += channel_in

            if self.aggregation_mlps and self.aggregation_mlps[k]:
                aggregation_mlp = self.aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if self.confidence_mlps and self.confidence_mlps[k]:
                confidence_mlp = self.confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None
            
            # ====================================================
            # 在for循环里面哦[k]  初始化，下面调用
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleFSMSG( # pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py
                    npoint_list=self.model_cfg.SA_CONFIG.NPOINT_LIST[k], # NPOINT_LIST: [[4096], [512, 512], [256, 256]]
                    sample_range_list=self.model_cfg.SA_CONFIG.SAMPLE_RANGE_LIST[k], # SAMPLE_RANGE_LIST: [[[0, 16384]], [[0, 4096], [0, 4096]], [[0, 512], [512, 1024]]]
                    sample_method_list=self.model_cfg.SA_CONFIG.SAMPLE_METHOD_LIST[k], # 方法参数配置 [['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']]
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    dilated_radius_group=dilated_group,
                    skip_connection=skip_connection,
                    weight_gamma=weight_gamma,
                    aggregation_mlp=aggregation_mlp, # 传参
                    confidence_mlp=confidence_mlp # 传参
                )
            )
            self.num_points_each_layer.append(
                sum(self.model_cfg.SA_CONFIG.NPOINT_LIST[k]))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        
        self.num_point_features = channel_out

        fp_mlps = self.model_cfg.get('FP_MLPS', None)
        if fp_mlps is not None:
            self.FP_modules = nn.ModuleList()
            l_skipped = self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__() - self.model_cfg.FP_MLPS.__len__()
            for k in range(fp_mlps.__len__()):
                pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else channel_out
                self.FP_modules.append(
                    pointnet2_modules.PointnetFPModule( # pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py
                        mlp=[pre_channel + skip_channel_list[k + l_skipped]] + fp_mlps[k]
                    )
                )
            self.num_point_features = fp_mlps[0][-1]
        else:
            self.FP_modules = None

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None) # intensity？
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_coords: (N, 3)            # 
                point_features: (N, C)          # 
                point_confidence_scores: (N, 1) # 
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # 原始点(16384, 5) 第一列没用，暂时还不知道第一列是啥~
        # tensor([[0.0000, 6.7280, 1.4690, -1.6452, 0.1900],
        #         [0.0000, 39.1460, 30.0250, 0.1300, 0.0000],
        #         [0.0000, 8.2680, -1.5030, -1.6800, 0.2900],
        #         ...,
        #         [0.0000, 6.9450, 4.8700, -1.6170, 0.2700],
        #         [0.0000, 11.0010, -7.9530, -0.8880, 0.5800],
        #         [0.0000, 6.6070, -0.7850, -1.6850, 0.2700]], device='cuda:0')
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3).contiguous()
        features = features.view(batch_size, -1, features.shape[-1]) if features is not None else None # (batch_size, 16384, 1) intensity
        features = features.permute(0, 2, 1).contiguous() if features is not None else None # (batch_size, 1, 16384) contiguous 一般用于 transpose/permute 后和 view 前

        batch_idx = batch_idx.view(batch_size, -1).float() # (1,16384)

        l_xyz, l_features, l_scores = [xyz], [features], [None] # (B, 16384,3)# (16384,1) (16384,1)
        l_sample_sfps = []
        for i in range(len(self.SA_modules)): # PointnetSAModuleFSMSG( # pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py
            li_xyz, li_features, li_scores, li_sample_sfps = self.SA_modules[i]( #  [['d-fps'], ['s-fps', 'd-fps'], ['s-fps', 'd-fps']]
                l_xyz[i], l_features[i], scores=l_scores[i])
            l_xyz.append(li_xyz) #
            l_features.append(li_features)
            l_scores.append(li_scores)
            l_sample_sfps.append(li_sample_sfps)
        # print("l_xyz，l_features，l_scores, l_sample_sfps: ",l_xyz, l_features, l_scores, l_sample_sfps) # list:4  (因为自身初始就有一个自身值)
        # 得到l_xyz，l_features，l_scores
        # l_xyz[tensor(1,16384,3),     tensor(1,4096,3),  tensor(1,1024,3),   tensor(1,512,3)]
        # l_features[tensor(1,1,16384),tensor(1,64,4096), tensor(1,128,1024), tensor(1,256,512)]
        # l_scores[      None,         tensor(1,4096),    tensor(1,1024),     None]
        # ======================================================================================================

        # 添加可视化
        batch_dict['sample_sfps'] = l_sample_sfps

        # prepare for confidence loss
        l_xyz_flatten, l_scores_flatten = [], []
        # 1 xyz
        for i in range(1, len(l_xyz)): # 循环下标 1,2,3
            l_xyz_flatten.append(torch.cat([
                batch_idx[:, :l_xyz[i].size(1)].reshape(-1, 1), # 1 列
                l_xyz[i].reshape(-1, 3)                        #3列
            ], dim=1))  # (N, 4)
        # 2 score
        for i in range(1, len(l_scores)):
            if l_scores[i] is None:
                l_scores_flatten.append(None)
            else:
                l_scores_flatten.append(l_scores[i].reshape(-1, 1))  # (N, 1)
        batch_dict['point_coords_list'] = l_xyz_flatten # {list:3} （4096，4），（1024，4），（512，4）
        batch_dict['point_scores_list'] = l_scores_flatten #  {list:3} (4096,1),(1024,1),(512,1)

        # 3 feature
        if self.FP_modules is not None:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i]( # pointnet2_modules.PointnetFPModule
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )  # (B, C, N)
        else:  # take l_xyz[i - 1] and l_features[i - 1]
            i = 0

        point_features = l_features[i - 1].permute(0, 2, 1).contiguous()  # (B, N, C) (1,512,256)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (512,256)
        # 候选点~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        batch_dict['point_coords'] = torch.cat((
            batch_idx[:, :l_xyz[i - 1].size(1)].reshape(-1, 1).float(),  # (512, 4)
            l_xyz[i - 1].view(-1, 3)), dim=1)
        batch_dict['point_scores'] = l_scores[-1]  # (B, N)     l_scores[None,tensor(1,4096),tensor(1,1024),None]
        # print('[PointNet2FSMSG]batch_dict-------------------','\n',batch_dict) #===================================================
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict
