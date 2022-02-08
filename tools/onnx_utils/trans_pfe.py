import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.part = 50000

    def forward(self, inputs):
        # nn.Linear performs randomly when batch size is too large
        x = self.linear(inputs)

        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        
        # if self.with_distance:
        #     num_point_features += 1
        self.num_point_features = num_point_features
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, features, **kwargs):
  
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features[:,0,:]
        return features


def build_pfe(ckpt,cfg):

    pfe =PillarVFE(            
                model_cfg=cfg.MODEL.VFE,
                num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,  
                voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)  

    pfe.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "vfe" in key:
            dicts[key[4:]] = checkpoint["model_state"][key]
    pfe.load_state_dict(dicts)

    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
    max_points_per_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
    dims_feature = pfe.num_point_features
    dummy_input = torch.ones(max_num_pillars,max_points_per_pillars,dims_feature).cuda()
    return pfe , dummy_input 

if __name__ == "__main__":
    from pcdet.config import cfg, cfg_from_yaml_file
    cfg_file = '/path/to/cbgs_pp_multihead.yaml'
    filename_mh = "/path/to/pp_multihead_nds5823_updated.pth"
    cfg_from_yaml_file(cfg_file, cfg)
    model_cfg=cfg.MODEL
    pfe , dummy_input  = build_pfe( filename_mh, cfg)
    pfe.eval().cuda()
    export_onnx_file = "/path/to/cbgs_pp_multihead_pfe.onnx"
    torch.onnx.export(pfe,
                    dummy_input,
                    export_onnx_file,
                    opset_version=12,
                    verbose=True,
                    do_constant_folding=True) # 输出名