'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-07-20 20:34:59
LastEditTime: 2021-09-06 15:51:47
FilePath: /PCDet/pcdet/models/backbones_3d/vfe/vfe_template.py
'''
import torch.nn as nn


class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C) # 多一个vfe特征 =============================================================
        """
        raise NotImplementedError
