'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2022-07-26 14:20:24
FilePath: /PCDet/pcdet/models/__init__.py
'''
from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

# 这里的build_networks是继承的Detector3DTemplate中的函数
def build_network(model_cfg, num_class, dataset):
    model = build_detector( # build_network函数内只有一个build_detector函数，build_detector的定来源：   pcdet/models/detectors/__init__.py
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

# 加载数据到gpu
def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray): #  #判断一个对象是否是一个已知的类型，ndarray对象是用于存放同类型元素的多维数组
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict) # 得到返回loss, tb_dict, disp_dict==============================================================

        loss = ret_dict['loss'].mean() # 获取loss的均值
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict) # 最终返回loss, tb_dict, disp_dict

    return model_func
