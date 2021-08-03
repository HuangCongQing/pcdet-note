'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-07-26 16:35:12
FilePath: /PCDet/tools/demo_npy_mayavi.py
'''
import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
# import torch

import sys
sys.path.append('..')
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

data_dict = np.load("/home/hcq/下载/npy/data_dict.npy", allow_pickle=True)
pred_boxes = np.load("/home/hcq/下载/npy/pred_boxes.npy", allow_pickle=True)
pred_labels = np.load("/home/hcq/下载/npy/pred_labels.npy", allow_pickle=True)
pred_scores = np.load("/home/hcq/下载/npy/pred_scores.npy", allow_pickle=True)
# data_dict = np.load("/home/hcq/下载/npy_epoch_40/data_dict.npy", allow_pickle=True)
# pred_boxes = np.load("/home/hcq/下载/npy_epoch_40/pred_boxes.npy", allow_pickle=True)
# pred_labels = np.load("/home/hcq/下载/npy_epoch_40/pred_labels.npy", allow_pickle=True)
# pred_scores = np.load("/home/hcq/下载/npy_epoch_40/pred_scores.npy", allow_pickle=True)
print(pred_scores.shape)
V.draw_scenes(
    points=data_dict, ref_boxes=pred_boxes,
    ref_scores=pred_scores, ref_labels=pred_labels
)
mlab.show(stop=True)



