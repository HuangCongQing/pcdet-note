'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-08-07 15:36:10
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
# data_dict = data_dict[:, 1:4]
# np.transpose(,[1,2,0])
# np.hstack([data_dict[:,1:3],data_dict[:,0:1]]).shape
# data_dict = np.hstack( [          data_dict[:,2:3] , data_dict[:,0:1],  data_dict[:,1:2]               ])
pred_boxes = np.load("/home/hcq/下载/npy/pred_boxes.npy", allow_pickle=True)
# pred_boxes = pred_boxes*100
# pred_boxes[:, 1] = pred_boxes[:, 1]-  40.0
# pred_boxes[:, 2] = pred_boxes[:, 2]-  3
# pred_boxes = pred_boxes*100
pred_labels = np.load("/home/hcq/下载/npy/pred_labels.npy", allow_pickle=True)
pred_scores = np.load("/home/hcq/下载/npy/pred_scores.npy", allow_pickle=True)
# data_dict = np.load("/home/hcq/下载/npy_epoch_40/data_dict.npy", allow_pickle=True)
# pred_boxes = np.load("/home/hcq/下载/npy_epoch_40/pred_boxes.npy", allow_pickle=True)
# pred_labels = np.load("/home/hcq/下载/npy_epoch_40/pred_labels.npy", allow_pickle=True)
# pred_scores = np.load("/home/hcq/下载/npy_epoch_40/pred_scores.npy", allow_pickle=True)

# GT
# rcnn_box = np.load("/home/hcq/下载/npy/rcnn_box.npy", allow_pickle=True)
# gt_boxes = np.load("/home/hcq/下载/npy/gt_boxes.npy", allow_pickle=True)
# rcnn_box = rcnn_box*100  
# POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] 
# rela_dis = [70.4, 80.0, 4.0] 
# pred_boxes[:,0] = pred_boxes[:, 0]*70.4
# pred_boxes[:,1] = pred_boxes[:, 1]* 80.0- 40
# pred_boxes[:,2] = pred_boxes[:, 2]* 4.0  -3
# for i  in range(3,6):
#     pred_boxes[:,i]  = pred_boxes[:,i] * rela_dis[i-3]

print('Number of label:{}'.format(len(pred_labels))) #
labels = ['None', 'Car', 'Pedestrian', 'Cyclists']
for i in range(4):
    print('\t {}; {}'.format((i, labels[i]), (pred_labels==i).sum()))

# gt_boxes[:,0] = gt_boxes[:, 0]*70.4
# gt_boxes[:,1] = gt_boxes[:, 1]* 80.0
# gt_boxes[:,2] = gt_boxes[:, 2]* 4.0

# print(pred_scores.shape)
V.draw_scenes(
    points=data_dict, ref_boxes=pred_boxes,
    ref_scores=pred_scores, ref_labels=pred_labels
)
# print('gt_boxes:' , gt_boxes[0:1,:])
# print('rcnn_box:' , rcnn_box[0:1,:])
# gt_boxes[0:1,:] =  [[11.358741, 10.752694, -0.90895593, 1.9995697, 0.5947438, 1.8765192,-1.3389492 ]]
# rcnn_box[0:1,:] =  [[11.274641, 10.491711,-1.219522, 1.3724763,0.7367202,2.000592,-0.21755815]]
# V.draw_scenes(
# points=data_dict, gt_boxes =gt_boxes[0:1,:]*8, ref_boxes=rcnn_box[0:1,:]*8,
#     ref_scores=None, ref_labels=None
# )
mlab.show(stop=True)



