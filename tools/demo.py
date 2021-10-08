'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-09-05 23:46:20
FilePath: /PCDet/tools/demo.py
'''
import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab  # 可视化
import numpy as np
import torch

import sys
sys.path.append('..')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate # 数据模板
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V # 可视化函数库

# 加载数据(from: pcdet/datasets/dataset.py)
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext # 文件后缀bin   txt  pcd,
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)
    
    def __getitem__(self, index):
        if self.ext == '.bin': # 判断什么文件类型
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

# 参数配置
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')# 1. '--cfg_file'   #指定配置
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory') # 2 . '--data_path'  #指定点云数据文件或目录
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model') #  3. '--ckpt'  #指定预训练模型
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file') # 4. '--ext'  #指定点云数据文件的扩展名

    args = parser.parse_args()
    # 加载yaml文件
    cfg_from_yaml_file(args.cfg_file, cfg) # cfg的参数在tools/cfg/kitti_models/pv-rcnn.yaml

    return args, cfg # cfg的参数在tools/cfg/kitti_models/pv-rcnn.yaml


def main():
    # 1 输入参数
    args, cfg = parse_config() # args: 就是文件后缀   cfg: 的参数在tools/cfg/kitti_models/pv-rcnn.yaml
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset( # 加载数据 返回值
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,   # cfg.CLASS_NAMES : ['Car', 'Pedestrian', 'Cyclist']
        root_path=Path(args.data_path), ext=args.ext, logger=logger # args.data_path：  '/root/PCDet/data/kitti/training/velodyne/000008.bin'
    )
    logger.info(f'【demo.py】Total number of samples: \t{len(demo_dataset)}') # 文件数量

    # 2 调用的这些包就是pcdet/models/detectors下的某个py文件，# class PVRCNN(Detector3DTemplate):
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset) #  # model_cfg.NAME:   'PVRCNN'      class PVRCNN(Detector3DTemplate):
    # model是实例化的class类！！！！model 对应  pcdet/models/detectors/pv_rcnn.py  对应参数在pcdet/models/detectors/__init__.py
    # 3 参数加载(.pth文件) 举例:'../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_1.pth'
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True) 
    # cuda( ) 和 eval( ) 都是数据处理
    model.cuda()
    model.eval() # 测试

    # torch.no_grad( ) 的目的是使得其中的数据不需要计算梯度，也不会进行反向传播
    with torch.no_grad():
        # 多少个bin文件，就遍历几次
        for idx, data_dict in enumerate(demo_dataset): # demo_dataset = DemoDataset( # 加载数据---  class DemoDataset(DatasetTemplate): ---class DatasetTemplate(torch_data.Dataset):
            # data_dict{dict: 6}： 'points'(60270,4) 'voxels'' voxel_num_points'
            logger.info(f'Visualized sample index: \t{idx + 1}') # 样本数
            # 4. collate_batch
            data_dict = demo_dataset.collate_batch([data_dict]) # 外面加了一个[] 来源：pcdet/datasets/dataset.py        def collate_batch(batch_list, _unused=False):
            load_data_to_gpu(data_dict) # 传递数据给gpu的
            pred_dicts, _ = model.forward(data_dict) # # data_dict: {dict: 6}  class PVRCNN(Detector3DTemplate):！！！！！  在神经网络中向前传递数据data_dict，得到预测数据pred_dicts     定位到forward，因为是PVRCNN类下的函数，先看__init__  /home/hcq/pointcloud/PCDet/pcdet/models/detectors/pv_rcnn.py
            # 可视化V
            V.draw_scenes( # 点，(真值框), 预测框，预测socre，预测label
                points=data_dict['points'][:, 1:],  gt_boxes=None, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True) # 调用可视化mayavi

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

