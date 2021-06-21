'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-06-20 16:45:12
FilePath: /PCDet/tools/demo.py
'''
import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

import sys
sys.path.append('..')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


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
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')# 1. '--cfg_file'   #指定配置
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory') # 2 . '--data_path'  #指定点云数据文件或目录
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model') #  3. '--ckpt'  #指定预训练模型
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file') # 4. '--ext'  #指定点云数据文件的扩展名

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg) # cfg的参数在tools/cfg/kitti_models/pv-rcnn.yaml

    return args, cfg # cfg的参数在tools/cfg/kitti_models/pv-rcnn.yaml


def main():
    # 1 输入参数
    args, cfg = parse_config() # cfg的参数在tools/cfg/kitti_models/pv-rcnn.yaml
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # 2 调用的这些包就是pcdet/models/detectors下的各个py文件，
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # 3 参数加载
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    # cuda( ) 和 eval( ) 都是数据处理
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}') # 样本数
            # 4. collate_batch
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict) # 传递数据给gpu的
            pred_dicts, _ = model.forward(data_dict) #  在神经网络中向前传递数据data_dict，得到预测数据pred_dicts     定位到forward，因为是PVRCNN类下的函数，先看__init__  /home/hcq/pointcloud/PCDet/pcdet/models/detectors/pv_rcnn.py
            # 可视化V
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

