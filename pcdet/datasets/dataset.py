from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor # 数据增强类
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder # 
# 参考：https://blog.csdn.net/weixin_41286628/article/details/115795114?spm=1001.2014.3001.5501
""" DatasetTemplate类继承了torch的Dataset类 """
class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg # # 传入dataset的config字典
        self.training = training # bool值
        self.class_names = class_names # ['Car', 'Pedestrian', 'Cyclist']
        self.logger = logger
        # 读取字典中DATA_PATH的值作为数据集的根目录，返回Path()对象
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH) # bin文件路径 /home/hcq/huituo_server/data/kitti/training/velodyne/000008.bin
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32) # POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] #  顺序为[x_min, y_min, z_min, x_max, y_max, z_max]
        
        # 初始化PointFeatureEncoder对象
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        """ 
        kitti_dataset.yaml 中 POINT_FEATURE_ENCODING 如下
			POINT_FEATURE_ENCODING: {
    			encoding_type: absolute_coordinates_encoding,
   				used_feature_list: ['x', 'y', 'z', 'intensity'],
    			src_feature_list: ['x', 'y', 'z', 'intensity'],
			}
		"""
        # # 训练模式下，定义数据增强类
        self.data_augmentor = DataAugmentor( # DataAugmentor
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        # # 定义数据处理类
        # 解决：_init__() missing 1 required positional argument: 'num_point_features
        # self.data_processor = DataProcessor(
        #     self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        # )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property  # @property 可以让对象像访问属性一样区访问方法 self.mode 
    def mode(self):
        return 'train' if self.training else 'test'  # 训练还是测试

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    # 更新成员变量的值
    def __setstate__(self, d):
        self.__dict__.update(d)

    # 自定义数据集时需要实现该方法,接收来自模型的预测结果
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.
        要支持自定义数据集，请实现此功能以接收来自模型的预测结果，
        然后将统一的标准坐标转换为所需的坐标，然后选择将其保存到磁盘。

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    # 自定义数据集时实现该方法，加载原始数据和labels，并将这些数据转换到统一的坐标下，调用self.prepare_data()来处理数据和送进模型
    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError
    # 处理gt数据=========================================================================
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # # 训练模式下，对存在于class_name中的数据进行增强（由默认的3个gt_boxes增加到33个）
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # 返回一个bool数组，记录自定义数据集中ground_truth_name列表在不在我们需要检测的类别列表self.class_name里面
            # 比如kitti数据集中data_dict['gt_names']=['car','person','cyclist']，self.class_name='car',则gt_boxes_mask=[True, False, False]
            # [n for n in data_dict['gt_names']] value:  ['Car', 'Car']
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            # 数据增强 传入字典参数，**data_dict是将data_dict里面的key-value对都拿出来
            # pcdet/datasets/augmentor/data_augmentor.py
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict, # **data_dict是将data_dict里面的key-value对都拿出来
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        # # 筛选需要检测的gt_boxes
        if data_dict.get('gt_boxes', None) is not None:
            #  下标: 返回data_dict['gt_names']中存在于class_name的下标， 也就是我们一开始指定要检测哪些类，只需要获得这些类的下标
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names) # # ['Car', 'Pedestrian', 'Cyclist']
            # # 根据selected，留下我们需要的gt_boxes和gt_names
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # 将当帧数据的gt_names中的类别名称对应到class_names的下标
            # 举个栗子，我们要检测的类别class_names = ['car','person']，对于当前帧，类别gt_names = ['car', 'person', 'car', 'car']，当前帧出现了3辆车，一辆单车，获取索引后，gt_classes = [1, 2, 1, 1]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            # 将类别index信息放到每个gt_boxes的最后
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            # 处理GT box   ，归一化 转【0，1】
            # rela_dis = [70.4, 80.0, 4.0] 
            # gt_boxes[:,0]  = (gt_boxes[:,0] - 0) / 70.4
            # gt_boxes[:,1]  = (gt_boxes[:,1]  + 40.0) / 80.0
            # gt_boxes[:,2]  = (gt_boxes[:,2]  + 3.0) / 4.0
            # for i  in range(3,6):
            #     gt_boxes[:,i]  = gt_boxes[:,i] / rela_dis[i-3]
            # for i in range(len(gt_boxes[0])):
            #     gt_boxes[:,i] = gt_boxes[:,i]/100
            #     gt_boxes[:,i]  = (gt_boxes[:,i] - gt_boxes[:,i].min()) / ( gt_boxes[:,i].max()-gt_boxes[:,i].min())
            data_dict['gt_boxes'] = gt_boxes # 真值框  gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        # 使用点的哪些属性 比如x,y,z等
        data_dict = self.point_feature_encoder.forward(data_dict)
        # 对点云进行预处理，包括移除超出point_cloud_range的点、 打乱点的顺序以及将点云转换为voxel
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod #  将data_dict传入后命名为batch_list  返回值：  ====================================================
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list) # defaultdict 表示在当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
        
        # 把batch里面的每个sample按照key-value合并
        for cur_sample in batch_list: # 遍历列表
            for key, val in cur_sample.items(): # 对字典6个参数进行遍历cur_sample 表示现在的样本值（current sample），也是DemoDataset类
                data_dict[key].append(val) # 保存在新的字典里面
        batch_size = len(batch_list) # 列表中目前只有一个字典，应该多多少个bin文件就有多少字典dict？？？？
        ret = {}
        # 遍历6次  # data_dict{dict: 6}： 'points'(60270,4) 'voxels'' voxel_num_points'
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']: # 对应体素=================================
                    ret[key] = np.concatenate(val, axis=0) # 多个数组的拼接
                elif key in ['points', 'voxel_coords']: # #对应关键点==================================
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i) #   constant_values=i 表示第一维填充i
                        """
                            ((0,0),(1,0))
                            在二维数组array第一维（此处便是行）前面填充0行，最后面填充0行；
                            在二维数组array第二维（此处便是列）前面填充1列，最后面填充0列
                            mode='constant'表示指定填充的参数
                            constant_values=i 表示第一维填充i
                        """
                        coors.append(coor_pad) #将coor_pad补充在coors后面
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']: # 对应真值================================================
                    max_gt = max([len(x) for x in val]) # 找寻最大价值的点
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32) # #画可视图用的
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d # 
                else:
                    ret[key] = np.stack(val, axis=0) # #类似concatenate,给指定axis增加维度
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret # 返回值
