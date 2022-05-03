'''
Author: https://blog.csdn.net/weixin_44128857/article/details/108516213
Date: 2021-07-30 11:53:21
LastEditTime: 2022-04-16 18:15:46
LastEditors: Please set LastEditors
Description: 最重要函数:def get_infos()==========================================================
FilePath: /PCDet/pcdet/datasets/kitti/kitti_dataset.py
'''
''' 
 gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)

 '''
import copy
import pickle

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

#定义kitti数据集的类
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        #初始化类，将参数赋值给 类的属性
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        #传递参数是 训练集train 还是验证集val
        """ 
            DATA_SPLIT: {
                'train': train,
                'test': val # 测试集 依据 data/kitti/ImageSets/val.txt
            }
         """
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        #  root_path的路径是/data/kitti/
        #kitti数据集一共三个文件夹“training”和“testing”、“ImageSets”
        #如果是训练集train，将文件的路径指为训练集training ，否则为测试集testing
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        #/data/kitti/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        #选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        #得到.txt文件里的序列号，组成列表sample_id_list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        #创建用于存放kitti信息的空列表
        self.kitti_infos = []
        self.include_kitti_data(self.mode) # #调用函数，加载kitti数据，mode的值为：train 或者  test

    def include_kitti_data(self, mode):
        if self.logger is not None: #  #如果日志信息存在，则加入'Loading KITTI dataset'的信息
            self.logger.info('Loading KITTI dataset')
        #创建新列表，用于存放信息
        kitti_infos = [] # bin文件数量

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path #  # root_path的路径是/data/kitti/  #则 info_path：/data/kitti/kitti_infos_train.pkl之类的文件
            if not info_path.exists(): #  #如果该文件不存在，跳出，继续下一个文件
                continue
            with open(info_path, 'rb') as f:
                #  pickle.load(f) 将该文件中的数据 解析为一个Python对象 infos，
                # 并将该内容添加到kitti_infos 列表中
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos) # bin文件数量

        #最后在日志信息中 添加 kitti数据集样本总个数
        if self.logger is not None:
            self.logger.info('【kitti_dataset.py】Total samples(bin文件数量) for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        #参数赋值
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        
        #root_path的路径是/data/kitti/ 
        # 则root_split_path=/data/kitti/ training或者testing
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        
        #/data/kitti/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        #选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt') #  train.txt
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    #根据序列号，获取lidar信息
    def get_lidar(self, idx):
        # lidar_file为某个点云的bin文件（序列）
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx) # 得到bin文件
        assert lidar_file.exists()#如果该文件不存在，直接跳出，并报错
        #读取该 bin文件类型，并将点云数据以 numpy的格式输出！！！
        #并且将数据 转换成 每行四个数据，刚好是一个点云数据的四个参数：X,Y,Z,R(强度或反射值）
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4) # reshape四列

    # 根据序列号，获取图像的信息
    def get_image_shape(self, idx):
        #获取到某个具体的图片
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #print(img_file)
        assert img_file.exists()   #如果该图片文件不存在，直接报错
        # 返回图片的数据，最终得到的是这张图片的 长和宽 的，如 (375, 1242)
        # 该函数的返回值是：array([ 375, 1242], dtype=int32)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)
    
    #根据序列号，获取标签label的信息
    def get_label(self, idx):
        #获取到某个标签的.txt文件 ，该文件表示 图片中物体的参数
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()   #如果不存在，直接报错
        # 调用get_objects_from_label函数，首先读取该文件的所有行 赋值为 lines
        # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
        # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等  # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
        return object3d_kitti.get_objects_from_label(label_file)

    #该函数是根据序列得到某一标定
    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    ## 如果有路面情况，调用该函数，获得路面的相关信息
    #该文件没有路面情况，故不分析
    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists(): #获取文件，如果不存在，报错
        #调用类和函数，该返回值是一个类的参数，包含相机自身的内参和外参数
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    #定义静态方法
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect: # pts_rect （M,3）,M是该场景下采集到点云的个数。
            img_shape:
            calib:
        #其中三个参数是这个场景下（一帧下的场景：同一个bin文件、图像）
        # ：pts_rect （M,3）,M是该场景下采集到点云的个数。
        #  info['image']['image_shape']：该图片的长和宽，如[375,1242]

        Returns:

        """
        #调用矫正类中的方法，将点的直角坐标转为 相机坐标，pts_img为（M,2）
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        #返回的结果是val_flag_1和val_flag_2 ：
        # array([ True,  True,  True,  True,  True,  True,  True,  True,  True, True])
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        #得到一系列标识符 true or false，用于判断该点云能否有效 （是否用于训练）
        #所以 pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个（M是该场景下采集到点云的个数）

        return pts_valid_flag

    # ######   获取信息##############
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        #处理单帧数据
        def process_single_scene(sample_idx):
             #  self.split 的实际值是 train训练集 or val验证集
            print('%s sample_idx: %s' % (self.split, sample_idx)) # 
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            #将目前的特征、序列加入info字典里
            info['point_cloud'] = pc_info

            #获取图像的信息，并加进去
            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            # calib是一个字典，里面是相机坐标的一些参数，返回P2,P3,R0,V2C等参数
            calib = self.get_calib(sample_idx)

            #在p2下面加了一行数，从（3,4）变为（4,4）
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            #生成与R0相同数据类型的4X4全零数组，该数组前三行三列为R0，最后一位数置为1
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            #V2C也加了一行 0 0 0 1
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info
            # 处理label数据================================================
            if has_label: # True
                # 调用get_objects_from_label函数，首先读取该文件的所有行 赋值为 lines
                # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
                # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等
                obj_list = self.get_label(sample_idx)
                annotations = {} # #定义一个空字典，annotations是注解的意思
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # # 计算有效物体的个数，如10个，object除去“DontCare”4个，还剩num_objects6个
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare']) # 除去“DontCare”
                #总物体的个数 10个
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                #由此可以得到 index=[0,1,2,3,4,5,-1,-1,-1,-1]
                annotations['index'] = np.array(index, dtype=np.int32)

                #假设有效物体的个数是N
                # 取有效物体的 location（N,3）、dimensions（N,3）、rotation_y（N,1）信息，
                loc = annotations['location'][:num_objects] # 中心点坐标==================================
                dims = annotations['dimensions'][:num_objects] # =====================================================
                rots = annotations['rotation_y'][:num_objects]
                #通过计算得到在lidar坐标系下的坐标，loc_lidar:（N,3）
                loc_lidar = calib.rect_to_lidar(loc) # 转换一下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                #分别取 dims中的第一列、第二列、第三列：l,h,w（N,1）
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                #  h[:, 0] ：（1,N）,通过下面计算后，得到loc_lidar[:, 2]：（1，N）
                loc_lidar[:, 2] += h[:, 0] / 2
                #下面计算得到的gt_boxes_lidar是(N,7) ,  np.newaxis 的功能是增加新的维度,
                #    x[:, np.newaxis] ，放在后面，会给列上增加维度
                #  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar # (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center ========================================

                info['annos'] = annotations

                if count_inside_pts:
                    # points 是一个数组，假设一个bin文件里点云的个数为M，
                    # 则points：（M,4）,四个数代表一个点云
                    points = self.get_lidar(sample_idx)
                    # get_calib返回的相机方面的参数，P2,R0,V2C等参数，
                    # 同时calib也是 calibration类的一个对象
                    calib = self.get_calib(sample_idx)
                    #一个点云有四个数字组成：前三个是坐标信息：x,y,z，最后一个是反射的强度值
                    # 所以，在以下函数调用中，取了points的前三列 作为参数  x,y,z：（M,3）
                    #返回得到的pts_rect：（M,3）
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    #其中三个参数是这个场景下（一帧下的场景：同一个bin文件、图像）
                    # ：pts_rect （M,3） info['image']['image_shape']：该图片的长和宽，如[375,1242]
                    #得到一系列标识符 true or false，用于判断该点云能否有效 （是否用于训练）
                    #所以 fov_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    #把为True所在的行挑选出来，假设有m个false，
                    # 所以pts_fov是（M-m,4）,里面的数据还是原始的x,y,z,反射值
                    pts_fov = points[fov_flag]

                    #gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    #经过一系列复杂的变换后boxes_to_corners_3d()，
                    # 返回值corners_lidar为（N,8,3）,数据类型是numpy.ndarray  # （N,8,3) ？？？？？？

                    #num_gt是这一帧图像里物体的总个数，假设为10，
                    # 则num_points_in_gt=array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    #num_objects是有效物体的个数，为N，假设为N=6
                   
                    for k in range(num_objects):
                        #in_hull函数是判断点云是否在bbox中，（是否在物体的2D检测框中）
                        #在这个函数里，判断点云的点是否在该检测框内，如果是，返回flag
                        #运用到了“三角剖分”的概念和方法
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k]) # def in_hull(p, hull):
                        #该函数的参数是：pts_fov[:, 0:3]，（M-m,3）:表示当前帧可用点云的三维位置信息
                        # corners_lidar[k]：当前帧第k个物体框的信息
                        # 则返回值是flag：array([False, False, True, False, True, False,...])，(一共M-m个)
                        #则，flag.sum()是计算，在当前框内的点云的个数（True的个数）
                        
                        #最后num_points_in_gt是一个数组:[5,8,10,5,4,...],其长度是框的个数，
                        # 里面的数字表示该框里包含点云的个数======================================
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        #是.txt文件下的序列号，组成列表sample_id_list，上面的函数的是一个帧的信息
        #下面几行是将该sample_id_list列表上的都执行一下，每个返回的信息info都存放在infos里面
        #最后执行完成后，infos是一个列表，每一个元素代表了一帧的信息
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos) # infos是一个列表，每一个元素代表了一帧的信息

    #建立地面真相数据库：翻译的意思是地面实况，放到机器学习里面，
    # 再抽象点可以把它理解为真值、真实的有效值或者是标准的答案
    # 用trainfile产生groundtruth_database，
    # 意思就是只保存训练数据中的gt_box及其包围的点的信息，用于数据增强=====数据增强=======
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        #创建保存文件的路径  root_path的路径是/data/kitti/
        #如果是“train”，创建的路径是  /data/kitti/gt_database 
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        #在/data/kitti/下创建保存 info的文件
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        #传入的参数 info_path 是一个.pkl文件，ROOT_DIR / 'data' / 'kitti'/('kitti_infos_%s.pkl' % train_split)
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        #调取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            #输出的是 第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            #取train.txt 里面的样本序列，其实就是data/kitti/ImageSets/train.txt里面的数字序列，
            # 如000000，000003,000007....
            sample_idx = info['point_cloud']['lidar_idx']

            
            #读取该 bin文件类型，并将点云数据以 numpy的格式输出！！！
            #将数据 转换成 每行四个数据，刚好是一个点云数据的四个参数：X,Y,Z,R(强度或反射值）
            #故 points是一个数组（M,4）
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            #name的数据是['car','car','pedestrian'...'dontcare'...]表示当前帧里面的所有物体objects
            names = annos['name']
            #difficulty：[0,1,2,-1,0,0,-1,1,...,]里面具体物体的难度，长度为总物体的个数
            difficulty = annos['difficulty']
            # bbox是一个数组，表示物体2D边框的个数，
            # 假设有效物体为N,dontcare个数为n,则bbox：（N+n,4）
            bbox = annos['bbox']
            #同样是一个数组：（N,7）,:  x,y,z,dx,dy,dz,heading，为有效物体的信息
            gt_boxes = annos['gt_boxes_lidar']
            #num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]

            #对参数的处理：首先转为tensor格式（M,3）（N,7）
            ##返回一个“全零"(后面又运行了一个cuda的函数，故值可能会变化)的张量，
            # 维度是（N,M）,  N是有效物体的个数，M是点云的个数，在转化为numpy
            #point_indices意思是点的索引
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                #创建文件名，并设置保存路径，最后文件如：000007_Cyclist_3.bin（位置：kitti/gt_database）
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                #point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                #再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                #gt_points中每个的前三列数据
                # 又都减去gt_boxes中当前物体的前三列的位置信息
                gt_points[:, :3] -= gt_boxes[i, :3]
                #把gt_points的信息写入文件里
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    #获取当前物体的信息
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    #把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        #把所有的all_db_infos写入到文件里面
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod # 生成预测结果
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id: 帧号
            pred_dicts: list of pred_dicts  预测后得到的列表
                pred_boxes: (N, 7), Tensor   预测的框，包含七个信息
                pred_scores: (N), Tensor      预测得分
                pred_labels: (N), Tensor        预测的标签
            class_names:
            output_path:

        Returns:

        """
        #获取预测后的模板字典 ret_dict，全部定义为全零的向量
        #参数num_samples 是这一帧里面的物体个数
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        #生成一个帧的预测字典
        #参数：box_dict是预测的结果，pred_dicts: list of pred_dicts  预测后得到的列表
        # 在 self.generate_prediction_dicts()中接收模型预测的在统一坐标系下表示的3D检测框，
        # 并转回自己所需格式即可。
        def generate_single_sample_dict(batch_index, box_dict):
            #pred_scores: (N), Tensor      预测得分，N是这一帧预测物体的个数
            #pred_boxes: (N, 7), Tensor   预测的框，包含七个信息
            #pred_labels: (N), Tensor        预测的标签
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            #定义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                #如果没有物体，则返回空字典
                return pred_dict

            # batch_dict:    frame_id: 帧号（但不是一个纯数字，应该是一个字典之类的）
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            #将预测完成的信息（相对激光雷达的）转化为相对相机的坐标系下
            #此处需要改！！！！
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            #向刚刚创建的字典中填充预测的信息，类别名，角度之类的
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

            '''    pred_dicts: list of pred_dicts  预测后得到的列表
                    pred_boxes: (N, 7), Tensor   预测的框，包含七个信息
                    pred_scores: (N), Tensor      预测得分
                    pred_labels: (N), Tensor        预测的标签
            '''

        annos = []
        #  index的值为1,2，。。。，N  ？？？？不确定
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            #得到单个  ！！！（帧）！！！的预测的结果，
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                #定义输出结果的文件，帧号.txt文件
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    #将预测信息写入该文件中
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        #打印输出 物体的名字当前帧中，每个物体的预测结果
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        #返回处理后的预测信息
        return annos
    # 评价指标====================================================================
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            #如果'annos'没在kitti信息里面，直接返回空字典。实际上在里面呢
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        #复制一下参数det_annos
        #copy.deepcopy()在元组和列表的嵌套上的效果是一样的，都是进行了深拷贝（递归的）
        eval_det_annos = copy.deepcopy(det_annos)
        # 一个info 表示一帧数据的信息，则下面是把所有数据的annos属性取出来，进行copy
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        #下面的函数相当于做了进一步的运算，然后返回结果===========================================
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        #等于返回训练帧的总个数，等于图片的总个数，帧的总个数
        return len(self.kitti_infos)

    #在 self._getitem_() 中加载自己的数据，
    #并将点云与3D标注框均转至前述统一坐标定义下，
    # 送入数据基类提供的 self.prepare_data()；
    #参数index 是需要送进来处理的 帧序号的索引值，如1,2,3,4.。。。
    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        #将第index帧的信息 全部赋值为info
        info = copy.deepcopy(self.kitti_infos[index])

        #将采样的序列号 赋值出来 sample_idx，这个序列号可能不是连续的
        #是在train.txt文件里的数据序列号
        sample_idx = info['point_cloud']['lidar_idx']

         #得到该序列号相应的 点云数据 （M,4）
        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx) # 得到该序列号相应的相机参数，如P2,R0,V2C

        #得到相应帧的图片长和宽，如[375,1242]
        img_shape = info['image']['image_shape']
        #在配置文件里FOV_POINTS_ONLY=true
        if self.dataset_cfg.FOV_POINTS_ONLY:
            #将雷达坐标系转为直角坐标，参数都是（M,3）
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            ##fov_flag得到一系列标识符 true or false，用于判断该点云能否有效 （是否用于训练）
            #所以 pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #取出有效的点云数据points
            points = points[fov_flag]

        #定义输入数据的字典：points 处理过后的点云数据，
        # frame_id 帧号（采样的序列号如000003,000015...，是train.txt文件里的数据）
        # calib：得到该序列号相应的相机参数，如P2,R0,V2C 。calib = self.get_calib(sample_idx)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            #将该帧信息中的annos 赋值出来
            annos = info['annos']
            #下面函数的作用是 在info中剔除包含'DontCare'的数据信息
            #不但从name中剔除，余下的location、dimensions等信息也都不考虑在内
            annos = common_utils.drop_info_with_name(annos, name='DontCare')

            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']

            #由下面得到的是 （N,7）,因为物体都是由相机测量得到的，
            # 所以这是相对于相机坐标系的坐标，但点云的数据是基于雷达坐标系的，
            # 所有要转换为激光雷达坐标系
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            """该函数的参数是    boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            Returns:    boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center"""
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            #将新的键值对 添加到输入的字典中去，此时输入中有五个键值对了
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            #如果有路面信息，则加入进去
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        #调用函数，将需要送入数据进行训练的input_dict
        #  进一步的处理，这一步经过了很多的处理。。。。。。
        data_dict = self.prepare_data(data_dict=input_dict)

        #得到相应帧的图片长和宽，如[375,1242]
        #把这个信息添加进去
        data_dict['image_shape'] = img_shape
        return data_dict

# main 入口
def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    #传递参数
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False) # class实例化
    train_split, val_split = 'train', 'val'

    #定义文件的路径和名称
    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split) # 训练
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split) # 验证
    trainval_filename = save_path / 'kitti_infos_trainval.pkl' # trainval
    test_filename = save_path / 'kitti_infos_test.pkl' # 测试

    print('---------------【kitti_dataset.py】Start to generate data infos---------------')
    # 训练集
    dataset.set_split(train_split) 
    #执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    ##  下面是得到train.txt 中序列相关的所有点云数据的信息，并且进行保存
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) # get_infos 获取处理后的点云数据
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename) #保存 kitti_infos_train.pkl
    # 验证集val.txt
    #开始对验证集的数据进行信息统计并保存
    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) # # get_infos 获取处理后的点云数据
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('【验证集】Kitti info val file is saved to %s' % val_filename) # 保存kitti_infos_val.pkl

    #把训练集和验证集的信息 合并写到一个文件里
    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename) # 保存
    # 测试集 #写测试集的信息并保存
    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    #生成数据增强相关的文件    # 用trainfile产生groundtruth_database，
    # 意思就是只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    #以下的命令和配置，只是为创建kitti信息做的
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        #以下的命令和配置，只是为创建kitti信息做的
        # dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos( # 调用函数 ，生成处理的数据集================
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'], # 修改分类数量
            data_path=ROOT_DIR / 'data' / 'kitti', # 数据路径
            save_path=ROOT_DIR / 'data' / 'kitti' # .pkl文件
        )
