'''
Description: https://blog.csdn.net/weixin_44128857/article/details/117445420
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-08-03 12:41:16
LastEditTime: 2021-10-17 20:29:10
FilePath: /PCDet/pcdet/datasets/huituo/robosense/robosense_dataset.py
'''
import numpy as np
import copy
import pickle
import os
import json
import numpy as np
import pcl
import pandas
import sys
import random

from skimage import io
from ...dataset import DatasetTemplate
from ....ops.roiaware_pool3d import roiaware_pool3d_utils
from ....utils import box_utils,  common_utils
from pathlib import Path 


class RobosenseDataset(DatasetTemplate):

    def __init__(self,dataset_cfg,class_names,training= True, root_path=None,logger = None):
        #参数：配置文件dataset_cfg, 要分类的类名class_names, 是否为训练training= True,
        # 数据集的路径root_path=None,    日志文件logger = None
        # 这里由于是类继承的关系，所以root_path在父类中已经定义
        #print("即将运行初始化")
        super().__init__(
            dataset_cfg = dataset_cfg,class_names=class_names, 
            training = training, root_path = root_path,logger = logger
        )
        self.robosense_infos =[]
        #用于存放文件路径的列表
        self.files_list_pcd = []
        self.files_list_label = []
        self.files_list_label_train = []
        self.files_list_label_val = []
        self.files_list_pcd_train = []
        self.files_list_pcd_val = []
        self.train_ratio_of_all_labels=self.dataset_cfg.TRAIN_RATIO_OF_ALL_LABELS

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.include_robosense_data(self.mode)


    def include_robosense_data(self,mode):
        if self.logger is not None:
            self.logger.info('Loading robosense dataset')
        
        robosense_infos =[]
        '''
        INFO_PATH:{
            'train':[robosense_infos_train.pkl],
            'test':[robosense_infos_val.pkl],}
        '''
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            
            info_path = str(self.root_path)+'/'+ info_path
            #info_path = self.root_path/ info_path
            if not Path(info_path).exists():
                continue

            with open(info_path,'rb') as f:
                infos = pickle.load(f)
                robosense_infos.extend(infos)

        self.robosense_infos.extend(robosense_infos)

        if self.logger is not None:
            self.logger.info('Total samples for robosense dataset: %d'%(len(robosense_infos)))

    #根据数据地址的路径，获取路径下 文件夹的名字列表
    def get_folder_list(self,root_path):
        folder_list = []
        root_path =root_path
        #读取该目录下所有文件夹的名字，并组成一个列表
        folder_list = os.listdir(root_path)
        return folder_list

    #根据文件夹的列表，返回包含所有文件名的列表 files_list_pcd 和files_list_label 
    def get_files_name_list(self):
        folder_list = []
        folder_list = self.get_folder_list(self.root_path)

        files_list_pcd = []
        files_list_label = []

        for per_folder in folder_list:

            #一条路的文件夹的路径one_road_path
            one_road_path = str(self.root_path+per_folder+'/')
            #一条路下文件夹下的文件列表 one_road_list =['label','pcd']
            one_road_list = self.get_folder_list(one_road_path)

            for one_folder in one_road_list:
                if one_folder == 'pcd':
                    pcd_path = str(one_road_path+one_folder)
                if one_folder == 'label':
                    label_path = str(one_road_path+one_folder)

            #获取pcd文件夹下面的文件名，并将文件的完整路径添加到列表里
            pcd_files = self.get_folder_list(pcd_path)
            for thisfile in pcd_files:
                if thisfile.endswith(".pcd"):
                    files_list_pcd.append(str(pcd_path+'/'+thisfile))

            #获取label文件夹下面的文件名，并将文件的完整路径添加到列表里
            label_files = self.get_folder_list(label_path)
            for thisfile in label_files:
                if thisfile.endswith(".json"):
                    files_list_label.append(str(label_path +'/'+ thisfile))
        
        #返回files_list_pcd和files_list_label的列表，
        # 该列表内包含了所有pcd和label文件的路径名
        return files_list_pcd,files_list_label

    #根据label的路径，得到对应的pcd路径
    def from_label_path_to_pcd_path(self,single_label_path):
        #根据label的路径，推出来pcd相应的路径，两者在倒数第二个文件夹不同
        single_pcd_path = ''
        strl1 = 'label'
        strl2 = '.json'
        if strl1 in single_label_path:
            single_pcd_path = single_label_path.replace(strl1,'pcd')
        if strl2 in single_pcd_path:
            single_pcd_path = single_pcd_path.replace(strl2,'.pcd')
        #由此得到了label对应的pcd文件的路径 ：single_pcd_path
        return single_pcd_path

    
    # 根据label文件路径列表，返回所有标签的数据
    def get_all_labels(self,num_workers = 4,files_list_label=None):
        import concurrent.futures as futures

        #根据一个label文件的路径single_label_path，获取该文件内的信息
        #信息包括：type, center ,size,rotation,id等信息
        global i 
        i =0
        def get_single_label_info(single_label_path):
            global i
            i=i+1
            single_label_path = single_label_path
            #打开文件
            with open(single_label_path,encoding = 'utf-8') as f:
                labels = json.load(f)
            
            #定义一个空字典，用于存放当前帧label所有objects中的信息
            single_objects_label_info = {}
            single_objects_label_info['single_label_path'] = single_label_path
            single_objects_label_info['single_pcd_path'] = self.from_label_path_to_pcd_path(single_label_path)
            single_objects_label_info['name'] = np.array([label['type'] for label in labels['labels']])
            single_objects_label_info['box_center'] = np.array([[label['center']['x'], label['center']['y'],label['center']['z']]  for  label in labels['labels']])
            single_objects_label_info['box_size'] = np.array([[label['size']['x'],label['size']['z'],label['size']['z']] for label in labels['labels']])
            single_objects_label_info['box_rotation'] = np.array([[label['rotation']['roll'],label['rotation']['pitch'],label['rotation']['yaw']]  for label in labels['labels']])
            single_objects_label_info['tracker_id'] = np.array([ label['tracker_id'] for label in labels['labels']])
            
            box_center = single_objects_label_info['box_center']
            box_size = single_objects_label_info['box_size']
            box_rotation = single_objects_label_info['box_rotation']

            rotation_yaw = box_rotation[:,2].reshape(-1,1)
            gt_boxes = np.concatenate([box_center,box_size,rotation_yaw],axis=1).astype(np.float32)
            single_objects_label_info['gt_boxes'] = gt_boxes

            print("The current processing progress is %d / %d "%(i,len(files_list_label)))
            return single_objects_label_info

        files_list_label = files_list_label
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(get_single_label_info,files_list_label)
        infos = list(infos)
        print("*****************************Done!***********************")
        print("type  of  infos :",type(infos))
        print("len  of  infos :",len(infos))
    
        #此时的infos是一个列表，列表里面的每一个元素是一个字典，
        #每个元素里面的内容是当前帧的信息
        return infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.robosense_infos) * self.total_epochs

        return len(self.robosense_infos)

    #去掉一帧里面无效的点云数据
    def remove_nan_data(self,data_numpy):
        data_numpy = data_numpy
        data_pandas = pandas.DataFrame(data_numpy)
        #删除任何包含nan的所在行 (实际有三分之一的数据无效，是[nan, nan, nan, 0.0])
        data_pandas = data_pandas.dropna(axis=0,how='any')
        data_numpy = np.array(data_pandas)

        return data_numpy

    #根据每一帧的pcd文件名和路径single_pcd_path，
    # 得到这一帧中的点云数据，返回点云的numpy格式（M,4）==============================================================
    # ！！！！对比参考：pcdet/datasets/kitti/kitti_dataset.py   def process_single_scene(sample_idx):
    def get_single_pcd_info(self,single_pcd_path):
        single_pcd_path = single_pcd_path
        single_pcd_points = pcl.load_XYZI(single_pcd_path)
        #将点云数据转化为numpy格式
        single_pcd_points_np = single_pcd_points.to_array()
        #去掉一帧点云数据中无效的点
        single_pcd_points_np = self.remove_nan_data(single_pcd_points_np)
        #print(single_pcd_points_np)
        #将点云数据转化为list格式
        #single_pcd_points_list =single_pcd_points.to_list()

        return single_pcd_points_np

    # 根据名字，去掉相应的信息，主要针对single_objects_label_info
    # single_objects_label_info 里关于‘unknown’的数据信息
    def drop_info_with_name(self,info,name):
        ret_info = {}
        info = info 
        keep_indices =[ i for i,x in enumerate(info['name']) if x != name]
        for key in info.keys():
            if key == 'single_label_path' or key == 'single_pcd_path':
                ret_info[key] = info[key]
                continue
            ret_info[key] = info[key][keep_indices]

        return ret_info
    
    #根据训练列表label的数据，得到对应的pcd的路径列表list
    def from_labels_path_list_to_pcd_path_list(self,labels_path_list):
        pcd_path_list = []
        for m in labels_path_list:
            pcd_path_list.append(self.from_label_path_to_pcd_path(m))
        return pcd_path_list

    #实现列表相减的操作,从被减数list_minute中去掉减数list_minus的内容
    def list_subtraction(self,list_minute,list_minus):
        list_difference = []
        for m in list_minute:
            if m not in list_minus:
                list_difference.append(m)
        return list_difference

    def __getitem__(self,index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.robosense_infos)

        single_objects_label_info = copy.deepcopy(self.robosense_infos[index])
        single_label_path = single_objects_label_info['single_label_path']
        single_pcd_path = self.from_label_path_to_pcd_path(single_label_path)

        #得到点云数据，且是有效的点云数据，返回点云的numpy格式（M,4）
        points = self.get_single_pcd_info(single_pcd_path)

        #定义输入数据的字典，包含：points，文件的路径，。。？
        input_dict = {
            'points': points, # 点云Mx4
            'frame_id': single_pcd_path,
            'single_pcd_path':single_pcd_path,
        }
        
        # 在single_objects_label_info字典里，剔除关于'unknown' 的信息
        single_objects_label_info = self.drop_info_with_name(info=single_objects_label_info,name='unknown')
        name =single_objects_label_info['name']             #(N,)
        box_center = single_objects_label_info['box_center']          #(N,3)
        box_size = single_objects_label_info['box_size']                    #(N,3)
        box_rotation  = single_objects_label_info['box_rotation']  #(N,3)
        tracker_id = single_objects_label_info['tracker_id']               #(N,)

        #以下是将 上面的3D框的数据 转化为统一的数据格式
        #数据格式为：(N,7)，分别代表 (N, 7) [x, y, z, l, h, w, r]
        # gt_boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center"""
        rotation_yaw = box_rotation[:,2].reshape(-1,1)
        gt_boxes = np.concatenate([box_center,box_size,rotation_yaw],axis=1).astype(np.float32)
        #print(gt_boxes.shape)
        #print(type(gt_boxes))

        input_dict.update({
                'gt_names':name,
                'gt_boxes':gt_boxes,
                'tracker_id':tracker_id
        })
        #print(input_dict)
        
        # 将点云与3D标注框均转至统一坐标定义后，送入数据基类提供的 self.prepare_data()
        #data_dict = input_dict
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    #由文件的完整路径得到文件的名字（去掉多余的信息）
    def from_filepath_get_filename(self,filepath):
        filename = ''
        filepath = filepath
        #得到一个元祖tuple,（目录，文件名）
        filepath_and_filename = os.path.split(filepath)
        filename = filepath_and_filename[1]
        #得到文件名+后缀,得到一个元祖tuple,（文件名，后缀）
        filename_and_extension = os.path.splitext(filename)
        filename = filename_and_extension[0]
        return filename

    #
    def create_groundtruth_database(self,info_path = None,used_classes =None,split = 'train'):
        import torch

        #database_save_path = str(Path(self.root_path))+'/'+('gt_database' if split =='train' else ('gt_database_%s'%split))
        #db_info_save_path = str(Path(self.root_path))+'/'+('robosense_dbinfos_%s.pkl'%split)
        database_save_path = Path(self.root_path)/('gt_database' if split =='train' else ('gt_database_%s'%split))
        db_info_save_path = Path(self.root_path)/('robosense_dbinfos_%s.pkl'%split)

        database_save_path.mkdir(parents=True,exist_ok=True)
        all_db_infos = {}

        with open(info_path,'rb') as f:
            infos = pickle.load(f)
        
        for k in range(len(infos)):
            print('gt_database sample:%d/%d'%(k+1,len(infos)))
            info = infos[k]
            #print("---------------去掉unknown之前的info--------------")
            #print(info)
            #去掉信息中 unknown的类别的信息
            info = self.drop_info_with_name(info=info,name='unknown')
            #print("---------------去掉unknown之后的info--------------")
            #print(info)


            single_label_path = info['single_label_path']
            single_pcd_path = info['single_pcd_path']
            points = self.get_single_pcd_info(single_pcd_path)

            #由文件的完整路径得到文件的名字（去掉多余的信息），方便后续的文件命名
            single_filename = self.from_filepath_get_filename(single_label_path)

            name = info['name']
            box_center = info['box_center']
            box_size = info['box_size']
            box_rotation = info['box_rotation']
            tracker_id = info['tracker_id']
            gt_boxes = info['gt_boxes']
            #num_obj是有效物体的个数
            num_obj = len(name)

            #对参数的处理：首先转为tensor格式（M,3）（N,7）
            ##返回一个“全零"(后面又运行了一个cuda的函数，故值可能会变化)的张量，
            # 维度是（N,M）,  N是有效物体的个数，M是点云的个数，在转化为numpy
            #point_indices意思是点的索引
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:,0:3]),torch.from_numpy(gt_boxes)
            ).numpy()   # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin'%(single_filename,name[i],i)
                filepath = database_save_path / filename

                #point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                #再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i]>0]

                #gt_points中每个的前三列数据
                # 又都减去gt_boxes中当前物体的前三列的位置信息
                gt_points[:, :3] -= gt_boxes[i, :3]

                #把gt_points 的信息写入文件里
                
                with open(filepath,'w') as f:
                    gt_points.tofile(f)
                
                
                if (used_classes is None) or name[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))   # gt_database/xxxxx.bin
                    #获取当前物体的信息
                    db_info = {
                        'name':name[i],'path':db_path,'image_idx':single_filename,
                        'gt_idx':i,'box3d_lidar':gt_boxes[i],'num_points_in_gt':gt_points.shape[0],
                        'box_center':box_center,'box_size':box_size,'box_rotation':box_rotation,'tracker_id':tracker_id
                    }

                    if name[i] in all_db_infos:
                        all_db_infos[name[i]].append(db_info)
                    else:
                        all_db_infos[name[i]] = [db_info]
        for k,v in all_db_infos.items():
            print('Database %s: %d'%(k,len(v)))
        
        with open(db_info_save_path,'wb') as f:
            pickle.dump(all_db_infos,f)
        
    #在 self.generate_prediction_dicts()中接收模型预测的
    # 在统一坐标系下表示的3D检测框，并转回自己所需格式即可。
    @staticmethod
    def generate_prediction_dicts(batch_dict,pred_dicts,class_names,output_path = None):

        '''
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
        '''

        #获取预测后的模板字典 ret_dict，全部定义为全零的向量
        #参数num_samples 是这一帧里面的物体个数
        def get_template_prediction(num_samples):
            ret_dict = {
                'name':np.zeros(num_samples),
                'box_center':np.zeros([num_samples,3]),
                'box_size':np.zeros([num_samples,3]),
                'box_rotation':np.zeros([num_samples,3]),
                'tracker_id':np.zeros(num_samples),
                'scores':np.zeros(num_samples),
                'pred_labels':np.zeros(num_samples),
                'pred_lidar':np.zeros([num_samples,7])
            }
            
            return ret_dict

        def generate_single_sample_dict(box_dict):

            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            #定义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                #如果这一帧的预测结果中，没有物体object，则返回空字典
                return pred_dict
            
            pred_dict['name'] = np.array(class_names)[pred_labels -1]
            pred_dict['scores'] = pred_scores
            pred_dict['pred_labels'] = pred_labels
            pred_dict['pred_lidar'] = pred_boxes

            pred_dict['box_center'] = pred_boxes[:,0:3]
            pred_dict['box_size'] = pred_boxes[:,3:6]
            pred_dict['box_rotation'][:,-1] = pred_boxes[:,6]

            return pred_dict
        
        #由文件的完整路径得到文件的名字（去掉多余的信息）
        def from_filepath_get_filename2(filepath):
            filename = ''
            filepath = filepath
            #得到一个元祖tuple,（目录，文件名）
            filepath_and_filename = os.path.split(filepath)
            filename = filepath_and_filename[1]
            #得到文件名+后缀,得到一个元祖tuple,（文件名，后缀）
            filename_and_extension = os.path.splitext(filename)
            filename = filename_and_extension[0]
            return filename
        
        annos = []
        for index,box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            
            #frame_id是当前帧的文件路径+文件名
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            #如果输出路径存在，则将预测的结果写入文件中
            if output_path is not None:
                filename = from_filepath_get_filename2(frame_id)
                cur_det_file = Path(output_path)/('%s.txt'%filename)
                with open(cur_det_file,'w') as f:
                    name =single_pred_dict['name']
                    box_center = single_pred_dict['box_center']
                    box_size = single_pred_dict['box_size']
                    box_rotation = single_pred_dict['box_rotation']

                    for idx in range(len(single_pred_dict['name'])):
                        print('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,'
                        %(name[idx],
                        box_center[idx][0],box_center[idx][1],box_center[idx][2],
                        box_size[idx][0],box_size[idx][1],box_size[idx][2],
                        box_rotation[idx][0],box_rotation[idx][1],box_rotation[idx][1]),
                        file=f)
        
        return annos

    def evaluation(self,det_annos,class_names,**kwargs):
        if 'name' not in self.robosense_infos[0].keys():
            #如果robosense_infos里没有信息，直接返回空字典
            return None,{}
        #参数det_annos 是验证集val下面的所有infos,是一个列表，每个元素是每一帧的字典数据
        #这里 的info是从model出来的，由generate_prediction_dicts函数得到，字典的键key:
        # name , box_center,box_size,box_rotation,tracked_id,    scores,pred_labels,pred_lidar,frame_id
        '''
        print('~~~~~~~~~~~~~det_annos~~~~~~~~~~~~~~~~~~')
        print(det_annos[0])
        print(len(det_annos))
        print('~~~~~~~~~~~~~~~class_names~~~~~~~~~~~~~~~~')
        print(class_names)
        '''

        from ...kitti.kitti_object_eval_python import eval3 as kitti_eval
        
        #复制一下参数det_annos
        #copy.deepcopy()在元组和列表的嵌套上的效果是一样的，都是进行了深拷贝（递归的）
        #eval_det_info的内容是从model预测出来的结果，等于det_annos
        eval_det_info = copy.deepcopy(det_annos)
        '''
        print('---------------------------eval_det_info--------------------------------------')
        print(eval_det_info[0].keys())
        print(type(eval_det_info))
        print(len(eval_det_info))
        '''
        

        # 一个info 表示一帧数据的信息，则下面是把所有数据的annos属性取出来，进行copy
        #实质上还是等于：eval_gt_infos = self.robosense_infos
        #eval_gt_infos的内容实际上是val的真实集合信息，
        eval_gt_infos = [copy.deepcopy(info) for info in self.robosense_infos]
        
        '''
        print('---------------------------eval_gt_infos--------------------------------------')
        print(eval_gt_infos[0].keys())
        print(type(eval_gt_infos))
        print(len(eval_gt_infos))
        print(class_names)
        '''
        

        #调用函数，预测得到ap的值
        #ap_result_str,ap_dict = kitti_eval.get_coco_eval_result1(eval_gt_infos,eval_det_info,class_names)
        ap_result_str,ap_dict = kitti_eval.get_official_eval_result(eval_gt_infos,eval_det_info,class_names)

        return ap_result_str,ap_dict 

def create_robosense_infos(dataset_cfg,class_names,data_path,save_path,workers=4):
    dataset = RobosenseDataset(dataset_cfg=dataset_cfg,class_names=class_names,root_path=data_path,training=False)
    train_split,val_split = 'train','val'
    #设置训练集的占比
    TRAIN_RATIO_OF_ALL_LABELS = dataset.train_ratio_of_all_labels

    #定义要保存的文件的路径和名称
    train_filename = save_path + '/' + ('robosense_infos_%s.pkl'%train_split)
    val_filename = save_path + '/' +('robosense_infos_%s.pkl'%val_split)
    
    trainval_filename = save_path + '/' + 'robosense_infos_trainval.pkl'
    test_filename = save_path + '/' +'robosense_infos_test.pkl'

    files_list_pcd,files_list_label =dataset.get_files_name_list()
    # 从总列表标签中取TRAIN_RATIO_OF_ALL_LABELS(0.5)的数据当做训练集train，
    # 剩下的当做val,并获取相应的文件路径列表
    files_list_label_train = random.sample(files_list_label,int(TRAIN_RATIO_OF_ALL_LABELS*len(files_list_label)))
    files_list_label_val = dataset.list_subtraction(files_list_label,files_list_label_train)
    files_list_pcd_train = dataset.from_labels_path_list_to_pcd_path_list(files_list_label_train)
    files_list_pcd_val = dataset.from_labels_path_list_to_pcd_path_list(files_list_label_val)

    #对类内的参数进行赋值
    dataset.files_list_pcd =files_list_pcd
    dataset.files_list_label =files_list_label
    dataset.files_list_label_train =files_list_label_train
    dataset.files_list_label_val =files_list_label_val
    dataset.files_list_pcd_train = files_list_pcd_train
    dataset.files_list_pcd_val = files_list_pcd_val

    print('------------------------Start to generate data infos-----------------------')

    robosense_infos_train = dataset.get_all_labels(files_list_label=files_list_label_train)
    with open(train_filename,'wb') as f:
        pickle.dump(robosense_infos_train,f)
    print('robosense info train file is saved to %s'%train_filename)

    robosense_infos_val = dataset.get_all_labels(files_list_label=files_list_label_val)
    with open(val_filename,'wb') as f:
        pickle.dump(robosense_infos_val,f)
    print('robosense info val file is saved to %s'%val_filename)

    with open(trainval_filename,'wb') as f:
        pickle.dump(robosense_infos_train + robosense_infos_val,f)
    print('robosense info trainval file is saved to %s'%trainval_filename)

    robosense_infos_test = dataset.get_all_labels(files_list_label=files_list_label)
    with open (test_filename,'wb') as f:
        pickle.dump(robosense_infos_test,f)
    print('robosense info test file is saved to %s'%test_filename)

    print('---------------------Strat create groundtruth database for data augmentation ----------------')
    #调用生成 database的函数，生成相应的文件   create_groundtruth_database
    dataset.create_groundtruth_database(info_path=train_filename,split=train_split)
    print('---------------------Congratulation !  Data preparation Done !!!!!!---------------------------')

    pass


if __name__ == '__main__':
    import sys
    if sys.argv.__len__()>1 and sys.argv[1] == 'create_robosense_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        #ROOT_DIR是得到当前项目的根目录：/root/dataset/OpenPCDet
        class_names= ['cone', 'pedestrain','bicycle','vehicle','big_vehicle'] # 数据类别
        
        create_robosense_infos(
            dataset_cfg=dataset_cfg,
            class_names= class_names,
            data_path='/root/dataset/RoboSense_Dataset/RS_datasets/datasets/',
            save_path='/root/dataset/RoboSense_Dataset/RS_datasets/datasets/'
        )


