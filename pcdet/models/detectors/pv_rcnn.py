'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-06-20 21:37:11
FilePath: /PCDet/pcdet/models/detectors/pv_rcnn.py
'''
from .detector3d_template import Detector3DTemplate
""" 
主要这两个函数forward和get_training_loss
    # 定位到forward，在神经网络中向前传递数据data_dict，得到预测数据pred_dicts
    def forward(self, batch_dict):
    #training_loss
    def get_training_loss(self): #training_loss

 """

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset): # 初始化参数
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() # ===>>>>这里的build_networks是继承的Detector3DTemplate中的函数  def build_networks(self):
    # 定位到forward，在神经网络中向前传递数据data_dict，得到预测数据pred_dicts
    def forward(self, batch_dict):  # 其他位置会引用，by ： pred_dicts, _ = model.forward(data_dict) 
        for cur_module in self.module_list:  #在__init__中
            batch_dict = cur_module(batch_dict)

        if self.training: # #如果用于训练
            loss, tb_dict, disp_dict = self.get_training_loss() #得到训练loss   关键函数1=====================local： 下面line40=====================================

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else: #其他情况，不用于训练，即用于测试
            pred_dicts, recall_dicts = self.post_processing(batch_dict) # #关键函数2===================local： pcdet/models/detectors/detector3d_template.py====================================
            return pred_dicts, recall_dicts
    # 关键函数1==========================================================
    def get_training_loss(self): #training_loss
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss() # models/dense_heads/anchor_head_template.py 206行
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn # #根据pv-rcnn论文，最终的loss是由这三个loss组成
        return loss, tb_dict, disp_dict
