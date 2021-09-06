'''
Description:  pointpillar
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-09-06 23:57:45
FilePath: /PCDet/pcdet/models/detectors/pointpillar.py
'''
from .detector3d_template import Detector3DTemplate  # 继承自pcdet/models/detectors/detector3d_template.py！！！！


class PointPillar(Detector3DTemplate): # 继承自pcdet/models/detectors/detector3d_template.py！！！！
    def __init__(self, model_cfg, num_class, dataset): # 初始化的三个参数【 参数都是从train.py传过来的】
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() # build网络=============================================================================

    def forward(self, batch_dict): # 得到预测数据pred_dicts
        for cur_module in self.module_list: # 遍历所需模块(module_topology的8个模块)
            # print('遍历所需模块:','\n',str(cur_module))
            batch_dict = cur_module(batch_dict) # 最终得到完整模型！！！！！！！！！！！！！============================================================================
        # 以上得到模型训练的所以数据=================================================

        if self.training: # #情况1：如果用于训练
            loss, tb_dict, disp_dict = self.get_training_loss() #得到训练loss   关键函数1=====================local： 下面line

            ret_dict = {
                'loss': loss # loss
            }
            return ret_dict, tb_dict, disp_dict # 返回数值
        else:  #情况2：#其他情况，不用于训练，即用于测试 用于测试推理！！！！！！！！！！！！！！！！
            pred_dicts, recall_dicts = self.post_processing(batch_dict) #关键函数2===================local： pcdet/models/detectors/detector3d_template.py
            return pred_dicts, recall_dicts
    
    # 关键函数1=================================================
    def get_training_loss(self): # 通过self.get_training_loss()调用
        disp_dict = {}
        # rpn_loss = cls_loss + box_loss(位置损失 + 方向(c朝向角)损失)
        loss_rpn, tb_dict = self.dense_head.get_loss()  # dense_head  models/dense_heads/anchor_head_template.py 245行
        tb_dict = {
            'loss_rpn(cls_loss +  box_loss(位置损失 + 方向(c朝向角)损失))': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
