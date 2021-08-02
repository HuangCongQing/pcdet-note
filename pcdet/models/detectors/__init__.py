'''
Description: https://blog.csdn.net/weixin_44579633/article/details/107542954#commentBox
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-02 23:48:58
LastEditTime: 2021-06-19 21:35:27
FilePath: /PCDet/pcdet/models/detectors/__init__.py
'''
from .detector3d_template import Detector3DTemplate #  对应类：class Detector3DTemplate(nn.Module):
from .PartA2_net import PartA2Net  # PartA2_net.py文件对应的类 ： class PartA2Net(Detector3DTemplate):
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
# 字典dict(包含各个模型的class)
__all__ = {
    'Detector3DTemplate': Detector3DTemplate, # class Detector3DTemplate(nn.Module):
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN, # class PVRCNN(Detector3DTemplate):
    'PointPillar': PointPillar, # # class PointPillar(Detector3DTemplate):
    'PointRCNN': PointRCNN
}
''' 
调用的这些包就是pcdet/models/detectors下的各个py文件，里面的函数用到再说

 '''


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME]( # model_cfg.NAME:   'PVRCNN'
        model_cfg=model_cfg, num_class=num_class, dataset=dataset # # num_class : ['Car', 'Pedestrian', 'Cyclist']
    )

    return model
""" 
class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset): # 初始化参数

 """