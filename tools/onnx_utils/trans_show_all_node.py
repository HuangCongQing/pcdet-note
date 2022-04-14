'''
Description: 展示中间过程shape
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-04-14 14:21:45
LastEditTime: 2022-04-14 14:21:46
FilePath: /PCDet/tools/onnx_utils/trans_show_all_node.py
'''

import onnx
onnxmodel = onnx.load('/root/ljc_ws/3DObjectDetection/tools/onnx_utils/results/hcq2.onnx')
# onnx.shape_inference.infer_shapes(onnxmodel)
onnx.save(onnx.shape_inference.infer_shapes(onnxmodel), './results/shape.onnx')
# print()