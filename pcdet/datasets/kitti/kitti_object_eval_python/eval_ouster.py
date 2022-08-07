'''
Description: 基于修改/mmdetection3d/mmdet3d/core/evaluation/kitti_utils/eval_ouster.py
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-08-05 22:31:39
LastEditTime: 2022-08-06 00:12:49
FilePath: /PCDet/pcdet/datasets/kitti/kitti_object_eval_python/eval_ouster.py
'''
import gc
import io as sysio
import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

# #这个函数是处理一帧的数据, current_class是6个类别中的其中一类 current_class =1（pedestrian）
def clean_data(gt_anno, dt_anno, current_class, difficulty):
    '''
        print("____________clean_data() args:________________")
        print('current_class  :  ',current_class)
        print('difficulty : ',difficulty)
            ____________clean_data() args:________________
            current_class  :   0
            difficulty :  0
    '''
    # CLASS_NAMES = ['car', 'pedestrian', 'cyclist'] # #类别
    CLASS_NAMES = ['Truck','Auxiliary','Car','Excavator','Widebody','Pedestrian']

    #检测难度从易到难，为了检测到同样数目的gt，使最小值减小，最大值增大
    # MIN_HEIGHT = [40, 25, 25] #高度
    # MAX_OCCLUSION = [0, 1, 2]  #遮挡
    # MAX_TRUNCATION = [0.15, 0.3, 0.5] #截断
    # dc_bboxes, ignored_gt = [], []
    ignored_gt, ignored_dt =  [], []
    
    current_cls_name = CLASS_NAMES[current_class].lower() # 'pedestrian'   报错：IndexError: list index out of range

    # 获取当前帧中物体object的个数
    num_gt = len(gt_anno['name'])  #gt数量
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0

    #对num_gt中每一个物体object：
    for i in range(num_gt):

        #获取这个物体的name，并小写
        gt_name = gt_anno["name"][i].lower()

        valid_class = -1

        # 如果该物体正好是 需要处理的当前的object，将valid_class值为 1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        
        ignore = False
        if valid_class == 1 and not ignore:
            # 如果 为有效的物体， 且该物体object不忽略，
            # 则ignored_gt上该值为0，有效的物体数num_valid_gt+1
            ignored_gt.append(0)
            num_valid_gt += 1 # 有效的gt数量
        else:
            ignored_gt.append(-1)

    #对num_dt中每一个物体object：
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    
    '''
    
    
        print("__________有效的gt数量num_valid_gt____________")
        print(num_valid_gt)
        print("__________ignored_gt____________")
        print(ignored_gt)
        print("__________ignored_dt____________")
        print(ignored_dt)
    
        该函数的输出结果是
        __________有效的gt数量num_valid_gt____________
4
__________ignored_gt____________
[0, 0, 0, 0]
__________ignored_dt____________
[0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

            __________num_valid_gt____________
            76
            __________ignored_gt____________
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 
            -1, -1, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0,
            0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1]
            __________ignored_dt____________
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1, -1,
            0, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 
            -1, -1, -1, 0, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 
            0, -1, -1, 0, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, 
            0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1]
    '''

    return num_valid_gt, ignored_gt, ignored_dt


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps

# mmdet3d/core/evaluation/kitti_utils/rotate_iou.py
def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.？？？？？？？？？？？？？//
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0] #
    for i in numba.prange(N): # 遍历每个gt box
        for j in numba.prange(K): # 遍历要检测的图像
            if rinc[i, j] > 0:  # 如果高度方向有重叠
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                #  # 重叠部分的高度
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) - # # 重叠部分的最高点（取两个图像各自最高点的最小值）
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4])) # 重叠部分的最低点（取两个图像各自最低点的最大值）

                if iw > 0:# 如果宽度方向有重叠
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]  # gt box 的面积
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]  #检测图像的面积
                    inc = iw * rinc[i, j] # 重叠部分的面积
                    if criterion == -1: # 默认执行criterion = -1
                        ua = (area1 + area2 - inc) # 总的面积（交集）
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua  # 计算得到iou=================================
                else:
                    rinc[i, j] = 0.0  # 否则就没有重叠

# boxes是GT， 
def d3_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval # mmdet3d/core/evaluation/kitti_utils/rotate_iou.py
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], # （9，7）  只要5维 ： centers, dims,angles(clockwise when positive) with the shape of [N, 5].
                               qboxes[:, [0, 2, 3, 5, 6]], 2) # iou = np.zeros((N, K), dtype=np.float32
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc # (9,9)

#  TP，FP，TN，FN
@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas[:, -1]   #获取预测的得分情况 最后一列
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,)) # 初始化为0
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size): # 遍历GT 
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue
        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框！！！
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            # 获取 overlaps 中相应的数值======================================================
            overlap = overlaps[j, i]
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap # 当全最大overlap
                det_idx = j # 当前的检测id
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # tp fn等计算
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的==================================================
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]  # 阈值得到
            thresh_idx += 1
            
            assigned_detection[det_idx] = True

    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]

# def compute_statistics_jit(overlaps,
#                            gt_datas,
#                            dt_datas,
#                            ignored_gt,
#                            ignored_det,
#                         #    dc_bboxes,
#                            metric,
#                            min_overlap,
#                            thresh=0,
#                            compute_fp=False,
#                            compute_aos=False):

#     det_size = dt_datas.shape[0]
#     gt_size = gt_datas.shape[0]
#     dt_scores = dt_datas[:, -1]
#     dt_alphas = dt_datas[:, 4]
#     gt_alphas = gt_datas[:, 4]
#     dt_bboxes = dt_datas[:, :4]
#     # gt_bboxes = gt_datas[:, :4]

#     assigned_detection = [False] * det_size
#     ignored_threshold = [False] * det_size
#     if compute_fp:
#         for i in range(det_size):
#             if (dt_scores[i] < thresh):
#                 ignored_threshold[i] = True
#     NO_DETECTION = -10000000
#     tp, fp, fn, similarity = 0, 0, 0, 0
#     # thresholds = [0.0]
#     # delta = [0.0]
#     thresholds = np.zeros((gt_size, ))
#     thresh_idx = 0
#     delta = np.zeros((gt_size, ))
#     delta_idx = 0
#     for i in range(gt_size):
#         if ignored_gt[i] == -1:
#             continue
#         det_idx = -1
#         valid_detection = NO_DETECTION
#         max_overlap = 0
#         assigned_ignored_det = False

#         for j in range(det_size):
#             if (ignored_det[j] == -1):
#                 continue
#             if (assigned_detection[j]):
#                 continue
#             if (ignored_threshold[j]):
#                 continue
#             overlap = overlaps[j, i]
#             dt_score = dt_scores[j]
#             if (not compute_fp and (overlap > min_overlap)
#                     and dt_score > valid_detection):
#                 det_idx = j
#                 valid_detection = dt_score
#             elif (compute_fp and (overlap > min_overlap)
#                   and (overlap > max_overlap or assigned_ignored_det)
#                   and ignored_det[j] == 0):
#                 max_overlap = overlap
#                 det_idx = j
#                 valid_detection = 1
#                 assigned_ignored_det = False
#             elif (compute_fp and (overlap > min_overlap)
#                   and (valid_detection == NO_DETECTION)
#                   and ignored_det[j] == 1):
#                 det_idx = j
#                 valid_detection = 1
#                 assigned_ignored_det = True

#         if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
#             fn += 1
#         elif ((valid_detection != NO_DETECTION)
#               and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
#             assigned_detection[det_idx] = True
#         elif valid_detection != NO_DETECTION:
#             tp += 1
#             # thresholds.append(dt_scores[det_idx])
#             thresholds[thresh_idx] = dt_scores[det_idx]
#             thresh_idx += 1
#             if compute_aos:
#                 # delta.append(gt_alphas[i] - dt_alphas[det_idx])
#                 delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
#                 delta_idx += 1

#             assigned_detection[det_idx] = True
#     if compute_fp:
#         for i in range(det_size):
#             if (not (assigned_detection[i] or ignored_det[i] == -1
#                      or ignored_det[i] == 1 or ignored_threshold[i])):
#                 fp += 1
#         nstuff = 0
#         # if metric == 0:
#         #     overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
#         #     for i in range(dc_bboxes.shape[0]):
#         #         for j in range(det_size):
#         #             if (assigned_detection[j]):
#         #                 continue
#         #             if (ignored_det[j] == -1 or ignored_det[j] == 1):
#         #                 continue
#         #             if (ignored_threshold[j]):
#         #                 continue
#         #             if overlaps_dt_dc[j, i] > min_overlap:
#         #                 assigned_detection[j] = True
#         #                 nstuff += 1
#         fp -= nstuff
#         if compute_aos:
#             tmp = np.zeros((fp + delta_idx, ))
#             # tmp = [0] * fp
#             for i in range(delta_idx):
#                 tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
#                 # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
#             # assert len(tmp) == fp + tp
#             # assert len(delta) == tp
#             if tp > 0 or fp > 0:
#                 similarity = np.sum(tmp)
#             else:
#                 similarity = -1
#     return tp, fp, fn, similarity, thresholds[:thresh_idx]

# 计算TP，FP，TN，FN
#@numba.jit(nopython=True)
def compute_statistics_jit1(
                           overlaps,
                           gt_datas, # 是一个数，表示当前帧中的物体个数
                           dt_datas,  # N x 1阵列，表示的是预测得到的N个物体的得分情况score
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0, # 自己设置的阈值
                           compute_fp=False,
                           compute_aos=False):

    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(ignored_gt)
    #print(ignored_det)
    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas  #获取预测得到的N个物体的得分情况====================================================
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size): # 遍历此帧的每个预测的的得分情况score
            # print(dt_scores, dt_scores[i], i)
            if (dt_scores[i] < thresh): # -1.0？？？？？done  # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    # 遍历GT的数据
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue

        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框，其下表id=det_idx
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            # 获取 overlaps 中相应的iou数值
            overlap = overlaps[j, i] ##！！！！============================================================
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True
        # 以上for循环结束，得到和GT最匹配的的一个预测框，其下表id=det_idx

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的
            tp += 1  # ===========================================================
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx] # dt_scores # N x 1阵列，表示的是预测得到的N个物体的得分情况score
            thresh_idx += 1
            
            assigned_detection[det_idx] = True # 其预测框的设置为True（默认都是False）
    
    
    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1 # gt是对的，预测为错的
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]




def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

# 将各部分数据融合===
# @numba.jit(nopython=True) # 注释@ 不然报错 - argument 6: Unsupported array dtype: object
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                            #  dc_nums,
                             gt_datas,
                             dt_datas,
                            #  dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    # dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            # gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            # dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            # ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            # ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            # dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            gt_data = gt_datas[i] # 修改！！！！！！！！！！！！==========================
            dt_data = dt_datas[i]
            ignored_gt = ignored_gts[i]
            ignored_det = ignored_dets[i]

            tp, fp, fn, similarity, _ = compute_statistics_jit1( #  计算tp, fp, fn, similarity, thresholds=======================================================
                overlap, # 单个图像的iou值b/n gt和dt
                gt_data, # # 是一个数，表示当前帧中的物体个数 # N x 5阵列
                dt_data, # N x 6阵列？？？？？？？？？
                ignored_gt,# 长度N数组，-1、0、1
                ignored_det,# 长度N数组，-1、0、1
                # dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh, # 阈值
                compute_fp=True,
                compute_aos=compute_aos)

            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity

        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        # dc_num += dc_nums[i]

# 计算iou（elif metric == 2:） num_parts=2  # 函数里面gt和dt互换了一下
def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50): # num_parts=50 num_parts修改为2=================================
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos) #
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)# 每帧的障碍物数量 [ 1  3  6  5 13]   [1 2 3 3 2 1 2 1 1 1 1 1 1 1 1 1 1 1 1]
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)# [50 50 50 50 50]
    num_examples = len(gt_annos) # 测试集文件数，这里是19
    split_parts = get_split_parts(num_examples, num_parts) # [9, 9, 1]
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts: #  [2, 2, 1] [10,10,10,10,10,1]
        # # 基本上将数据集分成多个部分并进行迭代
        gt_annos_part = gt_annos[example_idx:example_idx + num_part] # 分成9 9 1三部分
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        if metric == 0:  # metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
            gt_boxes = np.concatenate([a['bbox'] for a in gt_annos_part], 0) # 不会运行
            dt_boxes = np.concatenate([a['bbox'] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1: # Eval type.  1: bev================================================
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0) # ValueError: need at least one array to concatenate
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1) # (100, 7)
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            
            overlap_part = bev_box_overlap(gt_boxes, # rotate_iou_gpu_eval mmdet3d/core/evaluation/kitti_utils/rotate_iou.py
                                           dt_boxes).astype(np.float64)

        elif metric == 2: #:  2: 3d.=======================================================
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0) # ValueError: need at least one array to concatenate
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1) # (100, 7)
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = d3_box_overlap(gt_boxes,
                                          dt_boxes).astype(np.float64) # 计算3d IOU=============================
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part) # append([9,9])
        ''' (176, 16)
        [array([[0.00019199, 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.00085803, 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.00768037, 0.        ,
        0.        ],
       ...,
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.00692087,
        0.01676498]])]
        '''
        example_idx += num_part
    
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num # 返回值

# 准备数据
def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    # 数据初始化
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], [] # donecares不需要
    total_num_valid_gt = 0
    # 遍历每个图像gt
    for i in range(len(gt_annos)):
        #得到的是参数，当前帧的这个类别的 有效物体数，和有效物体的索引列表
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty) # 函数的调用======================================
        num_valid_gt, ignored_gt, ignored_det = rets # 得到结果
        # 将每一帧的ignored_gt数据类型进行转换为numpy格式，再添加到ignored_gts
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        # #! 最终形成ignored_gts的List
        # if len(dc_bboxes) == 0:
        #     dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        #     #! dc_boxes 是一个np array，形状(该图像中的don't care boxes 数量, 4)
        # else:
        #     dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        #     #! 每一列是一个Don't Care bbox
        # total_dc_num.append(dc_bboxes.shape[0])
        # #! don't care boxes的数量. total_dc_num是该图像dc_boxes数量的list，每个图像对应一个total_dc_num
        # dontcares.append(dc_bboxes)
        #! 该图像的dc_boxes list的list
        total_num_valid_gt += num_valid_gt # #! 有效 gt boxes 总数的计数器
        gt_datas_num = len(gt_annos[i]["name"])
        gt_datas_list.append(gt_datas_num)

        #dt_datas_score = dt_annos[i]["score"]
        dt_datas_score = dt_annos[i]["score"][..., np.newaxis] # KeyError: 'scores' 报错
        dt_datas_list.append(dt_datas_score) #   
    # 返回值
    return (
                gt_datas_list,  #存放的是 每一帧物体的个数
                dt_datas_list,  #存放的是每一帧 不同物体的score得分的情况，是（N,1）
                ignored_gts, ignored_dets,   #存在
                total_num_valid_gt  #有效GT总数量存在
                )
#         gt_datas = np.concatenate(
#             [gt_annos[i]['bbox'],  # #! bbox index 形状是 N x 4
#             gt_annos[i]['alpha'][..., np.newaxis]], 1) #! alpha index 形状是 N -> 当np.newaxis, 是 N x 1
#         #! 所以合并后成为 N x 5 ，5表示 [x1, y1, x2, y2, alpha]
#         dt_datas = np.concatenate([
#             dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
#             dt_annos[i]['score'][..., np.newaxis]
#         ], 1)
#         #! 类似的, 形状为N x 6， 6是 [x1, y1, x2, y2, alpha, score]
#         gt_datas_list.append(gt_datas)
#         dt_datas_list.append(dt_datas)
#         # boxes list 的 list
#         # gt_datas只和gt_annos[i]有关，dt_datas只和dt_annos[i]有关
#         # 因此每个图像对应一个gt_datas_list和dt_datas_list

#     total_dc_num = np.stack(total_dc_num, axis=0) # don't care boxes 数量
#     '''
#    此处的所有数组的长度 = 数据集中的图像数量
#    gt_datas_list：list（N x 5个数组）
#    dt_datas_list：list（N x 6个数组）
#    ignore_gts：list（长度为N的数组（值-1、0或1））
#    ignore_dets ：list（长度为N的数组（值-1、0或1））
#    dontcares：list（（图像x 4个数组中的无关框数量）
#    total_dc_num：list（图像值中的无关框数量）
#    total_num_valid_gt：有效gt的总数（int）
#    '''
#     return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
#             total_dc_num, total_num_valid_gt)

# ouster_eval-->do_eval-->eval_class
def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric, #  2: 3d
               min_overlaps,
               compute_aos=False,
               num_parts=2): # num_parts=200):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d ===============================================
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    #如果验证集gt_annos中的帧数 和 从model中验证出来dt_annos帧的长度不一致，直接报错！
    assert len(gt_annos) == len(dt_annos)
    # 验证集中帧的总数是 num_examples:51
    num_examples = len(gt_annos) # ouster:19
    #得到的split_parts是一个list的类型，num_parts=5,
    # 意思是将51分为5部分，经过一下函数得到的是：split_parts：[10,10,10,10,10,1]
    # if num_examples < num_parts:
    #     num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts) # [9, 9, 1]

    #计算iou
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts) # 1 计算iou（函数里面gt和dt互换了一下！！！）metric = 2=======================================
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets #
    N_SAMPLE_PTS = 41

    #获取min_overlaps的各个的维度，得到的是(2, 3, 5)
    # 获取当前类别的个数num_class：5，难度的个数为3
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    #初始化precision，recall，aos
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    
    # 每个类别
    for m, current_class in enumerate(current_classes):
        # 每个难度
        for idx_l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty) # 2 准备数据==================================================
            # (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt) = rets
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets# ==================================
            # 运行两次，首先进行中等难度的总体设置，然后进行简单设置。
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = [] # 初始化
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit( # 3 计算tp, fp, fn, similarity, thresholds====================================
                        overlaps[i], # gt和dt的iou======================
                        gt_datas_list[i],      # 是一个数，表示当前帧中的物体个数  19帧  [1, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                        dt_datas_list[i], # N x 1阵列，表示的是预测得到的N个物体的得分情况 举例： [[0.9999982 ], [0.999997  ], [0.99999654], [0.99999547], [0.99999547]]
                        ignored_gts[i],   # 长度N数组，-1、0
                        ignored_dets[i],     # 长度N数组，-1、0
                        # dontcares[i],
                        metric,          # 0, 1, 或 2 (bbox, bev, 3d)
                        min_overlap=min_overlap,      # 浮动最小IOU阈值为正阈值
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets # ======================================
                    thresholdss += thresholds.tolist() # [-1. -1. -1. -1. -1. -1.]？？？？？？？？？？？？？？？？？
                
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                # thresholds是 N_SAMPLE_PTS长度的一维数组，记录分数，递减，表示阈值
                # 储存有关gt/dt框的信息（是否忽略，fn，tn，fp）
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    # gt_datas_part = np.concatenate(
                    #     gt_datas_list[idx:idx + num_part], 0) # ValueError: zero-dimensional arrays cannot be concatenated
                    # dt_datas_part = np.concatenate(
                    #     dt_datas_list[idx:idx + num_part], 0)
                    # # dc_datas_part = np.concatenate(
                    # #     dontcares[idx:idx + num_part], 0)
                    # ignored_dets_part = np.concatenate(
                    #     ignored_dets[idx:idx + num_part], 0)
                    # ignored_gts_part = np.concatenate(
                    #     ignored_gts[idx:idx + num_part], 0)
                    gt_datas_part = np.array(gt_datas_list[idx:idx+num_part])
                    dt_datas_part = np.array(dt_datas_list[idx:idx+num_part])
                    ignored_dets_part = np.array(ignored_dets[idx:idx+num_part])
                    ignored_gts_part = np.array(ignored_gts[idx:idx+num_part])

                    # 再将各部分数据融合===
                    fused_compute_statistics( # 调用compute_statistics_jit1===================
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        # total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        # dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,  # 阈值
                        # compute_aos=compute_aos
                        )
                    idx += num_part
                # #计算recall和precision
                for i in range(len(thresholds)):
                    recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, idx_l, k, i] = pr[i, 0] / (
                        pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, idx_l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                # 返回各自序列的最值
                for i in range(len(thresholds)):
                    precision[m, idx_l, k, i] = np.max(
                        precision[m, idx_l, k, i:], axis=-1)
                    recall[m, idx_l, k, i] = np.max(
                        recall[m, idx_l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, idx_l, k, i] = np.max(
                            aos[m, idx_l, k, i:], axis=-1)
    ret_dict = {
        'recall': recall,  # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  
        'precision': precision,   # RECALLING RECALL的顺序，因此精度降低====================================
        'orientation': aos,
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict  # 返回

#
def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

# 是计算评估结果的重要函数
def do_eval(gt_annos,
            dt_annos,
            current_classes, # [0, 1, 2, 3, 4, 5, 6]
            min_overlaps, # min_overlaps(2,3,7)//
            eval_types=['3d']): # 修改      # eval_types=['bbox', 'bev', '3d']):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    mAP_bbox = None
    mAP_aos = None
    # if 'bbox' in eval_types: # 不运行
    #     ret = eval_class(
    #         gt_annos,
    #         dt_annos,
    #         current_classes,
    #         difficultys,
    #         0,
    #         min_overlaps,
    #         compute_aos=('aos' in eval_types))
    #     # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    #     mAP_bbox = get_mAP(ret['precision'])
    #     if 'aos' in eval_types:
    #         mAP_aos = get_mAP(ret['orientation'])

    mAP_bev = None # 初始化
    if 'bev' in eval_types: # 不运行
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, # 1 bev
                         min_overlaps)
        mAP_bev = get_mAP(ret['precision'])

    mAP_3d = None # 初始化
    # 3D的评测结果=====================================================================
    if '3d' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, # 得到结果 eval_types =2================================
                         min_overlaps)
        mAP_3d = get_mAP(ret['precision'])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


# 评测开始 入口 main=================================================================================================================
def ouster_eval(gt_annos,
               dt_annos,
               current_classes, #预测的类别，根据预测的类别显示顺序
               eval_types=['bev', '3d']):# 只要3D
            #    eval_types=['bbox', 'bev', '3d']):# 只要3D
    """KITTI evaluation.
    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
    if 'aos' in eval_types:
        assert 'bbox' in eval_types, 'must evaluate bbox when evaluating aos'
    # overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,0.5],
    #                         [0.7, 0.5, 0.5, 0.7, 0.5],
    #                         [0.7, 0.5, 0.5, 0.7, 0.5]])
    # overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
    #                         [0.5, 0.25, 0.25, 0.5, 0.25],
    #                         [0.5, 0.25, 0.25, 0.5, 0.25]])
    # min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    overlap_0_7 = np.array([[0.2, 0.5, 0.5, 0.7,0.5,0.5,0.5], # ncrease
                            [0.2, 0.5, 0.5, 0.7, 0.5,0.5,0.5],
                            [0.2, 0.5, 0.5, 0.7, 0.5,0.5,0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5,0.5,0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25]])
    # 通过下面一行，min_overlaps的形状是(2, 3, 7)的三维数组，7是因为有7个类别
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 7]

    # class_to_name = { # 分类标签修改
    #     0: 'Car',
    #     1: 'Pedestrian',
    #     2: 'Cyclist',
    #     3: 'Van',
    #     4: 'Person_sitting',
    # }
    class_to_name = { # 分类标签修改   match  overlap_0_7
        0: 'Truck', # important!   ##[[0.7,0.7,0.7],[0.5, 0.5, 0.5]]等于上面数组的：min_overlaps[:,:,0]
        1: 'Auxiliary', # important!
        2: 'Car', # 前移动！！
        3: 'Excavator',
        4: 'Widebody',
        5: 'Pedestrian',
        6: 'Others',
    }
    # 将名字和对应的类别号反一下，便于索引 {'Truck': 0, 'Auxiliary': 1, 'Car': 2, 'Excavator': 3, 'Widebody': 4, 'Pedestrian': 5, 'Others': 6}
    name_to_class = {v: n for n, v in class_to_name.items()}

    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    # 定义一个空列表，如果current_classes中每一类为str类型，则存入相应的类别号
    # 如当前判断的Car，Pedestrian，Cyclist，则current_classes_int=[ 0,1,2]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    #当前的类别变成了含有数字的列表   current_classes=[ 0,1,2]
    current_classes = current_classes_int # [0, 1, 2, 3, 4, 5, 6]
    
    # min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    # 取min_overlaps的前5列，因为有5个类别是需要分类和计算的
    #得到的min_overlaps的形状：（2,3,5）
    min_overlaps = min_overlaps[:, :, current_classes] #   计算得到当前类的min_overlaps

    result = ''
    # check whether alpha is valid
    compute_aos = False
    # pred_alpha = False # 用不到
    # valid_alpha_gt = False
    # for anno in dt_annos: # 遍历每个预测
    #     mask = (anno['alpha'] != -10)
    #     if anno['alpha'][mask].shape[0] != 0:
    #         pred_alpha = True
    #         break
    # for anno in gt_annos:  # 遍历每个GT
    #     if anno['alpha'][0] != -10:
    #         valid_alpha_gt = True # 合法
    #         break
    # compute_aos = (pred_alpha and valid_alpha_gt)
    # if compute_aos:
    #     eval_types.append('aos') # 是否添加aos！！！！
    
    # 计算结果的函数（后面详细说）

    # testing==============
    # dt_annos = gt_annos  #测试
    for i in range(len(dt_annos)):
        dt_annos[i]['score'] = dt_annos[i]['score'] + 2.0
    # test End

    mAPbbox, mAPbev, mAP3d, mAPaos = do_eval(gt_annos, dt_annos, # 调用============================================
                                             current_classes, min_overlaps,
                                             eval_types)
    #打印结果=================================================================================
    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']
    # for j, curcls in enumerate(current_classes): # 修改，只要前三个类别
    for j, curcls in enumerate(current_classes[0:3]): # 修改，只要前三个类别
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j])) # Truck AP@0.20, 0.20, 0.20:
            if mAPbbox is not None:
                result += 'bbox AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAPbbox[j, :, i])
            if mAPbev is not None:
                result += 'bev  AP:{:.4f}, {:.4f}, {:.4f}\n'.format( # 也需要哈
                    *mAPbev[j, :, i])
            if mAP3d is not None:
                result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format( # 3d   AP:6.8182, 6.8182, 6.8182
                    *mAP3d[j, :, i])

            if compute_aos:
                result += 'aos  AP:{:.2f}, {:.2f}, {:.2f}\n'.format(
                    *mAPaos[j, :, i])
            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                # prefix = f'KITTI/{curcls_name}'
                prefix = f'Ouster/{curcls_name}'  # ouster
                if mAP3d is not None:
                    ret_dict[f'{prefix}_3D_{postfix}'] = mAP3d[j, idx, i] # 只要mAP3d
                if mAPbev is not None:
                    ret_dict[f'{prefix}_BEV_{postfix}'] = mAPbev[j, idx, i]
                if mAPbbox is not None:
                    ret_dict[f'{prefix}_2D_{postfix}'] = mAPbbox[j, idx, i]

    # calculate mAP over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP@{}, {}, {}:\n'.format(*difficulty))
        if mAPbbox is not None:
            mAPbbox = mAPbbox.mean(axis=0)
            result += 'bbox AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAPbbox[:, 0])
        if mAPbev is not None:
            mAPbev = mAPbev.mean(axis=0)
            result += 'bev  AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAPbev[:, 0])
        if mAP3d is not None: # ==============================================
            mAP3d = mAP3d.mean(axis=0)
            result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP3d[:, 0]) # 只运行这一行
        if compute_aos:
            mAPaos = mAPaos.mean(axis=0)
            result += 'aos  AP:{:.2f}, {:.2f}, {:.2f}\n'.format(*mAPaos[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP3d is not None:
                ret_dict[f'KITTI/Overall_3D_{postfix}'] = mAP3d[idx, 0] # 只运行这一行==========================
            if mAPbev is not None:
                ret_dict[f'KITTI/Overall_BEV_{postfix}'] = mAPbev[idx, 0]
            if mAPbbox is not None:
                ret_dict[f'KITTI/Overall_2D_{postfix}'] = mAPbbox[idx, 0]

    return result, ret_dict


def kitti_eval_coco_style(gt_annos, dt_annos, current_classes):
    """coco style evaluation of kitti.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.

    Returns:
        string: Evaluation results.
    """
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(class_to_range[curcls])[:,
                                                                   np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f'{class_to_name[curcls]} '
                             'coco AP@{:.2f}:{:.2f}:{:.2f}:'.format(*o_range)))
        result += print_str((f'bbox AP:{mAPbbox[j, 0]:.2f}, '
                             f'{mAPbbox[j, 1]:.2f}, '
                             f'{mAPbbox[j, 2]:.2f}'))
        result += print_str((f'bev  AP:{mAPbev[j, 0]:.2f}, '
                             f'{mAPbev[j, 1]:.2f}, '
                             f'{mAPbev[j, 2]:.2f}'))
        result += print_str((f'3d   AP:{mAP3d[j, 0]:.2f}, '
                             f'{mAP3d[j, 1]:.2f}, '
                             f'{mAP3d[j, 2]:.2f}'))
        if compute_aos:
            result += print_str((f'aos  AP:{mAPaos[j, 0]:.2f}, '
                                 f'{mAPaos[j, 1]:.2f}, '
                                 f'{mAPaos[j, 2]:.2f}'))
    return result
