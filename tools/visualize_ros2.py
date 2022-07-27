# coding:utf-8

"""
******************************************************************************
 * Copyright (C) 2022 AutoX, Inc. All Rights Reserved.
******************************************************************************
author: shimingli
date: 2022-05-22
description:

"""

import os
import sys
from matplotlib.image import BboxImage

import numpy as np
from pytest import mark
from yaml import Mark, warnings
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
import warnings
import math
# from core.modules.od.utils import common_utils
from pcdet.utils import common_utils
from geometry_msgs.msg import Point

_AXES2TUPLE = {
    'sxyz': (0,0,0,0), 'sxyx': (0,0,1,0), 'sxzy':(0,1,0,0),
    'sxzx': (0,1,1,0), 'syzx': (1,0,0,0), 'syzy':(1,0,1,0),
    'syxz': (1,1,0,0), 'syxy': (1,1,1,0), 'szxy':(2,0,0,0),
    'szxz': (2,0,1,0), 'szyx': (2,1,0,0), 'szyz':(2,1,1,0),
    'rzyx': (0,0,0,1), 'rxyx': (0,0,1,1), 'ryzx':(0,1,0,1),
    'rxzx': (0,1,1,1), 'rxzy': (1,0,0,1), 'ryzy':(1,0,1,1),
    'rzxy': (1,1,0,1), 'ryxy': (1,1,1,1), 'ryxz':(2,0,0,1),
    'rzxz': (2,0,1,1), 'rxyz': (2,1,0,1), 'rzyz':(2,1,1,1),
}

_NEXT_AXIS = [1,2,0,1]

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        # _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0 
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


class PointCloudPublisher(Node):
    WAYMO_LABELS = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

    rate = 10
    moving = True
    width = 100
    height = 100

    header = Header()
    header.frame_id = 'map'

    dtype = PointField.FLOAT32
    point_step =16

    xyzi_fields = [PointField(name='x', offset=0, datatype=dtype, count=1),
              PointField(name='y', offset=4, datatype=dtype, count=1),
              PointField(name='z', offset=8, datatype=dtype, count=1),
              PointField(name='intensity', offset=12, datatype=dtype, count=1)]
    xyz_fields = [PointField(name='x', offset=0, datatype=dtype, count=1),
              PointField(name='y', offset=4, datatype=dtype, count=1),
              PointField(name='z', offset=8, datatype=dtype, count=1)]

    def __init__(self, node_name='sasa_pc_publisher', interval=10):
        super().__init__(node_name)
        self.pubs_dict = dict()
        self.pubs_dict['raw_cloud'] = self.create_publisher(PointCloud2, 'raw_cloud', 10) # 创建发布者对象
        self.pubs_dict['final_cloud'] = self.create_publisher(PointCloud2, 'final_cloud', 10)
        self.pubs_dict['shifted_cloud'] = self.create_publisher(PointCloud2, 'shifted_cloud', 10)
        self.pubs_dict['first_sample_pc'] = self.create_publisher(PointCloud2, 'first_sample_pc', 10)
        self.pubs_dict['second_sample_pc'] = self.create_publisher(PointCloud2, 'second_sample_pc', 10)
        self.pubs_dict['selected_points1'] = self.create_publisher(PointCloud2, 'selected_points1', 10)
        self.pubs_dict['gt'] = self.create_publisher(MarkerArray, "gt", 10)
        self.pubs_dict['pred'] = self.create_publisher(MarkerArray, 'pred', 10)
        self.pubs_dict['test'] = self.create_publisher(MarkerArray, 'test', 10)
        # sasa add
        self.pubs_dict['sa_raw_cloud'] = self.create_publisher(PointCloud2, 'sa_raw_cloud', 10) # 256采样点
        self.pubs_dict['candidate_cloud'] = self.create_publisher(PointCloud2, 'candidate_cloud', 10)
        self.pubs_dict['vote_cloud'] = self.create_publisher(PointCloud2, 'vote_cloud', 10) # 256采样点
        self.pubs_dict['sa_gt'] = self.create_publisher(MarkerArray, "sa_gt", 10)
        self.pubs_dict['sa_pred'] = self.create_publisher(MarkerArray, "sa_pred", 10)

        # self.bbox_publish = self.create_publisher()
        self.interval = interval  # each interval to publish
        self.counter = 0
        self.markerD = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.markerD.markers.append(marker)

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        """
          7 -------- 4
         /|         /|
        6 -------- 5 .
        | |        | |
        . 3 -------- 0
        |/         |/
        2 -------- 1
        Args:
            boxes3d:  (N, 8) cls_id, x,y,z,l,w,h,angle[-pi,pi], (x, y, z) is the box center

        Returns:
            corners3d: (N, 8, 3)
        """
        boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 4:7].repeat(1, 8, 1) * template[None, :, :]
        corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 7]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 1:4]

        return corners3d.numpy() if is_numpy else corners3d

    def update_counter(self):
        self.counter += 1

    def is_ok(self):
        if self.counter % self.interval == 0:
            return True
        else:
            return False
    
    # 点topic
    def publish(self, points, topic='raw_cloud', mode='xyz'):
        """
        Args:
            points: (N, C), 
            topic: string, 
            mode: string, it is for setting the type of fields.

        """
        # assert isinstance(points, np.ndarray)
        fields = None
        if mode == 'xyz':
            fields = self.xyz_fields
        elif mode == 'xyzi':
            fields = self.xyzi_fields
        else:
            warnings.warn(f"pub_mode is error, {mode} is invalid, we set fields into xyz_fields by default.", UserWarning)
            fields = self.xyz_fields
        
        self.header.stamp = self.get_clock().now().to_msg()
        # print("\n",self.header,fields,points.shape )
        #  std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1658745608, nanosec=370320169), frame_id='map') 
        # [sensor_msgs.msg.PointField(name='x', offset=0, datatype=7, count=1), sensor_msgs.msg.PointField(name='y', offset=4, datatype=7, count=1), sensor_msgs.msg.PointField(name='z', offset=8, datatype=7, count=1)] 
        # torch.Size([256, 3])
        pc2_msg = point_cloud2.create_cloud(self.header, fields, points)
        self.pubs_dict[topic].publish(pc2_msg)

    # bbox topic
    def publish_boxes(self, bboxes, topic='gt', color=(1.0,0.0,0.0, 0.3)):
        """
        Args:
            bboxes[np.ndarray]:(M, 8) cls_id, x,y,z,l,w,h,angle[-pi,pi] 注意输入类型
            topic[str]: gt or pred
            color[tuple]: (r,g,b,a)
        """
        assert bboxes.shape[1] == 8, "publish bboxes shape is error!"
        # clear last time markers
        self.pubs_dict[topic].publish(self.markerD)

        markersArr = MarkerArray()
        base_num = bboxes.shape[0]
        for i, box in enumerate(bboxes.astype(np.float32)):
            marker = Marker()
            marker.header.frame_id = self.header.frame_id
            marker.id = i  # must be unique
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(box[1])
            marker.pose.position.y = float(box[2])
            marker.pose.position.z = float(box[3])
            q = quaternion_from_euler(0, 0, box[7])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = float(box[4])
            marker.scale.y = float(box[5])
            marker.scale.z = float(box[6])

            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])

            markersArr.markers.append(marker)

            marker_txt = Marker()
            marker_txt.header.frame_id = self.header.frame_id
            marker_txt.id = base_num + i  # must be unique
            marker_txt.type = Marker.TEXT_VIEW_FACING
            marker_txt.action = Marker.ADD
            marker_txt.pose.position.x = float(box[1])
            marker_txt.pose.position.y = float(box[2])
            marker_txt.pose.position.z = float(box[3] + box[6] / 2.0)
            marker_txt.color.r = 1.0
            marker_txt.color.g = 1.0
            marker_txt.color.b = 1.0 
            marker_txt.color.a = 1.0
            marker_txt.scale.z = 0.3
            marker_txt.text = PointCloudPublisher.WAYMO_LABELS[int(box[0])] # label 类别
            markersArr.markers.append(marker_txt)

        self.pubs_dict[topic].publish(markersArr)

    def publish_boxes2(self, bboxes, topic='gt', color=(1.0,0.0,0.0,1.0), line_width=0.1):
        """
        Args:
            bboxes[np.ndarray]:(M, 8) cls_id, x,y,z,l,w,h,angle[-pi,pi]
            topic[str]: gt or pred
            color[tuple]: (r,g,b,a)
        """
        assert bboxes.shape[1] == 8, "publish bboxes shape is error!"
        # clear last time markers
        self.pubs_dict[topic].publish(self.markerD) # 发布出去
        
        bboxes = PointCloudPublisher.boxes_to_corners_3d(bboxes)

        markersArr = MarkerArray() # 框
        for i, box in enumerate(bboxes.astype(np.float32)):
            marker = Marker()
            marker.header.frame_id = self.header.frame_id
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = [Point(x=float(b[0]), y=float(b[1]), z=float(b[2])) for b in box[:4]]
            marker.points.append(Point(x=float(box[0,0]), y=float(box[0,1]), z=float(box[0,2])))
            marker.points += [Point(x=float(b[0]), y=float(b[1]), z=float(b[2])) for b in box[4:]]
            marker.points.append(Point(x=float(box[4,0]), y=float(box[4,1]), z=float(box[4,2])))
            temp = [5,1,2,6,7,3]
            marker.points += [Point(x=float(box[i, 0]), y=float(box[i, 1]), z=float(box[i, 2])) for i in temp]

            marker.scale.x = float(line_width)
            # marker.scale.y = float(line_width)
            # marker.scale.z = float(line_width)

            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])

            markersArr.markers.append(marker)

        self.pubs_dict[topic].publish(markersArr)


# 下面测试使用
if __name__ == '__main__':
    rclpy.init()
    pub = PointCloudPublisher("test_pub", interval=5)
    import time
    np.set_printoptions(suppress=True)
    for i in range(30):
        bboxes = np.load(f"../{i}_gt.npy") # 只有GT，没有原始点云？
        bboxes = np.concatenate([np.zeros((bboxes.shape[0],1), dtype=np.float32), bboxes], axis=1)
        
        pub.update_counter()
        if pub.is_ok():
            print(bboxes[:,4:7])
            pub.publish_boxes2(bboxes, 'test') # topic名字修改
        time.sleep(1)

    rclpy.shutdown()





