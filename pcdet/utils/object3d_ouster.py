import numpy as np


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Truck': 1, 'Auxiliary': 2, 'Car': 3, 'Excavator': 4, 'Widebody': 5, 'Pedestrian': 6, 'Others': 7}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

### this is a temp function for their mistakes
### 1:Pedestrian:6 2:Truck:1 3:Widebody:5 4:Car:3 5:Auxiliary:2 6:Excavator:4
# def cls_type_to_id_temp(cls_type):
#     type_to_id = {'1': 6, '2': 1, '3': 5, '4': 3, '5': 2, '6': 4, 'Others': 7}
#     if cls_type not in type_to_id.keys():
#         return -1
#     return type_to_id[cls_type]

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        #self.cls_id = cls_type_to_id_temp(self.cls_type)
        #self.truncation = float(label[1])
        #self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        #self.alpha = float(label[3])
        #self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.l = float(label[4])
        self.w = float(label[5])
        self.h = float(label[6])
        self.loc = np.array((float(label[1]), float(label[2]), float(label[3])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[7])
        self.track_id = float(label[8])    # for future use
        self.score = float(label[9]) if label.__len__() == 10 else -1.0
        #self.level_str = None
        #self.level = self.get_kitti_obj_level()

    # def get_kitti_obj_level(self):
    #     height = float(self.box2d[3]) - float(self.box2d[1]) + 1
    #
    #     if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
    #         self.level_str = 'Easy'
    #         return 0  # Easy
    #     elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
    #         self.level_str = 'Moderate'
    #         return 1  # Moderate
    #     elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
    #         self.level_str = 'Hard'
    #         return 2  # Hard
    #     else:
    #         self.level_str = 'UnKnown'
    #         return -1

    # def generate_corners3d(self):
    #     """
    #     generate corners3d representation for this object
    #     :return corners_3d: (8, 3) corners of box3d in camera coord
    #     """
    #     l, h, w = self.l, self.h, self.w
    #     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    #     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    #     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    #
    #     R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
    #                   [0, 1, 0],
    #                   [-np.sin(self.ry), 0, np.cos(self.ry)]])
    #     corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    #     corners3d = np.dot(R, corners3d).T
    #     corners3d = corners3d + self.loc
    #     return corners3d

    def to_str(self):
        print_str = '%s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
