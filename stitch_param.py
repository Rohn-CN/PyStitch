import numpy as np


class StitchParameter:
    def __init__(self):
        self.frame_list = []
        self.frame_trajectory = []
        self.relative_frame_overlap = []
        self.relative_frame_index = []
        self.homo_list = []
        self.H_bias = np.eye(3)
        self.w = 0
        self.h = 0
        self.map_size = dict["max_x":self.w, "max_y":self.h, "min_x":0, "min_y":0]

    def add_frame(self, frame, del_image=False):
        if del_image:
            del frame.image
        if not self.frame_list:
            self.w = frame.w
            self.h = frame.h
        self.frame_list.append(frame)

    def add_trajectory(self, corner_points_2d):
        self.frame_trajectory.append(corner_points_2d)

    def add_relative_frame_overlap(self, relative_frame_overlap):
        self.relative_frame_overlap.append(relative_frame_overlap)

    def add_relative_frame_index(self, relative_frame_idx):
        self.relative_frame_index.append(relative_frame_idx)

    def add_homo(self, homo):
        self.homo_list.append(homo)

    def set_map_size(self,max_x,max_y,min_x,min_y):
        self.map_size["max_x"] = max_x
        self.map_size["max_y"] = max_y
        self.map_size["min_x"] = min_x
        self.map_size["min_y"] = min_y
