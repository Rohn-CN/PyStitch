import numpy as np
import utm

class StitchParameter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stitch_count = 0
        self.frame_list = []
        self.frame_trajectory = []
        self.corner_points_utm_list = []
        self.corner_points_gps_list = []
        self.relative_frame_overlap = []
        self.relative_frame_index = []
        self.roi_list = []
        self.roi_mask_list = []
        self.homo_list = []
        self.w = 0
        self.h = 0
        self.map_size = dict["max_x":self.w, "max_y":self.h, "min_x":0, "min_y":0]
        self.mask = np.zeros((self.h, self.w))
        self.top_n_center_points_2d_utm = []
        self.top_n_center_points_2d_image = []

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

    def set_map_size(self, max_x, max_y, min_x, min_y):
        self.map_size["max_x"] = max_x
        self.map_size["max_y"] = max_y
        self.map_size["min_x"] = min_x
        self.map_size["min_y"] = min_y

    def add_roi(self, roi):
        self.roi_list.append(roi)

    def add_roi_mask(self,roi_mask):
        self.roi_mask_list.append(roi_mask)

    def set_image_dst(self, image_dst):
        self.image_dst = image_dst

    def set_mask_dst(self,mask_dst):
        self.mask_dst = mask_dst

    def set_H_bias(self,H_bias):
        self.H_bias = H_bias

    def add_top_n_center_points_2d_utm(self, top_n_center_points_2d_utm:np.ndarray):
        self.top_n_center_points_2d_utm.append(top_n_center_points_2d_utm)

    def add_top_n_center_points_2d_image(self, top_n_center_points_2d_image:np.ndarray):
        self.top_n_center_points_2d_image.append(top_n_center_points_2d_image)

    def set_geo_trans_model(self, geo_trans_model):
        self.geo_tran_model = geo_trans_model

    def add_stitch_count(self):
        self.stitch_count += 1

    def add_coords(self,corner_points_utm:np.ndarray,fake_coords = False):
        if fake_coords:
            self.corner_points_gps_list.append(np.zeros((4, 2)))
            self.corner_points_utm_list.append(np.zeros((4,2)))
        else:
            zone_number = self.cfg._dict["TRANS_MODEL"]["ZONE_NUMBER"]
            self.corner_points_utm_list.append(corner_points_utm)
            corner_points_gps = utm.to_latlon(corner_points_utm[:,0],corner_points_utm[:,1],zone_number)
            self.corner_points_gps_list.append(corner_points_gps)
