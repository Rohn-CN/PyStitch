#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   frame_selector
@Time    :   2023/01/01 22:48:00
@Author  :   ronghao
@Version :   1.0
@Contact :   ronghaoli1997@qq.com
@Desc    :   当前文件作用
'''

import numpy as np
from matcher import Matcher
from geo_trans import GeoTrans
from stitch_param import StitchParameter
from config import CFG
from frame import Frame


class Selector:
    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.key_frame_status = {"too_few_matches": -1,
                                 "excessive_overlap": -2,
                                 "insufficient_overlap": -3,
                                 "is_keyframe": 1}

        self.min_iou = cfg._dict["STITCH_PARAMETER"]["MIN_OVERLAP"]
        self.max_iou = cfg._dict["STITCH_PARAMETER"]["MAX_OVERLAP"]

    def is_keyframe(self, frame1, frame2, match_info_with_last_frame: Matcher):
        """
        新加入的帧是否是关键帧
        :param frame1:
        :param frame2:
        :param match_info_with_last_frame:
        :return: 1: keyframe, -1: too few matches, -2: excessive overlap, -3: insufficient overlap
        """
        w = frame1.w
        h = frame1.h
        goods = match_info_with_last_frame.get_good_match(frame1=frame1, frame2=frame2)

        if not match_info_with_last_frame.is_goods():
            return self.key_frame_status["too_few_matches"]

        match_info_with_last_frame.get_match_info(frame1=frame1, frame2=frame2)
        H = match_info_with_last_frame.M
        geo_transformer = GeoTrans(self.cfg, H=H)
        corner_points_3d = geo_transformer.get_corner_points_2d(w, h)
        corner_points_3d_trans = geo_transformer.transform_corner_points(w, h)
        assert corner_points_3d.shape == (4, 2), "wrong shape of pts"
        iou = geo_transformer.cal_overlap(corner_points_3d, corner_points_3d_trans)
        if iou > self.max_iou:
            return self.key_frame_status["excessive_overlap"]
        elif iou < self.min_iou:
            return self.key_frame_status["insufficient_overlap"]
        else:
            # TODO:确定是最佳匹配后需要将匹配和关键帧添加进来，放在函数外进行
            return self.key_frame_status["is_keyframe"]

    def get_relative_frames(self, current_frame: Frame, stitch_param: StitchParameter, corner_points_reprojected_2d):
        """
        获取当前帧的相关帧
        :param self:
        :param current_frame:
        :param stitch_param:
        :param corner_points_reprojected_2d:
        :return:
        """
        overlap_list = []
        relative_overlap_threshold = self.cfg._dict["STITCH_PARAMETER"]["RELATIVE_FRAME_OVERLAP_THRESHOLD"]

        for corner_points_2d in stitch_param.frame_trajectory:
            overlap = GeoTrans.cal_overlap(corner_points_reprojected_2d, corner_points_2d)
            overlap_list.append(overlap)

        overlap_list = np.array(overlap_list)
        idx_argsort = overlap_list.argsort()[::-1]

        relative_frame_idx = []
        relative_frame_overlap = []
        for idx in idx_argsort:
            if overlap_list[idx] >= relative_overlap_threshold:
                relative_frame_idx.append(idx)
                relative_frame_overlap.append(overlap_list[idx])

        stitch_param.add_relative_frame_overlap(relative_frame_overlap)
        stitch_param.frame_list[-1].set_relative_frame(relative_frame_overlap, relative_frame_idx)
        current_frame.set_relative_frame(relative_frame_overlap, relative_frame_idx)
        return relative_frame_idx
