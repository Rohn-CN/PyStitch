#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   frame.py
@Time    :   2023/01/01 21:55:41
@Author  :   ronghao
@Version :   1.0
@Contact :   ronghaoli1997@qq.com
@Desc    :   当前文件作用
'''
import cv2


class Frame:
    def __init__(self, cfg, image) -> None:
        self.cfg = cfg
        self.image = image
        self.h, self.w, self.c = self.image.shape
        self.relative_frame_overlap = []
        self.relative_frame_index = []
        self.resize_ratio = cfg._dict["IMAGE"]["RESIZE_RATIO"]
        self.crop_ratio = cfg._dict["IMAGE"]["CROP_RATIO"]

    def set_keypoints_info(self, keypoints, descriptor):
        self.keypoints = keypoints
        self.descriptor = descriptor

    def resize(self):
        self.image = cv2.resize(self.image, fx=self.resize_ratio, fy=self.resize_ratio)
        self.h, self.w, self.c = self.image.shape

    def crop(self):
        w = self.w * self.crop_ratio
        h = self.h * self.resize_ratio
        crop_border_w = (self.w - w) // 2
        crop_border_h = (self.h - h) // 2
        self.image = self.image[crop_border_w:crop_border_w + w, crop_border_h:crop_border_h + h, :]
        self.w = w
        self.h = h

    def set_relative_frame(self, relative_frame_overlap, relative_frame_index):
        self.relative_frame_index = relative_frame_index
        self.relative_frame_overlap = relative_frame_overlap
