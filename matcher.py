import numpy as np
import cv2
from frame import Frame
from config import CFG


class Matcher:
    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.distance_method = cfg._dict["MATCH_PARAMETER"]["DISTANCE_MODE"]
        self.distance_threshold = cfg._dict["MATCH_PARAMETER"]["DISTANCE_THRESHOLD"]
        self.min_nums_goods = cfg._dict["MATCH_PARAMETER"]["MIN_MATCH_NUM"]
        self.detector = cfg._dict["MATCH_PARAMETER"]["DETECTOR"]
        self.trans_mode = cfg._dict["TRANS_MODEL"]["STITCH_TRANS_MODEL"]
        self.sift_keypoints_num = cfg._dict["MATCH_PARAMETER"]["SIFT_KEYPOINTS_NUM"]
        self.dst_pts = []
        self.src_pts = []
        self.M = np.eye(3)
        self.inliers = []
        self.goods = []

    def calc_frame_keypoints_info(self, frame, current_image):
        if self.detector == "sift":
            sift = cv2.SIFT_create()
            keypoints, descriptor = sift.detectAndCompute(current_image)
            frame.set_keypoints_info(keypoints, descriptor)
        else:
            assert False, "specific detector not found"
        return frame

    def get_good_match(self, frame1: Frame, frame2: Frame):
        matcher = cv2.DescriptorMatcher.create(self.distance_method)
        matches = matcher.knnMatch(frame1.descriptor, frame2.descriptor, 2)
        for match in matches:
            if len(match) == 2 and match[0].distance < match[1].distance * self.distance_threshold:
                self.goods.append(match[0])
        return self.goods

    def is_goods(self):
        if len(self.goods) >= self.min_nums_goods:
            return True
        return False

    def get_match_info(self, frame1: Frame, frame2: Frame):
        if self.is_goods():
            dst_pts = np.float32([frame1.keypoints[match.queryIdx].pt for match in self.goods])
            src_pts = np.float32([frame2.keypoints[match.trainIdx].pt for match in self.goods])
        else:
            assert False, "too few of good matches"
        homo, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        if self.trans_mode == "homo":
            pass
        elif self.trans_mode == "affine":
            homo[2, :] = np.array([0, 0, 1])
        else:
            assert False, 'specified trans_mode not found'
        self.dst_pts = dst_pts
        self.src_pts = src_pts
        self.M = homo
        self.inliers = inliers
