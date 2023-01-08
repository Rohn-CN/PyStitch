import numpy as np
import cv2
from shapely.geometry import Polygon


class GeoTrans:
    def __init__(self, cfg, H=np.ones((3, 3))) -> None:
        self.H = H
        self.trans_mode = cfg._dict["TRANS_MODEL"]["GEO_TRANS_MODEL"]

    @staticmethod
    def get_corner_points_2d(w, h):
        return np.array([[0, 0], [w, 0], [w, h], [0, h]])

    @staticmethod
    def get_corner_points_3d(w, h):
        return np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])

    @staticmethod
    def cal_overlap(corner_points1, corner_points2):
        """
        :param corner_points1: 4 * 2
        :param corner_points2: 4 * 2
        :return:
        """
        polygon1 = Polygon(corner_points1).convex_hull
        polygon2 = Polygon(corner_points2).convex_hull
        if not polygon1.intersection(polygon2):
            return 0
        else:
            inter_area = polygon1.intersection(polygon2).area
            union_area = polygon1.union(polygon2).area
            return inter_area / union_area

    def transform_corner_points(self, w, h):
        corner_points_3d = self.get_corner_points_3d(w, h).T  # (3,4)
        assert corner_points_3d.shape == (3, 4), "false"
        corner_points_trans_3d = self.H @ corner_points_3d  #(3,4)
        assert corner_points_trans_3d.shape == (3, 4), "false"
        corner_points_trans_norm = (corner_points_trans_3d[:2, :] / corner_points_trans_3d[2, :]).T #(4. 2)
        assert corner_points_trans_norm.shape == (4, 2), "false"
        return corner_points_trans_norm

    def get_geo_transform_model(self,points_image,points_utm):
        # TODO: numpy 的解方程方法
        if self.trans_mode == "four_param":
            pass
        else:
            assert False,"specific mode not found"

    def points_image_to_utm(self,points_image,trans_model):
        if self.trans_mode == "four_param":
            assert trans_model.shape == (4, 1)
            A = trans_model[0]
            B = trans_model[1]
            C = trans_model[2]
            D = trans_model[3]
            x_utm = A * points_image[:, 0] - B * points_image[:, 1] + C
            y_utm = B * points_image[:, 0] + A * points_image[:, 1] + D
            return np.hstack([x_utm,y_utm])
        else:
            assert False,"specific mode not found"


