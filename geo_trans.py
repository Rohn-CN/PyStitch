import numpy as np
import cv2
from shapely.geometry import Polygon
from config import CFG

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
        corner_points_trans_3d = self.H @ corner_points_3d  # (3,4)
        assert corner_points_trans_3d.shape == (3, 4), "false"
        corner_points_trans_norm = (corner_points_trans_3d[:2, :] / corner_points_trans_3d[2, :]).T  # (4. 2)
        assert corner_points_trans_norm.shape == (4, 2), "false"
        return corner_points_trans_norm

    def get_geo_transform_model(self, points_list_image, points_list_utm):
        """
        :param points_list_image:  list (n, 2)
        :param points_list_utm:    list (n, 2)
        :return:
        """
        if points_list_utm[0][0] == 0 and points_list_utm[0][1] == 0:
            return np.zeros((4, 1))
        num_pts = len(points_list_image)
        if self.trans_mode == "four_param":
            A = np.zeros((num_pts * 2, 4))
            B = np.zeros((num_pts * 2, 1))
            for i in range(num_pts):
                x = points_list_image[i][0]
                y = - points_list_image[i][1]
                A[i * 2] = np.array([x, -y, 1, 0])
                A[i * 2 + 1] = np.array([y, x, 0, 1])
                B[i * 2] = points_list_utm[i][0]
                B[i * 2 + 1] = points_list_utm[i][1]
            res = np.linalg.lstsq(A, B, rcond=None)[0]
            return res
        else:
            assert False, "specific mode not found"


    def points_image_to_utm(self, points_image, trans_model):
        """
        :param points_image: (4, 2)
        :param trans_model: (4, 1)
        :return:
        """
        if self.trans_mode == "four_param":
            if not np.any(trans_model):
                return np.zeros((4, 2))
            assert trans_model.shape == (4, 1)
            A = trans_model[0]
            B = trans_model[1]
            C = trans_model[2]
            D = trans_model[3]
            x_utm = A * points_image[:, 0] - B * points_image[:, 1] + C
            y_utm = B * points_image[:, 0] + A * points_image[:, 1] + D
            return np.stack((x_utm, y_utm)).T
        else:
            assert False, "specific mode not found"


    @staticmethod
    def patch_points_rotate_north(patch_corner_points_utm_2d):
        """

        :param patch_corner_points_utm_2d: 4 * 2
        :return:
        """
        max_x = np.max(patch_corner_points_utm_2d[:, 0])
        max_y = np.max(patch_corner_points_utm_2d[:, 1])
        min_x = np.min(patch_corner_points_utm_2d[:, 0])
        min_y = np.min(patch_corner_points_utm_2d[:, 1])
        return np.array([[min_x, max_y],
                         [max_x, max_y],
                         [max_x, min_y],
                         [min_x, min_y]])


if __name__ == "__main__":
    config_file = r"D:\PyStitch\configfile\config.yaml"
    cfg = CFG()
    cfg.from_config_yaml(config_path=config_file)
    points_utm = np.random.random((10,2)).tolist()
    points_image = np.random.random((10,2)).tolist()
    x = GeoTrans(cfg, H = np.eye(3))
    res = x.get_geo_transform_model(points_image, points_utm)
    res2 = x.points_image_to_utm(np.array(points_image), res)
    print(res2 - points_utm)
    print(res)