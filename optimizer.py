import numpy as np

from geo_trans import GeoTrans
from frame import Frame
from stitch_param import StitchParameter
from matcher import Matcher


class Optimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def calc_reproject_error(H_dst, H_src, dst_pts, src_pts, inlier):
        """
        :param H_dst: 3 * 3
        :param H_src: 3 * 3
        :param dst_pts: n * 2
        :param src_pts: n * 2
        :param inlier : mask
        :return: n * 1
        """
        src_pts_3d = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
        dst_pts_3d = np.hstack([dst_pts, np.ones((dst_pts.shape[0], 1))])

        mask = np.squeeze(inlier).astype(np.bool)

        src_pts_3d_select_inlier = src_pts_3d[mask]
        dst_pts_3d_select_inlier = dst_pts_3d[mask]

        src_pts_3d_rep = (H_src @ src_pts_3d_select_inlier.T).T
        dst_pts_3d_rep = (H_dst @ dst_pts_3d_select_inlier.T).T  # n * 3

        src_pts_2d_rep = src_pts_3d_rep[:, :2] / src_pts_3d_rep[:, 2]
        dst_pts_2d_rep = dst_pts_3d_rep[:, :2] / dst_pts_3d_rep[:, 2]  # n ,2

        d = np.power(src_pts_2d_rep - dst_pts_2d_rep, 2)  # n, 2
        rep_err = np.sum(np.sqrt(d[:, 0] + d[:, 1]))
        return rep_err, d.shape[0]

    # TODO:完成这个函数

    def min_reproject_optimizer(self, current_frame: Frame, stitch_param: StitchParameter):
        relative_frame_idx = current_frame.relative_frame_index
        relative_frame_overlap = current_frame.relative_frame_overlap
        w = current_frame.w
        h = current_frame.h
        match_info_temp = []
        relative_frame_idx_temp = []
        relative_frame_overlap_temp = []
        error_current = []
        frame_temp = []
        for i in range(len(relative_frame_idx)):
            matcher = Matcher(self.cfg)
            goods = matcher.get_good_match(stitch_param.frame_list[relative_frame_idx[i]], current_frame)
            if matcher.is_goods():
                matcher.get_match_info(stitch_param.frame_list[relative_frame_idx[i]], current_frame)
                if matcher.M == np.eye(3):
                    continue
                else:
                    match_info_temp.append(matcher)
                    relative_frame_idx_temp.append(relative_frame_idx[i])
                    relative_frame_overlap_temp.append(relative_frame_overlap_temp[i])
                    frame_temp.append(stitch_param.frame_list[relative_frame_idx[i]])
            else:
                continue
        for i in range(len(relative_frame_idx)):
            num_of_pts = 0
            sum_of_err = 0
            H_j0 = match_info_temp[i].M
            H_j = stitch_param.homo_list[relative_frame_idx_temp[i]] @ H_j0
            for j in range(len(relative_frame_idx_temp)):
                if j != i:
                    err, num = self.calc_reproject_error(stitch_param.homo_list[relative_frame_idx_temp[j]],
                                                         H_j, match_info_temp[j].dst_pts,
                                                         match_info_temp[j].src_pts, match_info_temp[j].inliers)
                    num_of_pts += num / relative_frame_overlap_temp[j]
                    sum_of_err += err / relative_frame_overlap_temp[j]
            error_current.append(sum_of_err / num_of_pts)

        idx_of_min_error = error_current.index(min(error_current))
        idx_to_match = relative_frame_idx_temp[idx_of_min_error]

        H = stitch_param.homo_list[idx_to_match] @ match_info_temp[idx_of_min_error].M
        stitch_param.add_homo(H)
        corner_points_trans_2d = GeoTrans(self.cfg).transform_corner_points(w, h)
        stitch_param.add_trajectory(corner_points_trans_2d)
