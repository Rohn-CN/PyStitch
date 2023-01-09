from config import CFG
from stitch_param import StitchParameter
from frame import Frame
from merge import Merge
from geo_trans import GeoTrans
from matcher import Matcher
from frame_selection import Selector
from optimizer import Optimizer
import numpy as np


def stitch_init(cfg, stitch_param: StitchParameter, first_image, first_coords_utm):
    # 新的图像
    frame = Frame(cfg, first_image)
    frame.crop()
    frame.resize()
    detector = cfg._dict["MATCH_PARAMETER"]["DETECTOR"]
    Matcher.calc_frame_keypoints_info(detector, frame)

    # mask初始化
    stitch_param.mask = Merge.create_weight_mask(frame.w, frame.h)
    # H_bias和homo初始化
    stitch_param.set_H_bias(np.eye(3))
    stitch_param.add_homo(np.eye(3))

    # 轨迹初始化
    corner_points_2d = GeoTrans.get_corner_points_2d(frame.w, frame.h)
    stitch_param.add_trajectory(corner_points_2d)

    # 设置 mask_dst和 image_dst 和增加roi
    stitch_param.set_image_dst(frame.image)
    stitch_param.set_mask_dst(stitch_param.mask)
    stitch_param.add_roi(frame)

    # 设置画布大小
    stitch_param.set_map_size(frame.w, frame.h, 0, 0)

    # 坐标初始化
    stitch_param.add_top_n_center_points_2d_image(np.array([frame.w / 2, frame.h / 2]))
    stitch_param.add_top_n_center_points_2d_utm(np.array(first_coords_utm))
    # 加入frame
    stitch_param.add_frame(frame, del_image=True)
    stitch_param.add_stitch_count()
    # 添加四角坐标
    stitch_param.add_coords(np.zeros((4,2)),fake_coords=True)


def stitch(cfg, stitch_param: StitchParameter, image_current, coords_current_utm):
    # 输入当前帧
    current_frame = Frame(cfg, image_current)
    current_frame.crop()
    current_frame.resize()
    # 初始化image和mask
    image_src = current_frame.image.copy()
    mask_src = stitch_param.mask.copy()
    # 计算是否关键帧
    selector = Selector(cfg)
    match_info_with_last_frame = Matcher(cfg)
    key_frame_status = selector.is_keyframe(stitch_param.frame_list[-1], current_frame, match_info_with_last_frame)
    # 初始化优化器
    optimizer = Optimizer(cfg)
    # 初始化融合器
    merge = Merge(cfg)
    if key_frame_status == 1:
        stitch_param.add_stitch_count()
        # 单应
        H_lastframe = stitch_param.homo_list[-1] @ match_info_with_last_frame.M
        # 计算四角坐标并获取相关帧
        geo_trans = GeoTrans(cfg, H_lastframe)
        corner_points_trans_2d = geo_trans.transform_corner_points(current_frame.w, current_frame.h)
        relative_frame_idx = selector.get_relative_frames(current_frame, stitch_param, corner_points_trans_2d)
        # 利用相关帧进行优化
        optimizer.min_reproject_optimizer(current_frame, stitch_param)
        # 计算并扩充画布，并计算需要修改的patch
        image_dst, mask_dst, image_src, mask_src = merge.auto_expand(
            stitch_param.H_bias, stitch_param.image_dst, stitch_param.mask_dst,
            stitch_param.map_size, stitch_param.mask, current_frame
        )
        stitch_param.set_image_dst(image_dst)
        stitch_param.set_mask_dst(mask_dst)
        H = stitch_param.H_bias @ stitch_param.homo_list[-1]
        patch_info = merge.calc_patch_info(H, current_frame.w, current_frame.h)

        merge.patch_merge(stitch_param, image_src, mask_src, patch_info)
        if stitch_param.stitch_count < cfg["TRANS_MODEL"]["TRAIN_NUM"]:
            stitch_param.add_top_n_center_points_2d_image(np.array([current_frame.w / 2, current_frame.h / 2]))
            stitch_param.add_top_n_center_points_2d_utm(np.array(coords_current_utm))
            stitch_param.add_coords(np.zeros((4, 2)), fake_coords=True)
        # TODO:完成后续部分，查看推送上去的init的中心点是不是设置成了w/2
        else:
            patch_corner_points_image = merge.get_patch_corner_points(stitch_param.homo_list[-1],stitch_param.w, stitch_param.h)
            trans_model = geo_trans.get_geo_transform_model(stitch_param.top_n_center_points_2d_image,stitch_param.top_n_center_points_2d_utm)
            patch_corner_points_utm = geo_trans.points_image_to_utm(patch_corner_points_image,trans_model)
            stitch_param.add_coords(patch_corner_points_utm)
        stitch_param.add_frame(current_frame, del_image=True)
        return key_frame_status
    else:
        return key_frame_status


if __name__ == "__main__":
    config_file = "/Users/ronghao/code/stitch/pystitch/configfile"
    mode = "local"

    cfg = CFG()
    cfg.from_config_yaml(config_path=config_file)
