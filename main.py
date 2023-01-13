from config import CFG
from stitch_param import StitchParameter
from frame import Frame
from merge import Merge
from geo_trans import GeoTrans
from matcher import Matcher
from frame_selection import Selector
from optimizer import Optimizer
import numpy as np
import data_io as io
import os


def get_data(image_dir, coords_path, force_zone_number, need_geo=True, ):
    image_list = io.get_image_list(image_dir)
    if need_geo:
        coords_list = io.get_coords_utm_list(coords_path, force_zone_number)
    else:
        coords_list = [np.zeros((2,)) for _ in range(len(image_list))]
    return image_list, coords_list


def stitch_init(cfg, stitch_param: StitchParameter, first_image, first_coords_utm):
    # 新的图像
    frame = Frame(cfg, first_image)
    frame.crop()
    frame.resize()
    detector = cfg._dict["MATCH_PARAMETER"]["DETECTOR"]
    matcher = Matcher(cfg)
    matcher.calc_frame_keypoints_info(detector, frame)

    # mask初始化
    stitch_param.set_mask(Merge.create_weight_mask(frame.w, frame.h))
    # H_bias和homo初始化
    stitch_param.set_H_bias(np.eye(3))
    stitch_param.add_homo(np.eye(3))

    # 轨迹初始化
    corner_points_2d = GeoTrans.get_corner_points_2d(frame.w, frame.h)
    stitch_param.add_trajectory(corner_points_2d)

    # 设置 mask_dst和 image_dst 和增加roi
    stitch_param.set_image_dst(frame.image)
    stitch_param.set_mask_dst(stitch_param.mask)
    stitch_param.add_roi(frame.image)
    stitch_param.add_roi_mask(stitch_param.mask.copy())

    # 设置画布大小
    stitch_param.set_map_size(frame.w, frame.h, 0, 0)

    # 坐标初始化
    stitch_param.add_top_n_center_points_2d_image(np.array([frame.w / 2, frame.h / 2]))
    stitch_param.add_top_n_center_points_2d_utm(np.array(first_coords_utm))
    # 加入frame
    stitch_param.add_frame(frame, del_image=True)
    stitch_param.add_stitch_count()
    # 添加四角坐标
    stitch_param.add_coords(np.zeros((4, 2)))
    print("init finish")


def stitch(cfg, stitch_param: StitchParameter, image_current, coords_current_utm):
    # 输入当前帧
    current_frame = Frame(cfg, image_current)
    current_frame.crop()
    current_frame.resize()
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
        stitch_param.add_frame(current_frame, del_image=True)
        # 单应
        H_last_frame = stitch_param.homo_list[-1] @ match_info_with_last_frame.M
        # 计算四角坐标并获取相关帧
        geo_trans = GeoTrans(cfg, H_last_frame)
        corner_points_trans_2d = geo_trans.transform_corner_points(current_frame.w, current_frame.h)
        relative_frame_idx = selector.get_relative_frames(current_frame, stitch_param, corner_points_trans_2d)
        # 利用相关帧进行优化
        #TODO:这里到底要不要添加relativeframe
        optimizer.min_reproject_optimizer(current_frame, stitch_param)
        # 计算并扩充画布，并计算需要修改的patch
        image_dst, mask_dst, image_src, mask_src = merge.auto_expand(
            stitch_param.H_bias, stitch_param.image_dst, stitch_param.mask_dst,
            stitch_param.map_size, stitch_param.homo_list[-1], stitch_param.mask, current_frame
        )
        stitch_param.set_image_dst(image_dst)
        stitch_param.set_mask_dst(mask_dst)
        H = stitch_param.H_bias @ stitch_param.homo_list[-1]
        patch_info = merge.calc_patch_info(H, current_frame.w, current_frame.h)

        if stitch_param.stitch_count < cfg["TRANS_MODEL"]["TRAIN_NUM"]:
            stitch_param.add_top_n_center_points_2d_image(np.array([current_frame.w / 2, current_frame.h / 2]))
            stitch_param.add_top_n_center_points_2d_utm(np.array(coords_current_utm))
            patch_corner_points_utm = np.zeros((4, 2))
            # stitch_param.add_coords(np.zeros((4, 2)), fake_coords=True)
        # TODO:完成后续部分，查看推送上去的init的中心点是不是设置成了w/2
        else:
            patch_corner_points_image = merge.get_patch_corner_points(stitch_param.homo_list[-1], stitch_param.w,
                                                                      stitch_param.h)
            trans_model = geo_trans.get_geo_transform_model(stitch_param.top_n_center_points_2d_image,
                                                            stitch_param.top_n_center_points_2d_utm)
            patch_corner_points_utm = geo_trans.points_image_to_utm(patch_corner_points_image, trans_model)
            # stitch_param.add_coords(patch_corner_points_utm, fake_coords=False)
        merge.patch_merge(stitch_param, image_src, mask_src, patch_info, patch_corner_points_utm)
        if cfg._dict["TRANS_MODEL"]["NEED_NORTH"]:
            patch_corner_points_utm = geo_trans.patch_points_rotate_north(patch_corner_points_utm)

        stitch_param.add_coords(patch_corner_points_utm)
        stitch_param.add_stitch_count()
        return key_frame_status
    else:
        return key_frame_status


def after_stitch(stitch_param: StitchParameter, image_save_dir, roi_save_dir, coords_save_path, first_write=False):
    assert len(stitch_param.corner_points_gps_list) == len(stitch_param.roi_list) and \
           len(stitch_param.corner_points_gps_list) == stitch_param.stitch_count, "数量"
    number_stitch = str(stitch_param.stitch_count).zfill(5)
    # 存坐标
    io.save_coords(coords_save_path, stitch_param.corner_points_gps_list[-1], number_stitch, first_write)
    print("存坐标"+ str(number_stitch) +"\n")
    # 存大图
    image_dst_name = os.path.join(image_save_dir, number_stitch + ".jpg")
    io.save_image(image_dst_name, stitch_param.image_dst)
    print("存大图" + str(number_stitch) + "\n")

    # 存小图
    roi_name = os.path.join(roi_save_dir, number_stitch + ".png")
    io.save_image_png(roi_name, stitch_param.roi_list[-1], stitch_param.roi_mask_list[-1])
    print("存小图" + str(number_stitch) + "\n")



if __name__ == "__main__":
    config_file = "/Users/ronghao/code/stitch/pystitch/configfile/config.yaml"

    cfg = CFG()
    cfg.from_config_yaml(config_path=config_file)
    # 图像和坐标读取和保存目录
    image_dir = cfg._dict["OTHER"]["IMAGE_DIR"]
    image_save_dir = cfg._dict["OTHER"]["IMAGE_SAVE_DIR"]
    roi_save_dir = cfg._dict["OTHER"]["ROI_SAVE_DIR"]
    coords_path = cfg._dict["OTHER"]["COORDS_PATH"]
    coords_save_path = cfg._dict["OTHER"]["COORDS_SAVE_PATH"]

    # 是否需要geo信息
    need_geo = cfg._dict["OTHER"]["NEED_GEO"]
    # 读取图像和坐标
    image_list, coords_list = get_data(image_dir, coords_path, cfg._dict["TRANS_MODEL"]["ZONE_NUMBER"])

    stitch_param = StitchParameter(cfg)
    first_image, first_coords = image_list[0], coords_list[0]
    stitch_init(cfg, stitch_param, first_image, first_coords)
    after_stitch(stitch_param, image_save_dir, roi_save_dir, coords_save_path, first_write=True)
    print("init finish")
    for i in range(1, len(image_list)):
        image_current = image_list[i]
        coords_current_utm = coords_list[i]
        key_frame_status = stitch(cfg, stitch_param, image_current, coords_current_utm)
        if key_frame_status == 1:
            after_stitch(stitch_param, image_save_dir, roi_save_dir, coords_save_path)
        else:
            continue
