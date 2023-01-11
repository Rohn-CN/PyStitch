import numpy as np
import cv2
import math
from geo_trans import GeoTrans
from frame import Frame
from stitch_param import StitchParameter


class Merge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pyramid_layer = cfg._dict["IMAGE"]["PYRAMID_LAYER"]

    def copy_border(self, image):
        """
        扩充边界，防止不是2的倍数
        :param image: Mat
        :return: image:Mat
        """
        h, w, c = image.shape
        size_bottom = h % int(math.pow(2, self.pyramid_layer))
        size_right = w % int(math.pow(2, self.pyramid_layer))

        if size_right != 0 or size_bottom != 0:
            size_bottom = int(math.pow(2, self.pyramid_layer)) - size_bottom
            size_right = int(math.pow(2, self.pyramid_layer)) - size_right
            image = cv2.copyMakeBorder(image, 0, size_bottom, 0, size_right, 0, 0)
        return image

    @staticmethod
    def create_patch(image, left_top, width, height):
        """

        :param image:
        :param left_top: list:[col,row]
        :param width:
        :param height:
        :return:
        """
        return image[left_top[1]:left_top[1] + height, left_top[0]:left_top[0] + width, :]

    @staticmethod
    # TODO：设计函数位置
    def create_weight_mask(w, h):
        mask = np.zeros((w, h))
        x0 = w // 2
        y0 = h // 2
        for i in range(w):
            dy = np.power(i - y0, 2)
            for j in range(h):
                dx = np.power(j - x0, 2)
                mask[i][j] = 1 - np.sqrt(dx + dy) / np.sqrt(x0 * x0 + y0 * y0)
        return mask

    def create_gauss_pyr(self, image):
        gauss_pyr = list()
        gauss_pyr.append(image.copy())
        G = image.copy()
        for i in range(self.pyramid_layer - 1):
            cv2.pyrDown(G, G)
            gauss_pyr.append(G)
        return gauss_pyr

    def create_laplace_pyr(self, image):
        gauss_pyr = self.create_gauss_pyr(image)

        laplace_pyr = list()
        laplace_pyr.append(gauss_pyr[self.pyramid_layer - 1].copy())

        for i in range(self.pyramid_layer - 1, 0, -1):
            GE = np.zeros((1, 1))
            cv2.pyrUp(gauss_pyr[i], GE)
            L = gauss_pyr[i - 1] - GE
            laplace_pyr.append(L)
        return laplace_pyr

    @staticmethod
    def merge_by_weight(image_dst, mask_dst, image_src, mask_src):
        """
        :param image_dst: w,h,3
        :param mask_dst: w,h
        :param image_src: w,h,3
        :param mask_src: w,h
        :return:
        """
        h, w, c = image_dst.shape
        # mask_src_3d = np.repeat(mask_src.reshape(1, w, h), 3, axis=0).transpose(1, 2, 0)
        # mask_dst_3d = np.repeat(mask_dst.reshape(1,w,h),3,axis=0).transpose(1,2,0)
        idx = mask_dst <= mask_src
        idx_3d = np.repeat(idx.reshape(1, h, w), 3, axis=0).transpose(1, 2, 0)
        mask_dst[idx] = mask_src[idx]
        image_dst[idx_3d] = image_src[idx_3d]

    def laplace_merge(self, image_dst, mask_dst, image_src, mask_src):
        self.copy_border(image_dst)
        self.copy_border(mask_dst)
        self.copy_border(image_src)
        self.copy_border(mask_src)

        diff_mask = mask_src - mask_dst

        image_src_pyr = self.create_laplace_pyr(image_src)
        image_dst_pyr = self.create_laplace_pyr(image_dst)
        diff_mask_pyr = self.create_gauss_pyr(diff_mask)
        merge_pyr = []
        for i in range(self.pyramid_layer):
            dst_pyr = image_dst_pyr[i]
            src_pyr = image_src_pyr[i]
            mask_pyr = diff_mask_pyr[self.pyramid_layer - 1 - i]
            merge_map = dst_pyr.copy()
            h, w = mask_pyr.shape
            idx = mask_pyr > 0
            idx_3d = np.repeat(idx.reshape(1, h, w), axis=0).transpose(1, 2, 0)
            merge_map[idx_3d] = src_pyr[idx_3d]
            merge_pyr.append(merge_map)

        rebuild = merge_pyr[0]
        for i in range(self.pyramid_layer):
            cv2.pyrUp(rebuild, rebuild)
            rebuild += merge_pyr[i]
        rebuild.astype(np.uint8)
        idx2 = mask_dst < mask_src
        mask_dst[idx2] = mask_src[idx2]

    def calc_patch_info(self, H, w, h):
        geo_trans = GeoTrans(self.cfg, H)
        corner_points_trans_2d = geo_trans.transform_corner_points(w, h)  # 4 * 2
        min_x = np.min(corner_points_trans_2d[:, 0])
        max_x = np.max(corner_points_trans_2d[:, 0])
        min_y = np.min(corner_points_trans_2d[:, 1])
        max_y = np.max(corner_points_trans_2d[:, 1])
        dx = max_x - min_x
        dy = max_y - min_y
        patch_info = dict()
        patch_info["left_top"] = [min_x, min_y]
        patch_info["patch_width"] = dx
        patch_info["patch_height"] = dy

        return patch_info

    def patch_merge(self, stitch_param: StitchParameter, image_src, mask_src, patch_info, patch_corner_points_utm):
        left_top = patch_info["left_top"]
        patch_width = patch_info["patch_width"]
        patch_height = patch_info["patch_height"]

        image_dst_patch = self.create_patch(stitch_param.image_dst, left_top, patch_width, patch_height)
        image_src_patch = self.create_patch(image_src, left_top, patch_width, patch_height)
        mask_dst_patch = self.create_patch(stitch_param.mask_dst, left_top, patch_width, patch_height)
        mask_src_patch = self.create_patch(mask_src, left_top, patch_width, patch_height)

        self.laplace_merge(image_dst_patch, mask_dst_patch, image_src_patch, mask_src_patch)

        image_dst_patch = self.create_patch(image_dst_patch, [0, 0], patch_width, patch_height)
        mask_dst_patch = self.create_patch(mask_dst_patch, [0, 0], patch_width, patch_height)
        # 复制到dst
        stitch_param.image_dst[left_top[0] + patch_height, left_top[1] + patch_width, :] = image_dst_patch
        stitch_param.mask_dst[left_top[0] + patch_height, left_top[1] + patch_width, :] = mask_dst_patch

        # 添加roi到stitch_param，检查是否需要正北输出

        if self.cfg._dict["TRANS_MODEL"]["NEED_NORTH"]:
            image_dst_patch_north = self.patch_rotate_north(image_dst_patch, patch_corner_points_utm)
            mask_dst_patch_north = self.patch_rotate_north(mask_dst_patch, patch_corner_points_utm)
            stitch_param.add_roi(image_dst_patch_north)
            stitch_param.add_roi_mask(mask_dst_patch_north)
        else:
            stitch_param.add_roi(image_dst_patch)
            stitch_param.add_roi_mask(mask_dst_patch)

    def auto_expand(self, H_bias, image_dst, mask_dst, map_size, H, mask, current_frame: Frame):
        image = current_frame.image
        w = current_frame.w
        h = current_frame.h
        geo_trans = GeoTrans(self.cfg, H)
        corner_points_trans_2d = geo_trans.transform_corner_points(w, h)
        max_x = np.max(corner_points_trans_2d[:, 0])
        min_x = np.min(corner_points_trans_2d[:, 0])
        max_y = np.max(corner_points_trans_2d[:, 1])
        min_y = np.min(corner_points_trans_2d[:, 1])

        map_min_x = map_size["min_x"]
        map_max_x = map_size["max_x"]
        map_min_y = map_size["min_y"]
        map_max_y = map_size["max_y"]

        if max_x < map_max_x and max_y < map_max_y and min_x > map_min_x and min_y > map_min_y:
            map_width = map_max_x - map_min_x
            map_height = map_max_y - map_min_y
            image_dst_temp = image_dst.copy()
            mask_dst_temp = mask_dst.copy()
        else:
            top_expand = map_min_y - int(min_y) if min_y <= map_min_y else 0
            map_min_y = int(min_y) if min_y <= map_min_y else map_min_y
            left_expand = map_min_x - int(min_x) if min_x <= map_min_x else 0
            map_min_x = int(min_x) if min_x <= map_min_x else map_min_x
            bottom_expand = int(max_x) + 1 - map_max_x if max_x >= map_max_x else 0
            map_max_x = int(max_x) + 1 if max_x >= max_x >= map_max_x else map_max_x
            right_expand = int(max_y) + 1 - map_max_y if max_y >= map_max_y else 0
            map_max_y = int(max_y) + 1 if max_y >= map_max_y else map_max_y

            image_dst_temp = cv2.copyMakeBorder(image_dst, top_expand, bottom_expand, left_expand, right_expand)
            mask_dst_temp = cv2.copyMakeBorder(mask_dst, top_expand, bottom_expand, left_expand, right_expand)
            map_width = map_max_x - map_min_x
            map_height = map_max_y - map_min_y

            H_bias[0, 2] = -int(map_min_x)
            H_bias[1, 2] = -int(map_min_y)

        map_size["min_x"] = map_min_x
        map_size["max_x"] = map_max_x
        map_size["min_y"] = map_min_y
        map_size["max_y"] = map_max_y
        image_src_temp = cv2.warpPerspective(image, H_bias, (map_width, map_height))
        mask_src_temp = cv2.warpPerspective(mask, H_bias, (map_width, map_height))
        return image_dst_temp, mask_dst_temp, image_src_temp, mask_src_temp

    def get_patch_corner_points(self, H, w, h):
        geo_trans = GeoTrans(self.cfg, H)
        patch_corner_points_trans_2d = geo_trans.transform_corner_points(w, h)
        max_x = np.max(patch_corner_points_trans_2d[:, 0])
        min_x = np.min(patch_corner_points_trans_2d[:, 0])
        max_y = np.max(patch_corner_points_trans_2d[:, 1])
        min_y = np.min(patch_corner_points_trans_2d[:, 1])
        return np.array([min_x, - min_y], [max_x, -min_y], [max_x, -max_y], [min_x, -max_y])

    @staticmethod
    def patch_rotate_north(patch: np.ndarray, patch_corner_points_utm_2d):
        """
        TODO: 需要注意这里坐标参数，是否后面需要复用，需要的话传参应该是copy
        :param patch:
        :param patch_corner_points_utm_2d:
        :return:
        """
        if not np.any(patch_corner_points_utm_2d):
            res = patch
        else:
            h, w, c = patch.shape
            max_x = np.max(patch_corner_points_utm_2d[:, 0])
            min_x = np.min(patch_corner_points_utm_2d[:, 0])
            max_y = np.max(patch_corner_points_utm_2d[:, 1])
            min_y = np.min(patch_corner_points_utm_2d[:, 1])

            width = max_x - min_x
            height = max_y - min_y

            left_top = [min_x, max_y]

            patch_corner_points_image_2d = np.array([0, 0], [w, 0], [w, h], [0, h])
            resize_ratio = max(w / width, h / height)

            patch_corner_points_utm_2d[:, 0] -= min_x
            patch_corner_points_utm_2d[:, 1] -= min_y
            patch_corner_points_utm_2d[:, 1] = height - patch_corner_points_utm_2d[:, 1]
            patch_corner_points_utm_2d *= resize_ratio

            homo, mask = cv2.findHomography(patch_corner_points_image_2d, patch_corner_points_utm_2d)
            res = cv2.warpPerspective(patch, homo, (int(resize_ratio * width), int(resize_ratio * height)))
        return res
