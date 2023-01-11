import os
import cv2
import numpy as np
import utm


def get_image_list(image_folder):
    image_name_list = sorted(os.listdir(image_folder))
    image_path_list = list(map(lambda image_name: os.path.join(image_folder, image_name),
                               image_name_list))
    image_list = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image_list.append(image)
    return image_list


def save_image(save_path, image):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path, image)


def save_image_png(save_path, image, mask):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_png = np.zeros((image.shape[0], image.shape[1], 4))
    alpha = np.zeros((image.shape[0], image.shape[1]))
    alpha[np.where(mask != 0)] = 255
    image_png[..., : 3] = image
    image_png[..., 3] = alpha
    cv2.imwrite(save_path, image_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def get_coords_utm_list(coords_file, force_zone_number):
    coords_gps = np.loadtxt(coords_file, delimiter=',')
    easting, northing, _, _ = utm.from_latlon(coords_gps[:, 0], coords_gps[:, 1], force_zone_number=force_zone_number)
    coords_utm = np.vstack([easting, northing]).T
    return coords_utm


def get_coords_gps_list(coords_file):
    coords_gps = np.loadtxt(coords_file, delimiter=',')
    return coords_gps


def save_coords(save_file, coord_gps, num_coord, first_write=False):
    lat, lng = coord_gps
    coord_str = num_coord + "," + str(lat) + "," + str(lng)
    with open(save_file, "a") as f:
        # 查看txt文件是否为空
        if not os.path.getsize(save_file) == 0:
            if first_write:
                f.truncate(0)
                f.write(coord_str)
            else:
                f.write("\n" + coord_str)
        else:
            f.write(coord_str)


if __name__ == '__main__':
    coords_utm = get_coords_utm_list(r"D:/Code2/dataset/865_coords.txt")
    save_file = "./coords.txt"
    save_coords(save_file, coords_utm[0], first_write=True)
    for i in range(1, coords_utm.shape[0]):
        save_coords(save_file, coords_utm[i])

    print(coords_utm.shape)
