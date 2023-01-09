import os
import cv2
import numpy as np
class DataIO:

    def get_image_list(self,image_folder):
        image_name_list = sorted(os.listdir(image_folder))
        image_path_list = list(map(lambda image_name:os.path.join(image_folder,image_name),
                                   image_name_list))
        image_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            image_list.append(image)
        return image_list

    def save_image(self,save_path,image):
        cv2.imwrite(save_path,image)

    def save_image_png(self,save_path,image,mask):
        image_png = np.zeros((image.shape[0],image.shape[1], 4))
        alpha = np.zeros((image.shape[0],image.shape[1]))
        alpha[np.where(mask != 0)] = 255
        image_png[...,: 3] = image
        image_png[..., 3] = alpha
        cv2.imwrite(save_path, image_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])



