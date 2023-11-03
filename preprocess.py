# Author: Tiankai Yang <raymondyangtk@gmail.com>

import os
import cv2
from config import DefaultConfig

config = DefaultConfig()


def resize_image_and_mask(ori_image_dir, ori_mask_dir, saved_image_dir, saved_mask_dir, width, height):
    if not os.path.exists(saved_image_dir):
        os.makedirs(saved_image_dir)
    if not os.path.exists(saved_mask_dir):
        os.makedirs(saved_mask_dir)

    ori_name_list = os.listdir(ori_image_dir)

    for ori_name in ori_name_list:
        ori_image_path = os.path.join(ori_image_dir, ori_name)
        ori_mask_path = os.path.join(ori_mask_dir, ori_name)
        saved_image_path = os.path.join(saved_image_dir, ori_name)
        saved_mask_path = os.path.join(saved_mask_dir, ori_name)
        ori_image = cv2.imread(ori_image_path)
        ori_mask = cv2.imread(ori_mask_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(ori_image, (width, height))
        resized_mask = cv2.resize(ori_mask, (width, height))
        cv2.imwrite(saved_image_path, resized_image)
        cv2.imwrite(saved_mask_path, resized_mask)


if __name__ == '__main__':
    resize_image_and_mask(config.ori_kvair_seg_image_dir, config.ori_kvair_seg_masks_dir,
                          config.kvair_seg_image_dir, config.kvair_seg_masks_dir,
                          config.width, config.height)
