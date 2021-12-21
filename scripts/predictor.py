#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : train.py
#   Author      : VXallset
#   Created date: 2021-11-18
#   Description : demo for make a prediction using bisenet_v2
#
#================================================================
import os

import cv2

from core.bisenet_v2 import BiseNetV2
from core.config import cfg
from core.dataset import Dataset
import numpy as np


class TrackSegmentor:
    def __init__(self, model_path=cfg.TEST.MODEL_PATH):
        self.net = BiseNetV2(cfg)
        self.net.model.load_weights(model_path)

    def segment(self, img_path, output_path='demo.png'):
        try:
            ori_image = cv2.imread(img_path)
        except Exception:
            print("Reading image {} error! Please check if the input image path "
                  "is correct!".format(img_path))
            return -1

        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

        ori_shape = ori_image.shape

        pre_processed_img = Dataset.image_preporcess(ori_image, (1080, 1920))
        image_data = pre_processed_img[np.newaxis, ...].astype(np.float32)

        segment_prediction = self.net.inference(image_data)

        result_image = Dataset.image_postporcess(np.copy(segment_prediction[0, :, :]), target_size=ori_shape)

        if output_path is not None:
            cv2.imwrite(output_path, result_image)
        return result_image


if __name__ == '__main__':
    my_segmentor = TrackSegmentor()

    input_folder = "C:\\home\\dataset\\track_segmentation\\track_simple\\track_ori_imgs"

    output_folder = "../data/demo/"

    img_names = os.listdir(input_folder)
    for img_name in img_names:
        my_segmentor.segment(os.path.join(input_folder, img_name), os.path.join(output_folder, img_name))
        

