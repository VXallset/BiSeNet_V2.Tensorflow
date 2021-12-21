#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : train.py
#   Author      : VXallset
#   Created date: 2021-11-18
#   Description : demo for train the bisenet_v2 network
#
#================================================================

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import shutil

from core.bisenet_v2 import BiseNetV2
from core.config import cfg
from core.dataset import Dataset
import cv2

trainset = Dataset('train')
logdir = "../data/log"
steps_per_epoch = len(trainset)

global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

bisenet_v2 = BiseNetV2(cfg)

def train_step(image_data, target, generate_segmentation=False, seg_filename=None):
    with tf.GradientTape() as tape:
        ret = bisenet_v2.compute_loss(image_data, target, generate_segmentation=generate_segmentation)
        total_loss = ret['total_loss']
        l2_loss = ret['l2_loss']
        segment_loss = ret['segment_loss']
        gradients = tape.gradient(total_loss, bisenet_v2.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, bisenet_v2.model.trainable_variables))
        train_info = "=> STEP %4d   lr: %.6f | segment_loss: %4.2f l2_loss: %4.2f | " \
                     "total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(), segment_loss, l2_loss, total_loss)

        # update learning rate
        global_steps.assign_add(1)

        if generate_segmentation:
            segmentation = np.array(ret['segment_prediction'])
            result_img = segmentation[0, :, :] * 255
            if not os.path.exists(os.path.dirname(seg_filename)):
                os.mkdir(os.path.dirname(seg_filename))
            cv2.imwrite(seg_filename, result_img)
            cv2.imwrite(seg_filename[:-4] + '_ori'+seg_filename[-4:], image_data[0, :, :, :] * 255)


        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/segment_loss", segment_loss, step=global_steps)
            tf.summary.scalar("loss/l2_loss", l2_loss, step=global_steps)

        writer.flush()
        return train_info

for epoch in range(cfg.TRAIN.EPOCHS):
    pbar = tqdm(trainset)
    i = 0
    for image_data, target in pbar:
        if i % 10 == 0:
            info = train_step(image_data, target, generate_segmentation=True,
                              seg_filename='../data/log/img/epoch{}_step{}.png'.format(epoch, i))
            pbar.set_description(info)
        else:
            info = train_step(image_data, target)
            pbar.set_description(info)

        i += 1
    bisenet_v2.model.save_weights("../core/data/bisenet_v2_epoch{}".format(epoch))