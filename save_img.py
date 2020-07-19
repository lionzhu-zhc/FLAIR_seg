#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2019/11/1 15:50
# @Author  : LionZhu

import os
import numpy as np
import scipy.misc as smc
# import cv2
from PIL import Image

def save_imgs_2d(result_path, name_pre, label_batch, pred_batch):
    IMAGE_DEPTH = 1
    IMAGE_HEIGHT = label_batch.shape[0]
    IMAGE_WIDTH = label_batch.shape[1]
    str_split = name_pre.split('_')

    casePath = result_path + 'imgs/' + (str_split[0]+ '_'+ str_split[1]) + '/'
    if not(os.path.exists(casePath)):
        os.makedirs(casePath)

    for dept in range(IMAGE_DEPTH):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        pred_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch
        pred_slice = pred_batch

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        label_img_mat[0, label_cord[0], label_cord[1]] = 255
        label_img_mat[1, label_cord[0], label_cord[1]] = 0
        label_img_mat[2, label_cord[0], label_cord[1]] = 0

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        pred_cord = np.where(pred_slice == 0)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 128

        pred_cord = np.where(pred_slice == 1)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 0
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 0
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 255

        pred_img_mat = np.transpose(pred_img_mat, [1, 2, 0])

        # dst = cv2.addWeighted(label_img_mat, 0.4, pred_img_mat, 0.6, 0)
        # cv2.imwrite(casePath + str_split[2]  + '-seg.png', dst)

        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            casePath + str_split[2]  + '-mask.png' )
        smc.toimage(pred_img_mat, cmin=0.0, cmax=255).save(
            casePath + str_split[2]  + '-pred.png' )

def save_imgs(result_path, name_pre, label_batch, pred_batch):
    # red is mask, blue is pred, green is pred*mask
    IMAGE_DEPTH = 1
    IMAGE_HEIGHT = label_batch.shape[0]
    IMAGE_WIDTH = label_batch.shape[1]
    str_split = name_pre.split('_')
    casePath = result_path + 'imgs/' +  str_split[0] + '/'
    # casePath = resultPath + 'imgs/' + str_split[1]  + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    for dept in range(IMAGE_DEPTH):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch
        pred_slice = pred_batch

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        # red, false negative, loujian
        label_img_mat[0, label_cord[0], label_cord[1]] = 255
        label_img_mat[1, label_cord[0], label_cord[1]] = 0
        label_img_mat[2, label_cord[0], label_cord[1]] = 0

        pred_cord = np.where(pred_slice == 1)
        # blue, false positive, wujian
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 0
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 0
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 255

        pred_label = pred_slice * label_slice
        pred_cord = np.where(pred_label == 1)
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 0
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 255
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 0

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            casePath + str_split[1] + '-seg.png')


def save_npys(res_path, name_pre, label_batch, pred_batch, score_batch= None):
    str_split = name_pre.split('_')
    casePath = res_path + 'npys/' + str_split[0] + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    np.save(casePath + str_split[1] + '-mask.npy', label_batch)
    np.save(casePath + str_split[1] + '-pred.npy', pred_batch)
    if score_batch is not None:
        np.save(casePath + str_split[1] + '-score.npy', score_batch)