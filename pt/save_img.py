#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2019/11/1 15:50
# @Author  : LionZhu

import os
import numpy as np
import scipy.misc as smc
import cv2
# from PIL import Image


def save_imgs(result_path, name_pre, label_batch, pred_batch, img_depth =1):
    # red is mask, blue is pred, green is pred*mask
    IMAGE_HEIGHT = label_batch.shape[-2]
    IMAGE_WIDTH = label_batch.shape[-1]
    str_split = name_pre.split('_')
    casePath = result_path + 'imgs/' +  str_split[0] + '_' + str_split[1] + '/'
    # casePath = resultPath + 'imgs/' + str_split[1]  + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    for dept in range(img_depth):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch
        pred_slice = pred_batch

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        # blue, false negative, loujian
        label_img_mat[0, label_cord[0], label_cord[1]] = 50
        label_img_mat[1, label_cord[0], label_cord[1]] = 50
        label_img_mat[2, label_cord[0], label_cord[1]] = 250

        pred_cord = np.where(pred_slice == 1)
        # green, false positive, wujian
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        pred_label = pred_slice * label_slice
        pred_cord = np.where(pred_label == 1)
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            casePath + str_split[-1] + '-seg.png')
        # cv2.imwrite(casePath + str_split[-1] + '-seg.png', label_img_mat)


def save_npys(res_path, name_pre, label_batch, pred_batch, score_batch= None):
    str_split = name_pre.split('_')
    casePath1 = res_path + 'npys/mask/' + str_split[0] + '_' + str_split[1] + '/'
    casePath2 = res_path + 'npys/pred/' + str_split[0] + '_' + str_split[1] + '/'
    casePath3 = res_path + 'npys/score/' + str_split[0] + '_' + str_split[1] + '/'
    if not (os.path.exists(casePath1)):
        os.makedirs(casePath1)
    if not (os.path.exists(casePath2)):
        os.makedirs(casePath2)
    np.save(casePath1 + str_split[-1] + '.npy', label_batch)
    np.save(casePath2 + str_split[-1] + '.npy', pred_batch.astype(np.uint8))
    if score_batch is not None:
        if not (os.path.exists(casePath3)):
            os.makedirs(casePath3)
        np.save(casePath3 + str_split[1] + '.npy', score_batch)



def save_imgs_3d(result_path, name_pre, label_batch, pred_batch, img_depth =1):
    # red is mask, blue is pred, green is pred*mask
    IMAGE_HEIGHT = label_batch.shape[-2]
    IMAGE_WIDTH = label_batch.shape[-1]
    casePath = result_path + 'imgs/' + name_pre + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    for dept in range(img_depth):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch[dept, ...]
        pred_slice = pred_batch[dept, ...]

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        # blue, false negative, loujian
        label_img_mat[0, label_cord[0], label_cord[1]] = 50
        label_img_mat[1, label_cord[0], label_cord[1]] = 50
        label_img_mat[2, label_cord[0], label_cord[1]] = 250

        pred_cord = np.where(pred_slice == 1)
        # green, false positive, wujian
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        pred_label = pred_slice * label_slice
        pred_cord = np.where(pred_label == 1)
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            casePath + '{}-seg.png'.format(dept))
        # cv2.imwrite(casePath + '{}-seg.png'.format(dept), label_img_mat)

def save_npys_3d(res_path, name_pre, label_batch, pred_batch, score_batch= None, img_depth = 1):
    casePath = res_path + 'npys/' + name_pre + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    np.save(casePath + 'mask.npy', label_batch)
    np.save(casePath + 'pred.npy', pred_batch.astype(np.uint8))
    if score_batch is not None:
        np.save(casePath +'score.npy', score_batch)