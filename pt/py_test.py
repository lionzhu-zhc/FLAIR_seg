#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/16 17:15
# @Author  : LionZhu

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random
random.seed(10)
import shutil
import torch
import pt.loss as loss
import scipy.io as sio

# fig1 = plt.figure(1)
# a = [0,1,2,3,4]
# plt.plot(a)
# plt.pause(1)
# plt.draw()
# plt.savefig('xx.jpg', dpi=200)
# plt.close(fig1)
# path = 'D:\datasets\diyiyiyuan\DWIFLAIR\dwi_npy2d_all\exps\exp1\imgs/'
# ori_path = 'D:\datasets\diyiyiyuan\DWIFLAIR/dwi_npy2d_all/test/'
# dst_path = 'D:\datasets\diyiyiyuan\DWIFLAIR/dwi_npy2d_all/valid/'
# ori_path2 = 'D:\datasets\diyiyiyuan\DWIFLAIR/flair_npy2d_all/test/'
# dst_path2 = 'D:\datasets\diyiyiyuan\DWIFLAIR/flair_npy2d_all/valid/'
#
# pats = os.listdir(path)
# random.shuffle(pats)

# for i in range (40):
#     pat = pats[i]
#     npys = os.listdir(ori_path+'img/')
#     for npy in npys:
#         splts = npy.split('_')
#         if pat == splts[0]:
#             shutil.copy(ori_path+'img/'+npy, dst_path+'img/'+npy)
#             shutil.copy(ori_path+'seg/'+npy, dst_path+'seg/'+npy)
#             shutil.copy(ori_path2+'img/'+npy, dst_path2+'img/'+npy)
#             shutil.copy(ori_path2+'seg/'+npy, dst_path2+'seg/'+npy)
#
# for i in range (40):
#     pat = pats[i]
#     npys = os.listdir(ori_path+'img/')
#     for npy in npys:
#         splts = npy.split('_')
#         if pat == splts[0]:
#             os.remove(ori_path+'img/'+npy)
#             os.remove(ori_path+'seg/'+npy)
#             os.remove(ori_path2 + 'img/' + npy)
#             os.remove(ori_path2+'seg/'+npy)


#
# plt.imshow(a, cmap='gray', vmin=-0.5, vmax=1)
# plt.show()

#
# img = cv2.imread('D:/12.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# cv2.imshow("img", img)
# cv2.waitKey(0)

path1 = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data/224x224xN/resize_mat\more45\dwi\seg/'
path2 = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data/224x224xN/resize_mat\more45/flair\seg/'
pat = 'S60021-3.mat'

dwi = sio.loadmat(path1+pat)
flair = sio.loadmat(path2+pat)

dmat = dwi['mat']
fmat = flair['mat']

for i in range (dmat.shape[0]):
    print(i)
    d = dmat[i, ...]
    f = fmat[i, ...]

    label_img_mat = np.zeros((3, 224, 224))
    label_img_mat[...] = 128
    d_cord = np.where(d == 1)
    # blue, dwi
    label_img_mat[0, d_cord[0], d_cord[1]] = 50
    label_img_mat[1, d_cord[0], d_cord[1]] = 50
    label_img_mat[2, d_cord[0], d_cord[1]] = 250

    f_cord = np.where(f == 1)
    # green, flair
    label_img_mat[0, f_cord[0], f_cord[1]] = 10
    label_img_mat[1, f_cord[0], f_cord[1]] = 210
    label_img_mat[2, f_cord[0], f_cord[1]] = 10

    label_img_mat = np.transpose(label_img_mat, [1, 2, 0])
    plt.imshow(label_img_mat.astype(np.uint8))
    plt.pause(1)


# npys = os.listdir(path+'img/')
# for name in npys:
#     names = name.split('_')
#     if names[-1] == 'dwi.npy':
#         dwi_img = np.load(path+'img/'+name)
#         dwi_seg = np.load(path+'seg/'+name)
#         np.save(dst_path1 + 'img/' + names[0]+'_'+names[1] + '.npy', dwi_img)
#         np.save(dst_path1 + 'seg/' + names[0]+'_'+names[1] + '.npy', dwi_seg)
#     if names[-1] == 'flair.npy':
#         flair_img = np.load(path+'img/'+name)
#         flair_seg = np.load(path+'seg/'+name)
#         np.save(dst_path2 + 'img/' + names[0]+'_'+names[1] + '.npy', flair_img)
#         np.save(dst_path2 + 'seg/' + names[0]+'_'+names[1] + '.npy', flair_seg)

# coding: utf-8

# import torch
# import torch.nn as nn
# import numpy as np
# import math
#
# # ----------------------------------- CrossEntropy loss: base
#
# loss_f = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False)
# # 生成网络输出 以及 目标输出
# output = torch.ones(2, 3, requires_grad=True) * 0.5      # 假设一个三分类任务，batchsize=2，假设每个神经元输出都为0.5
# target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
#
# loss = loss_f(output, target)
#
# print('--------------------------------------------------- CrossEntropy loss: base')
# print('loss: ', loss)
# print('由于reduce=False，所以可以看到每一个样本的loss，输出为[1.0986, 1.0986]')
#
#
# # 熟悉计算公式，手动计算第一个样本
# output = output[0].detach().numpy()
# output_1 = output[0]              # 第一个样本的输出值
# target_1 = target[0].numpy()
#
# # 第一项
# x_class = output[target_1]
# # 第二项
# exp = math.e
# sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2])
# log_sigma_exp_x = math.log(sigma_exp_x)
# # 两项相加
# loss_1 = -x_class + log_sigma_exp_x
# print('---------------------------------------------------  手动计算')
# print('第一个样本的loss：', loss_1)
#
#
# # ----------------------------------- CrossEntropy loss: weight
#
# weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
# loss_f = nn.CrossEntropyLoss(weight=weight, size_average=True, reduce=False)
# output = torch.ones(2, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
# target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
# loss = loss_f(output, target)
# print('\n\n--------------------------------------------------- CrossEntropy loss: weight')
# print('loss: ', loss)  #
# print('原始loss值为1.0986, 第一个样本是第0类，weight=0.6,所以输出为1.0986*0.6 =', 1.0986*0.6)
#
# # ----------------------------------- CrossEntropy loss: ignore_index
#
# loss_f_1 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=1)
# loss_f_2 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=2)
#
# output = torch.ones(3, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
# target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)
#
# loss_1 = loss_f_1(output, target)
# loss_2 = loss_f_2(output, target)
#
# print('\n\n--------------------------------------------------- CrossEntropy loss: ignore_index')
# print('ignore_index = 1: ', loss_1)     # 类别为1的样本的loss为0
# print('ignore_index = 2: ', loss_2)     # 类别为2的样本的loss为0