#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/16 17:15
# @Author  : LionZhu

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random
random.seed(10)
import shutil
import torch
import pt.loss as loss

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
pred = np.load('D:\datasets\diyiyiyuan\DWIFLAIR\exp_data2/npys/flair\exps\exp1/npys\L_S50680(1)/8-pred.npy')
mask = np.load('D:\datasets\diyiyiyuan\DWIFLAIR\exp_data2/npys/flair\exps\exp1/npys\L_S50680(1)/8-mask.npy')
pred_num = np.count_nonzero(pred == 1)
mask_num = np.count_nonzero(mask == 1)
c = pred * mask
inter = np.count_nonzero(c)
print(pred_num)
print(mask_num)
print(inter)

# plt.imshow(a, cmap='gray', vmin=0, vmax=1)
# plt.show()
# print(a)
print('ok')