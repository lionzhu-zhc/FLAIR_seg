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


path = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\cls_npys\data/test/'
dst_path1 = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\cls_npys\data2\dwi/test/'
dst_path2 = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\cls_npys\data2/flair/test/'

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

a = np.load('D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\cls_npys\data/flair/train\img/L_S1040-1.npy')
a = np.expand_dims(a,axis=0)
print('ok')