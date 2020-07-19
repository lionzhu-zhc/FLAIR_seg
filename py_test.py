#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/16 17:15
# @Author  : LionZhu

import numpy as np
import matplotlib.pyplot as plt

path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4h5\FLAIR_npys/test/'
img = np.load(path+'/img/S60062-3_14.npy')
seg = np.load(path+'/seg/S59960-3_7.npy')

plt.imshow(img, cmap='gray', vmin= 500, vmax=1000)
plt.show()
plt.imshow(seg, cmap='gray')
plt.show()
print(img[seg==1])