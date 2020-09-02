#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/16 17:15
# @Author  : LionZhu

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

path = 'D:\datasets\gulou_ctpcta/npys/train\ctp/20200304001394759_793.17.npy'

# plt.imshow(a, cmap='gray', vmin=-0.5, vmax=1)
# plt.show()

a = np.array([[1,2,3,4],[5,6,7,8]])
b = Image.fromarray(a)
c = b.resize((4,8), Image.BILINEAR)
d = np.array(c)
print('ok')