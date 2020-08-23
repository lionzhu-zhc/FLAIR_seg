#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/16 17:15
# @Author  : LionZhu

import numpy as np
import matplotlib.pyplot as plt

path = 'D:\datasets\gulou_ctpcta/npys/train\ctp/20200304001394759_793.17.npy'
seg = np.load(path)
a = seg[8,...]
plt.imshow(a, cmap='gray', vmin=-0.5, vmax=1)
plt.show()
