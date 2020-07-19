#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/11 15:47
# @Author  : LionZhu

import os
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import matplotlib.pyplot as plt

path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4h5/'

pats = os.listdir(path + 'FLAIR')

for i in range(1, len(pats)):
    img = nib.load(path + 'FLAIR/' + pats[i] + '/I10.nii.gz')
    print(img.affine.shape)
    # roi = nib.load(path + 'FLAIR/' + pats[i] + '/I10.nii.gz')
    #
    # roi_data = roi.get_data()
    # roi_data = np.squeeze(roi_data)
    # # roi_data = np.transpose(roi_data, [1,0,2])
    # a= roi_data[:,:,9]
    # plt.imshow(a, cmap='gray', vmin= 100, vmax=1500)
    # plt.show()
    # if not os.path.exists(path + 'FLAIR/' + i + '/I10.nii.gz'):
    #     print('no' + i + '/I10.nii.gz')
    # if not os.path.exists(path + 'FLAIR/' + i + '/I10_roi.nii.gz'):
    #     print('no' + i + '/I10_roi.nii.gz')