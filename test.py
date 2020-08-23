#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/14 10:19
# @Author  : LionZhu

import torch
import os
from loss import *
from ISSegNet import *
from nets.unet3d import *
import torchvision.transforms as transforms
from PIL import Image
from datasets import ImageDataset
from torch.utils.data import DataLoader
from solver import *
import matplotlib.pyplot as plt
from save_img import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4.5h\dwi_npys_3d/'
out_path = path + 'exps/exp1/'
res_path ='D:\datasets\diyiyiyuan\DWIFLAIR\more4.5h\dwi_npys_3d/\exps\exp1/'
restore_path = res_path + 'pkls/net_paras_epoch100.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----------------Img parameters----------------------------------------------
img_paras = {'img_H': 128,
             'img_D': 16,
             'in_channel': 1 ,
             'class_num': 2,
             'drop': 0.0}

#------------training parameters----------------------------------------------
train_paras = {'Epoch':100,
               'BS':16,
               'lr': 1e-3,
               'ValFlag': True,
               'TestFlag': True
               }


model = UNet(in_dim= img_paras['in_channel'], out_dim= img_paras['class_num'], num_filters= 4)
model.load_state_dict(torch.load(restore_path))
model.eval()
model.to(device = device)

if __name__ == '__main__':
    dataloader = DataLoader(ImageDataset(path,mode='test'),
                                    batch_size=train_paras['BS'], shuffle=False, num_workers=1)
    model.eval()
    for i, batch in enumerate(dataloader):
        img_tensor = batch['img']
        msk_tensor = batch['msk']
        img_names = batch['name_A']
        msk_names = batch['name_B']
        img_tensor = img_tensor.to(device=device)
        msk_tensor = msk_tensor.to(device=device)
        with torch.no_grad():
            probs = model(img_tensor)
            pred = torch.sigmoid(probs)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
        msk_batch = msk_tensor.cpu().numpy()
        for j in range(len(img_names)):
            label = np.squeeze(msk_batch[j, ...])
            seg = np.squeeze(pred[j, ...])
            names = img_names[j].split('\\')
            name_pre = names[-1]
            name_pre = name_pre[:-4]
            print("test_itr:", name_pre)
            save_imgs_3d(out_path, name_pre, label, seg, img_depth=img_paras['img_D'])
            save_npys_3d(out_path, name_pre, label, seg, img_depth=img_paras['img_D'])