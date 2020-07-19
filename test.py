#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/14 10:19
# @Author  : LionZhu

import torch
import os
from loss import *
from ISSegNet import *
import torchvision.transforms as transforms
from PIL import Image
from datasets import ImageDataset
from torch.utils.data import DataLoader
from solver import *
import matplotlib.pyplot as plt
from save_img import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4h5\FLAIR_npys/'
out_path = path + 'exps/exp1/'
res_path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4h5\FLAIR_npys\exps\exp1/'
restore_path = res_path + 'pkls/net_paras_epoch20.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_paras = {'img_H': 256,
             'in_channel': 1 ,
             'class_num': 2,
             'drop': 0.0}
train_paras = {'Epoch':20,
               'BS':24,
               'ImgNum': 1800,
               'lr': 1e-3,
               'ValFlag': True,
               'TestFlag': True
               }


model = ISSegNet(in_channel= img_paras['in_channel'], class_num= img_paras['class_num'], )
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
            save_imgs(out_path, name_pre, label, seg)
            save_npys(out_path, name_pre, label, seg)
