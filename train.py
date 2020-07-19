#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/13 17:01
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

Server = 0
#---------------------paths--------------------------------------------------
if Server == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = 'D:\datasets\diyiyiyuan\DWIFLAIR\more4.5h\dwi_npys/'
    out_path = path + 'exps/exp2/'
    npy_path = out_path + 'npys/'
    pkl_path = out_path + 'pkls/'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------Img parameters----------------------------------------------
img_paras = {'img_H': 128,
             'in_channel': 1 ,
             'class_num': 2,
             'drop': 0.0}

#------------training parameters----------------------------------------------
train_paras = {'Epoch':100,
               'BS':32,
               'lr': 1e-3,
               'ValFlag': True,
               'TestFlag': True
               }

SaveInterval = train_paras['Epoch'] // 2
DecayInterval = train_paras['Epoch'] // 10

model = ISSegNet(in_channel= img_paras['in_channel'], class_num= img_paras['class_num'], )
model.to(device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=train_paras['lr'])
weights = torch.Tensor([1,1]).to(device=device)
criterion = WeightedBCE(weight= weights)

# Dataset loader
transforms_ = [ transforms.ToPILImage(),
                transforms.Resize(int(img_paras['img_H']*1.12), Image.BICUBIC),
                transforms.RandomCrop(img_paras['img_H']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) ]

if __name__ == '__main__':
    lr= train_paras['lr']
    # train------------------------------------------------------------------------
    dataloader = DataLoader(ImageDataset(path),
                            batch_size=train_paras['BS'], shuffle=True, num_workers=1)
    model.train()
    Train_loss = []
    Epoch = []
    for epoch in range(train_paras['Epoch']):
        if epoch % DecayInterval == 0 and epoch > 0:
            lr = lr * 0.9
            adjust_lr(optimizer= optimizer, LR= lr)
        for i, batch in enumerate(dataloader):
            img_tensor = batch['img']       # NCHW, Tensor
            msk_tensor = batch['msk']       # NHW, Tensor
            img_tensor = img_tensor.to(device=device)
            msk_tensor = msk_tensor.to(device=device)
            probs = model(img_tensor)
            loss = criterion(probs, msk_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: {} | {}, Iter: {}, loss: {}'.format(epoch, train_paras['Epoch'], i, loss.item()))
        Train_loss.append(loss.item())
        Epoch.append(epoch)

        if (epoch+1) % SaveInterval == 0:
            torch.save(model.state_dict(), pkl_path + 'net_paras_epoch{}.pkl'.format(epoch+1))

    fig1 = plt.figure(1)
    plt.plot(Epoch, Train_loss, 'o-')
    plt.title('Train loss vs. Epoch')
    plt.pause(2)
    plt.draw()
    plt.savefig(out_path + 'loss.jpg')
    plt.close(fig1)

    # test -----------------------------------------------------------------------------
    if train_paras['TestFlag']:
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
