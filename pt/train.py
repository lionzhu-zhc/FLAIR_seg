#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/13 17:01
# @Author  : LionZhu

from loss import *
from nets.ISSegNet import *
from nets.Unet import  *
from nets.cenet import *
import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
from solver import *
import matplotlib.pyplot as plt
from save_img import *

Server = 0
#---------------------paths--------------------------------------------------
if Server == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data2/npys/dwi/'
    data_path = path +'data/'
    out_path = path + 'exps/exp5/'
    npy_path = out_path + 'npys/'
    pkl_path = out_path + 'pkls/'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

elif Server == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = '/opt/zhc/dwi_flair/exp_data2/dwi/'
    data_path = path + 'data/'
    out_path = path + 'exps/exp4/'
    npy_path = out_path + 'npys/'
    pkl_path = out_path + 'pkls/'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------Img parameters----------------------------------------------
img_paras = {'img_H': 224,
             'in_channel': 1 ,
             'class_num': 2,
             'drop': 0.0}

#------------training parameters----------------------------------------------
train_paras = {'Epoch':1,
               'BS':16,
               'lr': 1e-3,
               'ValFlag': True,
               'TestFlag': True
               }

SaveInterval = train_paras['Epoch'] // 2 if train_paras['Epoch'] // 2 > 0 else 1
DecayInterval = train_paras['Epoch'] // 20 if train_paras['Epoch'] // 20 > 0 else 1

# model = ISSegNet(in_channel= img_paras['in_channel'], class_num= img_paras['class_num'], )
# model = UNet(n_channels=img_paras['in_channel'], n_classes= img_paras['class_num'])
model = CE_Net_(num_classes= img_paras['class_num'])
model.to(device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=train_paras['lr'])
criterion = CE_DiceLoss()

# Dataset loader-- useless---------------------------------------------------------------------------------------------
transforms_train = transforms.Compose(
                [ transforms.ToPILImage(),
                transforms.Resize(int(img_paras['img_H']*1.12), Image.BICUBIC),
                transforms.RandomCrop(img_paras['img_H']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

transforms_test = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.ToTensor()
                                        ])
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    lr = train_paras['lr']
    data_train = DataLoader(ImageDataset(data_path, aug= True, mode= 'train'),
                            batch_size=train_paras['BS'], shuffle=True, num_workers=1)
    Train_loss = []
    Valid_loss = []
    Ep = []
    iter = 0
    for epoch in range(train_paras['Epoch']):
        # train------------------------------------------------------------------------
        model.train()
        loss_sum = 0
        Ep.append(epoch)
        if epoch % DecayInterval == 0 and epoch > 0:
            lr = lr * 0.9
            adjust_lr(optimizer= optimizer, LR= lr)
        for i, batch in enumerate(data_train):
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
            loss_sum += loss.item()
        loss_sum = loss_sum / (i+1)
        Train_loss.append(loss_sum)

        # valid -------------------------------------------------------------------------------------------
        if train_paras['ValFlag']:
            model.eval()
            data_valid = DataLoader(ImageDataset(data_path, aug=False, mode='valid'),
                                   batch_size=train_paras['BS'], shuffle=False, num_workers=1)
            loss_sum2 = 0
            for i, batch in enumerate(data_train):
                img_tensor = batch['img']  # NCHW, Tensor
                msk_tensor = batch['msk']  # NHW, Tensor
                img_tensor = img_tensor.to(device=device)
                msk_tensor = msk_tensor.to(device=device)
                with torch.no_grad():
                    probs = model(img_tensor)
                    loss = criterion(probs, msk_tensor)
                loss_sum2 += loss.item()
            loss_sum2 = loss_sum2 / (i+1)
            # early stop------------------------------
            # if loss_sum2 > Valid_loss[-1] and len(Valid_loss) >= 1:
            #     torch.save(model.state_dict(), pkl_path + 'net_paras_earlystop{}.pkl'.format(epoch + 1))
            #     break
            Valid_loss.append(loss_sum2)

        if (epoch+1) % SaveInterval == 0:
            torch.save(model.state_dict(), pkl_path + 'net_paras_epoch{}.pkl'.format(epoch+1))

        fig1 = plt.figure(1)
        plt.plot(Ep, Train_loss, 'o-')
        plt.plot(Ep, Valid_loss, '+-')
        plt.legend(['TrainLoss', 'ValidLoss'], loc='upper right')
        plt.title('Loss vs. Epoch')
        plt.pause(1)
        plt.draw()
        plt.savefig(out_path + 'loss.jpg')
        plt.close(fig1)

    # test -----------------------------------------------------------------------------
    if train_paras['TestFlag']:
        data_test = DataLoader(ImageDataset(data_path, aug= False, mode='test'),
                                batch_size=train_paras['BS'], shuffle=False, num_workers=1)
        model.eval()
        for i, batch in enumerate(data_test):
            img_tensor = batch['img']
            msk_tensor = batch['msk']
            img_names = batch['name_A']
            msk_names = batch['name_B']
            img_tensor = img_tensor.to(device=device)
            msk_tensor = msk_tensor.to(device=device)
            with torch.no_grad():
                probs = model(img_tensor)
                pred = torch.softmax(probs, dim=1)
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

    # cal dice
    pats = os.listdir(npy_path)
    liver_label = 0
    liver_pred = 0
    liver_labPred = 0

    for j in range(len(pats)):
        npys = os.listdir(npy_path + pats[j])
        for img_i in range(0, len(npys), 2):
            label_batch = np.load(npy_path + pats[j] + '/' + npys[img_i])
            # label_batch = label_batch[0:110, 10:128]
            pred_batch = np.load(npy_path + pats[j] + '/' + npys[img_i+1])
            # pred_batch = pred_batch[0:110, 10:128]
            liver_label = liver_label + np.count_nonzero(label_batch == 1)
            liver_pred = liver_pred + np.count_nonzero(pred_batch == 1)

            label_bool = (label_batch == 1)
            pred_bool = (pred_batch == 1)
            # common = np.logical_and(label_bool, pred_bool)
            common = label_bool * pred_bool
            liver_labPred = liver_labPred + np.count_nonzero(common)
    liver_dice_coe = 2 * liver_labPred / (liver_label + liver_pred + 1e-6)
    with open(path + 'exps/dice.txt', 'a+') as resltFile:
        resltFile.write(out_path + ":  %.3f " %(liver_dice_coe) +
                        'label: {} pred: {} labpred: {} \n'.format(liver_label, liver_pred, liver_labPred))

    print('ok')