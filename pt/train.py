#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/6/13 17:01
# @Author  : LionZhu

from loss import *
# from nets.ISSegNet import *
# from nets.Unet import  *
from nets.cenet import *
from nets.CM_Net import *
# import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
from solver import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from save_img import *
from tensorboardX import SummaryWriter

Server = 0
#---------------------paths--------------------------------------------------
if Server == 0:
    split_mark = '\\'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data/224x224xN\seg_npys/dwi/'
    data_path = path +'data/'
    out_path = path + 'seg_exps/exp10/'
    npy_path = out_path + 'npys/'
    pkl_path = out_path + 'pkls/'
    pretrain_path = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\seg_npys\dwi\exps\exp1\pkls/net_paras.pkl'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

elif Server == 1:
    split_mark = '/'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # path = '/media/omnisky/Data1/zhc2/dyyy/dwi/'
    path = '/mnt/zhc/dyyy/seg_npys/dwi/'
    data_path = path + 'data/'
    out_path = path + 'seg_exps/exp3/'
    npy_path = out_path + 'npys/'
    pkl_path = out_path + 'pkls/'
    pretrain_path = '/media/omnisky/Data/zhc2/diyiyiyuan/seg_npys/dwi/exps/exp2/pkls/net_paras.pkl'
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
train_paras = {'Epoch':200,
               'BS':16,
               'lr': 1e-4,
               'ValFlag': True,
               'TestFlag': True,
               'DeepSuperv': False
               }

SaveInterval = train_paras['Epoch'] // 2 if train_paras['Epoch'] // 2 > 0 else 1
DecayInterval = train_paras['Epoch'] // 20 if train_paras['Epoch'] // 20 > 0 else 1

# model = ISSegNet(in_channel= img_paras['in_channel'], class_num= img_paras['class_num'], )
# model = UNet(n_channels=img_paras['in_channel'], n_classes= img_paras['class_num'],
#              init_w= True, deep_supervise= train_paras['DeepSuperv'])
model = CMNet_OnePath(img_paras['in_channel'], img_paras['class_num'], [224,224], layers= [3, 3, 3, 3, 1])
# model = CE_Net_(num_classes= img_paras['class_num'])
# model.load_state_dict(torch.load(pretrain_path))
model.to(device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=train_paras['lr'])
# optimizer = torch.optim.SGD(model.parameters(), lr= train_paras['lr'], momentum= 0.9, weight_decay= 1e-4)
weights = torch.Tensor([0.1,1]).to(device=device)
criterion = CE_DiceLoss(alpha=0.4)
# criterion = WeightCE(weight= weights)

# Dataset loader-- useless---------------------------------------------------------------------------------------------
# transforms_train = transforms.Compose(
#                 [ transforms.ToPILImage(),
#                 transforms.Resize(int(img_paras['img_H']*1.12), Image.BICUBIC),
#                 transforms.RandomCrop(img_paras['img_H']),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor()])
#
# transforms_test = transforms.Compose([
#                                         transforms.ToPILImage(),
#                                         transforms.ToTensor()
#                                         ])
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    writer = SummaryWriter(out_path + 'logs/')
    lr = train_paras['lr']
    Train_loss = []
    Valid_loss = []
    Valid_Dice = []
    Ep = []
    iter = 0
    for epoch in range(train_paras['Epoch']):
        # train train train train------------------------------------------------------------------------
        model.train()
        loss_sum = 0
        Ep.append(epoch)
        if epoch % DecayInterval == 0 and epoch > 0:
            lr = lr * 0.9
            adjust_lr(optimizer= optimizer, LR= lr)

        data_train = DataLoader(ImageDataset(data_path, aug=True, mode='train'),
                                batch_size=train_paras['BS'], shuffle=True, num_workers=4)
        for i, batch in enumerate(data_train):
            img_tensor = batch['img']       # NCHW, Tensor
            msk_tensor = batch['msk']       # NHW, Tensor
            img_tensor = img_tensor.to(device=device)
            msk_tensor = msk_tensor.to(device=device)
            if train_paras['DeepSuperv']:
                probs, logits1, logits2, logits3 = model(img_tensor)
                loss1 = criterion(logits1, msk_tensor)
                loss2 = criterion(logits2, msk_tensor)
                loss3 = criterion(logits3, msk_tensor)
                loss = criterion(probs, msk_tensor) + loss1 + loss2 +loss3
            else:
                probs = model(img_tensor)
                loss = criterion(probs, msk_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: {} | {}, Iter: {}, loss: {}'.format(epoch, train_paras['Epoch'], i, loss.item()))
            loss_sum += loss.item()
        loss_sum = loss_sum / (i+1)
        Train_loss.append(loss_sum)
        writer.add_scalar('train_loss:', loss_sum, epoch)

        # valid valid valid  --------------------------------------------------------------------------------------
        if train_paras['ValFlag']:
            model.eval()
            data_valid = DataLoader(ImageDataset(data_path, aug=False, mode='valid'),
                                   batch_size=train_paras['BS'], shuffle=False, num_workers=1)
            loss_sum2 = 0
            liver_label = liver_pred = liver_labPred = 0
            for i, batch in enumerate(data_valid):
                img_tensor = batch['img']  # NCHW, Tensor
                msk_tensor = batch['msk']  # NHW, Tensor
                img_tensor = img_tensor.to(device=device)
                msk_tensor = msk_tensor.to(device=device)
                with torch.no_grad():
                    if train_paras['DeepSuperv']:
                        probs, logits1, logits2, logits3 = model(img_tensor)
                        loss1 = criterion(logits1, msk_tensor)
                        loss2 = criterion(logits2, msk_tensor)
                        loss3 = criterion(logits3, msk_tensor)
                        loss = criterion(probs, msk_tensor) + loss1 + loss2 + loss3
                    else:
                        probs = model(img_tensor)
                        loss = criterion(probs, msk_tensor)

                    pred = torch.softmax(probs, dim=1)
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.cpu().numpy().astype(np.uint8)
                msk_batch = msk_tensor.cpu().numpy().astype(np.uint8)
                loss_sum2 += loss.item()
                # cal validation dice
                liver_label = liver_label + np.count_nonzero(msk_batch == 1)
                liver_pred = liver_pred + np.count_nonzero(pred == 1)
                label_bool = (msk_batch == 1)
                pred_bool = (pred == 1)
                common = label_bool * pred_bool
                liver_labPred = liver_labPred + np.count_nonzero(common)
            loss_sum2 = loss_sum2 / (i+1)
            print('valid loss: {}'.format(loss_sum2))
            valid_dice = 2 * liver_labPred / (liver_label + liver_pred + 1e-6)
            print('valid dice: {}'.format(valid_dice))
            Valid_loss.append(loss_sum2)
            Valid_Dice.append(valid_dice)
            writer.add_scalar('valid_loss:', loss_sum2, epoch)
            writer.add_scalar('valid_dice', valid_dice, epoch)
            writer.add_scalar('lr', lr, epoch)
            # early stop------------------------------
            # if loss_sum2 > Valid_loss[-1] and len(Valid_loss) >= 1:
            #     torch.save(model.state_dict(), pkl_path + 'net_paras_earlystop{}.pkl'.format(epoch + 1))
            #     break


        # if (epoch+1) % SaveInterval == 0:
        #     torch.save(model.state_dict(), pkl_path + 'net_paras_epoch{}.pkl'.format(epoch+1))
        if not np.isnan(loss_sum):
            torch.save(model.state_dict(), pkl_path + 'net_paras.pkl')

    fig1 = plt.figure(1)
    plt.plot(Ep, Train_loss, 'r')
    plt.plot(Ep, Valid_loss, 'b')
    plt.legend(['TrainLoss', 'ValidLoss'], loc='upper right')
    plt.title('Loss vs. Epoch')
    plt.pause(1)
    plt.draw()
    plt.savefig(out_path + 'loss.jpg', dpi=200)
    plt.close(fig1)
    fig1 = plt.figure(2)
    plt.plot(Ep, Valid_Dice, 'b')
    plt.legend(['Valid_Dice'], loc='upper right')
    plt.title('Loss vs. Dice')
    plt.pause(1)
    plt.draw()
    plt.savefig(out_path + 'validdice.jpg', dpi=200)
    plt.close(fig1)

    # test test test test-----------------------------------------------------------------------------
    if train_paras['TestFlag']:
        data_test = DataLoader(ImageDataset(data_path, aug= False, mode='test'),
                                batch_size=train_paras['BS'], shuffle=False, num_workers=1)
        model.eval()
        for i, batch in enumerate(data_test):
            img_tensor = batch['img']
            msk_tensor = batch['msk']
            img_names = batch['name_A']
            img_tensor = img_tensor.to(device=device)
            with torch.no_grad():
                if train_paras['DeepSuperv']:
                    probs, logits1, logits2, logits3 = model(img_tensor)
                else:
                    probs = model(img_tensor)
                pred = torch.softmax(probs, dim=1)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()
            msk_batch = msk_tensor.numpy()
            for j in range(len(img_names)):
                label = np.squeeze(msk_batch[j, ...])
                seg = np.squeeze(pred[j, ...])
                names = img_names[j].split(split_mark)
                name_pre = names[-1]
                name_pre = name_pre[:-4]
                print("test_itr:", name_pre)
                save_imgs(out_path, name_pre, label, seg)
                save_npys(out_path, name_pre, label, seg)

    # cal dice----------------------------------------------------------------------------------------------
    pats = os.listdir(npy_path + 'mask/')
    liver_label = 0
    liver_pred = 0
    liver_labPred = 0
    for j in range(len(pats)):
        npys = os.listdir(npy_path + 'mask/' + pats[j])
        for img_i in range(0, len(npys)):
            label_batch = np.load(npy_path + 'mask/' + pats[j] + '/' + npys[img_i])
            # label_batch = label_batch[0:110, 10:128]
            pred_batch = np.load(npy_path + 'pred/' + pats[j] + '/' + npys[img_i])
            # pred_batch = pred_batch[0:110, 10:128]
            liver_label = liver_label + np.count_nonzero(label_batch == 1)
            liver_pred = liver_pred + np.count_nonzero(pred_batch == 1)

            label_bool = (label_batch == 1)
            pred_bool = (pred_batch == 1)
            # common = np.logical_and(label_bool, pred_bool)
            common = label_bool * pred_bool
            liver_labPred = liver_labPred + np.count_nonzero(common)
    liver_dice_coe = 2 * liver_labPred / (liver_label + liver_pred + 1e-6)
    print(out_path + ":  %.3f " %(liver_dice_coe))
    with open(path + 'seg_exps/dice.txt', 'a+') as resltFile:
        resltFile.write(out_path[-20:] + ":  %.3f " %(liver_dice_coe) +
                        'label: {} pred: {} labpred: {} \n'.format(liver_label, liver_pred, liver_labPred))

    print('ok')