import numpy as np
import os


# ori_path = '/opt/zhc/dwi_flair/exp_data2/flair/exps/'
path = 'D:\datasets\diyiyiyuan\DWIFLAIR\exp_data\seg_npys/flair/'
data_path = path + 'data/'
out_path = path + 'exps/exp4/'
npy_path = out_path + 'npys/'

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
with open(path + 'exps/dice.txt', 'a+') as resltFile:
    resltFile.write(out_path + ":  %.3f " %(liver_dice_coe) +
                    'label: {} pred: {} labpred: {} \n'.format(liver_label, liver_pred, liver_labPred))

