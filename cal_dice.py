import numpy as np
import os

# ori_path = 'D:/datasets/stroke_sryzd/zd_all/data/128/'
ori_path = 'D:\datasets\diyiyiyuan\DWIFLAIR/flair_npy2d_all\exps/'
path = ori_path+ 'exp2/'
exp_path = path + 'npys/'
pats = os.listdir(exp_path)

liver_label = 0
liver_pred = 0
liver_labPred = 0

dice = []

for j in range(len(pats)):
    npys = os.listdir(exp_path + pats[j])
    for img_i in range(0, len(npys), 2):
        label_batch = np.load(exp_path + pats[j] + '/' + npys[img_i])
        # label_batch = label_batch[0:110, 10:128]
        pred_batch = np.load(exp_path + pats[j] + '/' + npys[img_i+1])
        # pred_batch = pred_batch[0:110, 10:128]
        print(np.count_nonzero(label_batch == 2))
        print(np.count_nonzero(pred_batch == 2))
        liver_label = liver_label + np.count_nonzero(label_batch == 1)
        liver_pred = liver_pred + np.count_nonzero(pred_batch == 1)

        label_bool = (label_batch == 1)
        pred_bool = (pred_batch == 1)
        # common = np.logical_and(label_bool, pred_bool)
        common = label_bool * pred_bool
        liver_labPred = liver_labPred + np.count_nonzero(common)

liver_dice_coe = 2*liver_labPred/(liver_label + liver_pred + 1e-6)
print("lesion_dice:", liver_dice_coe)
print("lesion_label", liver_label)
print("lesion_pred", liver_pred)
print("lesion_labPred", liver_labPred)

#     dice.append(liver_dice_coe)
#
# dice_arr = np.array(dice)
# dice_avg = np.mean(dice_arr)
# dice_std = np.std(dice_arr)
dice_std = 0

with open(ori_path + 'dice.txt', 'a+') as resltFile:
	resltFile.write(path + ":  %.3f , %.3f " %(liver_dice_coe, dice_std) +
                    'label: {} pred: {} labpred: {} \n'.format(liver_label, liver_pred, liver_labPred))


