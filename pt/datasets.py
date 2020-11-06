import glob
import random
import os

from torch.utils.data import Dataset
# from PIL import Image
import torchvision.transforms as transforms
import numpy as np



# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode='train'):
#         self.trans = transforms_
#         self.files_A = sorted(glob.glob(os.path.join(root, '%s/img' % mode) + '/*.*'))
#         self.files_B = sorted(glob.glob(os.path.join(root, '%s/seg' % mode) + '/*.*'))
#
#     def __getitem__(self, index):
#         img = np.load(self.files_A[index % len(self.files_A)])
#         img = np.expand_dims(img, axis=0)
#         name_img = self.files_A[index % len(self.files_A)]
#         seg = np.load(self.files_B[index % len(self.files_B)])
#         name_seg = self.files_B[index % len(self.files_B)]
#
#         return {'img': img, 'msk': seg, 'name_A': name_img, 'name_B': name_seg}   # CHW, HW or CDHW, DHW
#
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))


def randomFlipud(img, msk, u=0.5):
    if random.random() < u:
        img = np.flipud(img)
        msk = np.flipud(msk)
    return img, msk

def randomFliplr(img, msk, u=0.5):
    if random.random() < u:
        img = np.fliplr(img)
        msk = np.fliplr(msk)
    return img, msk

def randomRotate90(img, msk, u=0.5):
    if random.random() < u:
        img = np.rot90(img)
        msk = np.rot90(msk)
    return img, msk

class ImageDataset(Dataset):
    def __init__(self, root, aug=False, mode='train'):

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/img' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/seg' % mode) + '/*.*'))
        self.aug = aug

    def __getitem__(self, index):
        img = np.load(self.files_A[index % len(self.files_A)])
        seg = np.load(self.files_B[index % len(self.files_B)])
        if self.aug:
            img, seg = randomFlipud(img, seg)
            img, seg = randomFliplr(img, seg)
            img, seg = randomRotate90(img, seg)
            img = np.ascontiguousarray(img, dtype=np.float32)
            seg = np.ascontiguousarray(seg, dtype=np.uint8)

        img = np.expand_dims(img, axis=0)   # expand img channel 1
        name_img = self.files_A[index % len(self.files_A)]
        name_seg = self.files_B[index % len(self.files_B)]

        return {'img': img, 'msk': seg, 'name_A': name_img, 'name_B': name_seg}   # CHW, HW or CDHW, DHW

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))