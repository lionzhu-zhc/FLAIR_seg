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

def randFlipud(img1, msk1, u=0.5):
    if random.random() < u:
        img1 = np.flipud(img1)
        msk1 = np.flipud(msk1)
    return img1, msk1

def randFliplr(img1, msk1, u=0.5):
    if random.random() < u:
        img1 = np.fliplr(img1)
        msk1 = np.fliplr(msk1)
    return img1, msk1

def randRotate90(img1, msk1, u=0.5):
    if random.random() < u:
        img1 = np.rot90(img1)
        msk1 = np.rot90(msk1)
    return img1, msk1


def randomFlipud(img1, msk1, img2, msk2, u=0.5):
    if random.random() < u:
        img1 = np.flipud(img1)
        msk1 = np.flipud(msk1)
        img2 = np.flipud(img2)
        msk2= np.flipud(msk2)
    return img1, msk1, img2, msk2

def randomFliplr(img1, msk1, img2, msk2, u=0.5):
    if random.random() < u:
        img1 = np.fliplr(img1)
        msk1 = np.fliplr(msk1)
        img2= np.fliplr(img2)
        msk2 = np.fliplr(msk2)
    return img1, msk1, img2, msk2

def randomRotate90(img1, msk1, img2, msk2, u=0.5):
    if random.random() < u:
        img1 = np.rot90(img1)
        msk1 = np.rot90(msk1)
        img2 = np.rot90(img2)
        msk2 = np.rot90(msk2)
    return img1, msk1, img2, msk2

class ImageDataset(Dataset):
    def __init__(self, root, aug=False, mode='train'):

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/img' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/seg' % mode) + '/*.*'))
        self.aug = aug

    def __getitem__(self, index):
        img = np.load(self.files_A[index % len(self.files_A)])
        seg = np.load(self.files_B[index % len(self.files_B)])
        if self.aug:
            img, seg = randFlipud(img, seg)
            img, seg = randFliplr(img, seg)
            img, seg = randRotate90(img, seg)
            img = np.ascontiguousarray(img, dtype=np.float32)
            seg = np.ascontiguousarray(seg, dtype=np.uint8)

        img = np.expand_dims(img, axis=0)   # expand img channel 1
        name_img = self.files_A[index % len(self.files_A)]
        name_seg = self.files_B[index % len(self.files_B)]

        return {'img': img, 'msk': seg, 'name_A': name_img, 'name_B': name_seg}   # CHW, HW or CDHW, DHW

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_2M(Dataset):
    # dataset for cross modal network
    def __init__(self, root_dwi, root_flair, aug=False, mode='train'):

        self.dwi_imgs = sorted(glob.glob(os.path.join(root_dwi, 'data/%s/img' % mode) + '/*.*'))
        self.dwi_segs = sorted(glob.glob(os.path.join(root_dwi, 'data/%s/seg' % mode) + '/*.*'))
        self.flair_imgs = sorted(glob.glob(os.path.join(root_flair, 'data/%s/img' % mode) + '/*.*'))
        self.flair_segs = sorted(glob.glob(os.path.join(root_flair, 'data/%s/seg' % mode) + '/*.*'))
        self.aug = aug

    def __getitem__(self, index):
        dwi_img = np.load(self.dwi_imgs[index % len(self.dwi_imgs)])
        dwi_seg = np.load(self.dwi_segs[index % len(self.dwi_segs)])
        flair_img = np.load(self.flair_imgs[index % len(self.flair_imgs)])
        flair_seg = np.load(self.flair_segs[index % len(self.flair_imgs)])
        if self.aug:
            dwi_img, dwi_seg, flair_img, flair_seg = randomFlipud(dwi_img, dwi_seg, flair_img, flair_seg)
            dwi_img, dwi_seg, flair_img, flair_seg = randomFliplr(dwi_img, dwi_seg, flair_img, flair_seg)
            dwi_img, dwi_seg, flair_img, flair_seg = randomRotate90(dwi_img, dwi_seg, flair_img, flair_seg)
            dwi_img = np.ascontiguousarray(dwi_img, dtype=np.float32)
            dwi_seg = np.ascontiguousarray(dwi_seg, dtype=np.uint8)
            flair_img = np.ascontiguousarray(flair_img, dtype=np.float32)
            flair_seg = np.ascontiguousarray(flair_seg, dtype=np.uint8)

        dwi_img = np.expand_dims(dwi_img, axis=0)   # expand img channel 1
        flair_img = np.expand_dims(flair_img, axis=0)   # expand img channel 1
        flair_img_name = self.flair_imgs[index % len(self.flair_imgs)]
        flair_seg_name = self.flair_segs[index % len(self.flair_segs)]

        return {'d_img': dwi_img, 'd_msk': dwi_seg, 'f_img':flair_img, 'f_msk':flair_seg, 'f_img_name': flair_img_name,
                'f_seg_name': flair_seg_name}   # CHW, HW or CDHW, DHW

    def __len__(self):
        return max(len(self.dwi_imgs), len(self.flair_imgs))