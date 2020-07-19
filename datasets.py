import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#
#         self.files_A = sorted(glob.glob(os.path.join(root, '%s/cta' % mode) + '/*.*'))
#         self.files_B = sorted(glob.glob(os.path.join(root, '%s/ctp' % mode) + '/*.*'))
#
#     def __getitem__(self, index):
#         item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
#
#         if self.unaligned:
#             item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
#         else:
#             item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
#
#         return {'A': item_A, 'B': item_B}
#
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))
#

class ImageDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/img' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/seg' % mode) + '/*.*'))

    def __getitem__(self, index):
        img = (np.load(self.files_A[index % len(self.files_A)]))
        img = np.expand_dims(img, axis=0)
        name_img = self.files_A[index % len(self.files_A)]
        seg = (np.load(self.files_B[index % len(self.files_B)]))
        name_seg = self.files_B[index % len(self.files_B)]

        return {'img': img, 'msk': seg, 'name_A': name_img, 'name_B': name_seg}   # CHW, HW

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))