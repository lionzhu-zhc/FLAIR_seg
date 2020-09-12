import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from lovasz_losses import lovasz_softmax

# copy from https://github.com/yassouali/pytorch_segmentation
# and https://github.com/wolny/pytorch-3dunet


def to_onehot(array, num_class):
    if torch.is_tensor(array):
        res = F.one_hot(array.to(torch.int64), num_classes = num_class)
    else:
        tensor = torch.from_numpy(array)
        one_hots = F.one_hot(tensor.to(torch.int64), num_classes = num_class)  #NHWC
        res = one_hots.numpy()
    return res  # NHWC

def from_onehot(one_hot):
    if torch.is_tensor(one_hot):
        arr = torch.argmax(one_hot, axis=-1)   #NHWC
    else:
        arr = np.argmax(one_hot, axis= -1)
    return arr


class WeightCE(nn.Module):
    def __init__(self, weight=None):
        super(WeightCE, self).__init__()
        self.weight = weight

    def __call__(self, input, target):
        Criterion = torch.nn.CrossEntropyLoss(self.weight)
        if target.dtype != torch.long:
            target = target.long()
        loss = Criterion(input, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = to_onehot(target, output.size()[1])
        target = target.permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1).to(torch.float32)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, alpha = 0.4, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = WeightCE()
        self.alpha = alpha

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + self.alpha * dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


