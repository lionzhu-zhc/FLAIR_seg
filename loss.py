import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np

def to_onehot(array, num_class):
    if torch.is_tensor(array):
        res = F.one_hot(array.to(torch.int64), num_classes = num_class)
    else:
        tensor = torch.from_numpy(array)
        one_hots = F.one_hot(tensor.to(torch.int64), num_classes = num_class)  #NHWC
        res = one_hots.numpy()
    return res

def from_onehot(one_hot):
    if torch.is_tensor(one_hot):
        arr = torch.argmax(one_hot, axis=-1)   #NHWC
    else:
        arr = np.argmax(one_hot, axis= -1)
    return arr

class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def __call__(self, input, target, num_class):
        input = torch.argmax(input, dim=1)
        input = F.one_hot(input.to(torch.int64), num_classes=num_class).permute(0, 3, 1, 2)
        target = F.one_hot(target.to(torch.int64), num_classes=num_class).permute(0, 3, 1, 2)
        N = target.size(0)
        smooth = 1e-4

        input_flat = input.reshape(N, -1)
        target_flat = target.reshape(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1).float() + smooth) / \
               (input_flat.sum(1).float() + target_flat.sum(1).float() + smooth)
        # loss = 1 - loss.sum() / N
        loss = loss.sum() / N

        return loss





class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=False, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # # one hot code for gt
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         gt = gt.view((shp_y[0], 1, *shp_y[1:]))
        #
        #     if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
        #         # if this is the case then gt is probably already a one hot encoding
        #         y_onehot = gt
        #     else:
        #         gt = gt.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if net_output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(net_output.device.index)
        #         y_onehot.scatter_(1, gt, 1)
        #
        # if self.apply_nonlin == True:
        #     output = softmax_helper(net_output)
        #
        # # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        # w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        # intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", output, y_onehot)
        # union: torch.Tensor = w * (einsum("bcxyz->bc", output) + einsum("bcxyz->bc", y_onehot))
        # divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (
        # einsum("bc->b", union) + self.smooth)
        # gdc = divided.mean()
        #
        # return gdc



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=False, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))     # add to (b, 1,x,y,z)

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin:
            net_output = softmax_helper(net_output)
        else:
            net_output = torch.sigmoid(net_output)

        input = flatten(net_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)     # sum shape has no -1 dim of target
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)


class WeightedBCE(torch.nn.CrossEntropyLoss):
    """
        Network has to have NO NONLINEARITY!
        """
    def __init__(self, weight=None):
        super(WeightedBCE, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        num_classes = inp.size()[1]
        if target.size() != inp.size():
            target = to_onehot(target, num_class= num_classes)  #NHWC

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):  # inp NCHW -> NHWC
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes) #[*, C]

        target = target.view(-1, num_classes).float()
        wce_loss = torch.nn.BCEWithLogitsLoss(weight=self.weight)

        return wce_loss(inp, target)

class CE_Dice_Loss(nn.Module):
    def __init__(self, alpha = 0.8, weight = None):
        super(CE_Dice_Loss,self).__init__()
        self.alpha = alpha
        self.weight = weight
        self.ce = WeightedBCE(self.weight)
        self.dice = GDiceLossV2()

    def __call__(self, y_pred, y_true):
        a = self.ce(y_pred, y_true)
        b = self.dice(y_pred, y_true)
        return self.alpha * a + (1-self.alpha) * b