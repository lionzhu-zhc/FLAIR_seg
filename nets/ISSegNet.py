#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2019/10/23 15:40
# @Author  : LionZhu

import torch
import torch.nn as nn
import torch.nn.functional as F

class ISSegNet(nn.Module):
    def __init__(self, in_channel, out_channel = 64, class_num = None, drop= 0.0, init_w = False):    # in: 1, out: 64
        super(ISSegNet, self).__init__()

        # pre_conv, downsampled to 1/2
        self.pre_conv = _PreConv(in_channel, out_channel, drop)

        # res block 1
        self.res_block_1 = _ResBlock(out_channel, out_channel*2, drop)
        # res block 2, downsample to 1/4
        self.res_block_2 = _ResBlock(out_channel*2, out_channel*2, drop)
        self.pool_1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        # res block 3
        self.res_block_3 = _ResBlock(out_channel*2, out_channel*4, drop)
        # res block 4, downsample to 1/8
        self.res_block_4 = _ResBlock(out_channel*4, out_channel*4, drop)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.aspp_1 = _ASPP_BR1(out_channel*4, out_channel*2, drop)
        self.aspp_2 = _ASPP_BR2(out_channel*4, out_channel*2, drop)
        self.aspp_3 = _ASPP_BR3(out_channel*4, out_channel*2, drop)
        self.aspp_4 = _ASPP_BR4(out_channel*4, out_channel*2, drop)

        self.conv_after_aspp = nn.Sequential(
            nn.Conv2d(in_channels= out_channel*8, out_channels= out_channel*2, kernel_size= 3, padding= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel*2, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels= out_channel*4, out_channels= out_channel*2, kernel_size= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel*2, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.up_1 = nn.Sequential(
            # nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True),
            nn.ConvTranspose2d(in_channels= out_channel*2, out_channels= out_channel*4, kernel_size= 1, stride= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel * 4, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel*2, kernel_size=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel*2, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.up_2 = nn.Sequential(
            # nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True),
            nn.ConvTranspose2d(in_channels= out_channel*4, out_channels= out_channel*2, kernel_size= 3,
                               stride= 2, padding= 1, output_padding= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel * 2, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.up_3 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(in_channels= out_channel*2, out_channels= out_channel, kernel_size= 4, stride= 4),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

        self.out_conv = nn.Conv2d(in_channels= out_channel, out_channels= class_num, kernel_size= 1, padding= 0)


        if init_w:
            self._init_weights()

    def forward(self, x):
        pre_conv = self.pre_conv(x)                     # 5->64, 1/2
        res_1 = self.res_block_1(pre_conv)              # 64->128
        res_2 = self.res_block_2(res_1)                 # 128->128
        pool_1 = self.pool_1(res_2)                     # 128->128, 1/4
        res_3 = self.res_block_3(pool_1)                # 128->256
        res_4 = self.res_block_4(res_3)                 # 256->256
        pool_2 = self.pool_1(res_4)                     # 256->256, 1/8

        aspp_1 = self.aspp_1(pool_2)
        aspp_2 = self.aspp_2(pool_2)
        aspp_3 = self.aspp_3(pool_2)
        aspp_4 = self.aspp_4(pool_2)
        aspp = torch.cat((aspp_1, aspp_2, aspp_3, aspp_4), dim= 1)      # 128-> 128*4

        aspp_conv = self.conv_after_aspp(aspp)                          # 128*4 -> 128

        up_1 = self.up_1(aspp_conv)                                     # 128->256, 1/8
        fuse_1 = up_1 + pool_2                                          # 256

        up_2 = self.up_2(fuse_1)                                        # 256->128, 1/2
        fuse_2 = up_2 + pool_1

        up_3 = self.up_3(fuse_2)                                        # 64->64, 1

        out = self.out_conv(up_3)                                       # 64 -> 2

        # out = F.softmax(out, dim= 1)    #softmax
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class _PreConv(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_PreConv, self).__init__()
        self.pre_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= out_channel, kernel_size= 3, stride= 2, padding= 1, dilation= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(num_features= out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.pre_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(num_features=out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.pre_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(num_features=out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.pre_conv_1(x)
        x = self.pre_conv_2(x)
        x = self.pre_conv_3(x)
        return x

class _ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_ResBlock,self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= 64, kernel_size= 1, padding= 0),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(num_features= 64, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, padding= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(num_features= 128, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(128, out_channel, 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= out_channel, kernel_size= 1, padding=  0),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.conv1_1(x)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        z = self.conv2(x)
        # return torch.cat((y,z), dim= 1)     #NCHW, c = out_channel*2
        return (y + z)

class _ASPP_BR1(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_ASPP_BR1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= out_channel, kernel_size= 3, padding= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _ASPP_BR2(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_ASPP_BR2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= 128, kernel_size= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(128, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels=out_channel, kernel_size= 3, padding= 2, dilation= 2),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels= 128, out_channels= out_channel, kernel_size= 1),
        #     nn.Dropout2d(drop),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(True)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class _ASPP_BR3(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_ASPP_BR3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= 128, kernel_size= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(128, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels=out_channel, kernel_size= 3, padding= 4, dilation= 4),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels= 128, out_channels= out_channel, kernel_size= 1),
        #     nn.Dropout2d(drop),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(True)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class _ASPP_BR4(nn.Module):
    def __init__(self, in_channel, out_channel, drop):
        super(_ASPP_BR4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channel, out_channels= 128, kernel_size= 1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(128, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels=out_channel, kernel_size= 3, padding= 8, dilation= 8),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channel, affine= True, track_running_stats= True),
            nn.ReLU(True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels= 128, out_channels= out_channel, kernel_size= 1),
        #     nn.Dropout2d(drop),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(True)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x