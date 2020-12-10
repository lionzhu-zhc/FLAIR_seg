#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/8/4 15:24
# @Author  : LionZhu

import torch.nn as nn
import torch
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class concaten(nn.Module):
    def __init__(self, in_c):
        super(concaten, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c//2, kernel_size=1)
        self.conv_br = nn.Conv2d(in_c, in_c//2, kernel_size=1)
    def forward(self, x, x_br):
        x = self.conv(x)
        x_br = self.conv_br(x_br)
        x = torch.cat([x,x_br], dim=1)
        return x

class UNet_2M(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, init_w = False, deep_supervise= False):
        super(UNet_2M, self).__init__()

        # down operation----
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)      # 1/2
        self.down2 = down(128, 256)     # 1/4
        self.down3 = down(256, 512)     # 1/8
        self.down4 = down(512, 512)     # 1/16

        self.inc_br = inconv(n_channels, 64)
        self.cat_1 =  concaten(64)
        self.down1_br = down(64, 128)  # 1/2
        self.cat_2 =  concaten(128)
        self.down2_br = down(128, 256)  # 1/4
        self.cat_3 =  concaten(256)
        self.down3_br = down(256, 512)  # 1/8
        self.cat_4 = concaten(512)
        self.down4_br = down(512, 512)  # 1/16
        self.cat_5 = concaten(512)

        # up operation----
        self.up1 = up(1024, 256)
        self.conv1 = nn.Sequential( nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.outc1 = outconv(64, n_classes)
        self.up2 = up(512, 128)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.outc2 = outconv(64, n_classes)
        self.up3 = up(256, 64)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.outc3 = outconv(64, n_classes)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        self.d_supervise = deep_supervise
        if init_w:
            self._init_weights()

    def forward(self, input1, input2):
        x1 = self.inc(input1)
        x1_br = self.inc_br(input2)
        x1 = self.cat_1(x1, x1_br)
        x2 = self.down1(x1)
        x2_br = self.down1_br(x1_br)
        x2 = self.cat_2(x2, x2_br)
        x3 = self.down2(x2)
        x3_br = self.down2_br(x2_br)
        x3 = self.cat_3(x3, x3_br)
        x4 = self.down3(x3)
        x4_br = self.down3_br(x3_br)
        x4 = self.cat_4(x4, x4_br)
        x5 = self.down4(x4)
        x5_br = self.down4_br(x4_br)
        x5 = self.cat_5(x5, x5_br)
        x = self.up1(x5, x4)        #1/8
        if self.d_supervise:
            logits_1 = self.conv1(x)
            logits_1 = F.interpolate(logits_1, scale_factor=8, mode='bilinear')
            logits_1 = self.outc1(logits_1)
        x = self.up2(x, x3)         #1/4
        if self.d_supervise:
            logits_2 = self.conv2(x)
            logits_2 = F.interpolate(logits_2, scale_factor=4, mode='bilinear')
            logits_2 = self.outc2(logits_2)
        x = self.up3(x, x2)         #1/2
        if self.d_supervise:
            logits_3 = self.conv3(x)
            logits_3 = F.interpolate(logits_3, scale_factor=2, mode='bilinear')
            logits_3 = self.outc3(logits_3)
        x = self.up4(x, x1)         #1
        x = self.outc(x)

        if self.d_supervise:
            return torch.sigmoid(x), torch.sigmoid(logits_1), torch.sigmoid(logits_2), torch.sigmoid(logits_3)
        else:
            return torch.sigmoid(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)