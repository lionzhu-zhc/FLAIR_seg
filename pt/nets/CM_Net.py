# cross modal segnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # residual block
    expansion = 2

    def __init__(self, inplanes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * Bottleneck.expansion, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(inplanes * Bottleneck.expansion, inplanes * Bottleneck.expansion,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes * Bottleneck.expansion, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(inplanes * Bottleneck.expansion, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckA(nn.Module):
    # with atrous conv
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * BottleneckA.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * BottleneckA.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckB(nn.Module):
    # side path with conv1x1  + atrous conv
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * BottleneckB.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * BottleneckB.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * BottleneckB.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * BottleneckB.expansion)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CMNet_OnePath(nn.Module):
    def __init__(self, IN_CHANNEL, NUM_CLASS, INPUT_SIZE, layers):
        super(CMNet_OnePath, self).__init__()

        self.input_size = INPUT_SIZE
        # stem net
        self.d_conv1 = nn.Conv2d(IN_CHANNEL, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.d_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.d_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.d_bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.d_relu = nn.ReLU(inplace=True)

        # stage 1
        self.d_1x_stage1 = self._make_layer(Bottleneck, 64, layers[0])  # out:64

        # stage 2
        self.d_1x_stage2 = self._make_layer(Bottleneck, 64, layers[1])  # out:64
        self.d_2x_fuse_stage2 = conv3x3(in_planes= 64, out_planes= 128, stride= 2)
        self.d_2x_stage2 = self._make_layer(Bottleneck, 128, layers[1])  # out 128

        # stage 3
        self.d_1x_fuse_stage3 = conv3x3(in_planes= 128, out_planes= 64)
        self.d_1x_stage3 = self._make_layer(Bottleneck, 64, layers[2])  # out 128
        self.d_2x_fuse_stage3 = conv3x3(64, 128, stride= 2)
        self.d_2x_stage3 = self._make_layer(Bottleneck, 128, layers[2]) # out 128
        self.d_4x_fuse1_stage3 = conv3x3(in_planes= 64, out_planes= 256, stride= 4)
        self.d_4x_fuse2_stage3 = conv3x3(in_planes= 128, out_planes= 256, stride= 2)
        self.d_4x_stage3 = self._make_layer(Bottleneck, 256, layers[2])  # out 256

        # stage 4
        self.d_1x_fuse1_stage4 = conv3x3(in_planes=128, out_planes=64)
        self.d_1x_fuse2_stage4 = conv3x3(in_planes=256, out_planes=64)
        self.d_1x_stage4 = self._make_layer(Bottleneck, 64, layers[3])
        self.d_2x_fuse1_stage4 = conv3x3(64, 128, stride= 2)
        self.d_2x_fuse2_stage4 = conv3x3(256, 128)
        self.d_2x_stage4 = self._make_layer(Bottleneck, 128, layers[3])
        self.d_4x_fuse1_stage4 = conv3x3(64, 256, stride= 4)
        self.d_4x_fuse2_stage4 = conv3x3(128, 256, stride= 2)
        self.d_4x_stage4 = self._make_layer(Bottleneck, 256, layers[3])

        # stage 5
        self.d_1x_fuse1_stage5 = conv3x3(in_planes=128, out_planes=64)
        self.d_1x_fuse2_stage5 = conv3x3(in_planes=256, out_planes=64)
        self.d_1x_stage5 = self._make_layer(BasicBlock, 64, layers[4])
        self.d_2x_fuse1_stage5 = conv3x3(64, 128, stride=2)
        self.d_2x_fuse2_stage5 = conv3x3(256, 128)
        self.d_2x_stage5 = self._make_layer(BasicBlock, 128, layers[4])
        self.d_4x_fuse1_stage5 = conv3x3(64, 256, stride=4)
        self.d_4x_fuse2_stage5 = conv3x3(128, 256, stride=2)
        self.d_4x_stage5 = self._make_layer(BasicBlock, 256, layers[4])

        # last conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels= 448, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(256, momentum= BN_MOMENTUM),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 256, out_channels= NUM_CLASS, kernel_size= 1, stride= 1, padding= 1)
        )



    def forward(self, x, ):
        d_stem = self.d_conv1(x)
        d_stem = self.d_bn1(d_stem)
        d_stem = self.d_relu(d_stem)
        d_stem = self.d_conv2(d_stem)
        d_stem = self.d_bn2(d_stem)
        d_stem = self.d_relu(d_stem)

        # stage 1 *********************************************
        d_1x_stage1 = self.d_1x_stage1(d_stem)

        # stage 2 *********************************************
        d_1x_stage2 = self.d_1x_stage2(d_1x_stage1)

        d_2x_fuse_stage2 = self.d_2x_fuse_stage2(d_1x_stage1)
        d_2x_stage2 = self.d_2x_stage2(d_2x_fuse_stage2)

        # stage 3 *********************************************
        d_1x_stage3_fuse = F.interpolate(d_2x_stage2, scale_factor=2,
                                         mode='bilinear', align_corners= True)
        d_1x_stage3_fuse = self.d_1x_fuse_stage3(d_1x_stage3_fuse)
        d_1x_stage3_fuse = d_1x_stage3_fuse + d_1x_stage2
        d_1x_stage3 = self.d_1x_stage3(d_1x_stage3_fuse)

        d_2x_stage3_fuse = self.d_2x_fuse_stage3(d_1x_stage2)
        d_2x_stage3_fuse = d_2x_stage3_fuse + d_2x_stage2
        d_2x_stage3 = self.d_2x_stage3(d_2x_stage3_fuse)

        d_4x_fuse1_stage3 = self.d_4x_fuse1_stage3(d_1x_stage2)
        d_4x_fuse2_stage3 = self.d_4x_fuse2_stage3(d_2x_stage2)
        d_4x_stage3_fuse = d_4x_fuse1_stage3 + d_4x_fuse2_stage3
        d_4x_stage3 = self.d_4x_stage3(d_4x_stage3_fuse)

        # stage 4
        d_1x_stage4_fuse1 = F.interpolate(d_2x_stage3, scale_factor=2,
                                         mode='bilinear', align_corners=True)
        d_1x_stage4_fuse1 = self.d_1x_fuse1_stage4(d_1x_stage4_fuse1)
        d_1x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage4_fuse2 = self.d_1x_fuse2_stage4(d_1x_stage4_fuse2)
        d_1x_stage4_fuse = d_1x_stage3 + d_1x_stage4_fuse1 + d_1x_stage4_fuse2
        d_1x_stage4 = self.d_1x_stage4(d_1x_stage4_fuse)

        d_2x_stage4_fuse1 = self.d_2x_fuse1_stage4(d_1x_stage3)
        d_2x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=2,
                                         mode='bilinear', align_corners=True)
        d_2x_stage4_fuse2 = self.d_2x_fuse2_stage4(d_2x_stage4_fuse2)
        d_2x_stage4_fuse = d_2x_stage4_fuse1 + d_2x_stage4_fuse2 + d_2x_stage3
        d_2x_stage4 = self.d_2x_stage4(d_2x_stage4_fuse)

        d_4x_stage4_fuse1 = self.d_4x_fuse1_stage4(d_1x_stage3)
        d_4x_stage4_fuse2 = self.d_4x_fuse2_stage4(d_2x_stage3)
        d_4x_stage4_fuse = d_4x_stage3 + d_4x_stage4_fuse1 + d_4x_stage4_fuse2
        d_4x_stage4 = self.d_4x_stage4(d_4x_stage4_fuse)

        # stage 5 ***********************************************************
        d_1x_stage5_fuse1 = F.interpolate(d_2x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse1 = self.d_1x_fuse1_stage5(d_1x_stage5_fuse1)
        d_1x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse2 = self.d_1x_fuse2_stage5(d_1x_stage5_fuse2)
        d_1x_stage5_fuse = d_1x_stage4 + d_1x_stage5_fuse1 + d_1x_stage5_fuse2
        d_1x_stage5 = self.d_1x_stage5(d_1x_stage5_fuse)

        d_2x_stage5_fuse1 = self.d_2x_fuse1_stage5(d_1x_stage4)
        d_2x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_2x_stage5_fuse2 = self.d_2x_fuse2_stage5(d_2x_stage5_fuse2)
        d_2x_stage5_fuse = d_2x_stage5_fuse1 + d_2x_stage5_fuse2 + d_2x_stage4
        d_2x_stage5 = self.d_2x_stage5(d_2x_stage5_fuse)

        d_4x_stage5_fuse1 = self.d_4x_fuse1_stage5(d_1x_stage4)
        d_4x_stage5_fuse2 = self.d_4x_fuse2_stage5(d_2x_stage4)
        d_4x_stage5_fuse = d_4x_stage4 + d_4x_stage5_fuse1 + d_4x_stage5_fuse2
        d_4x_stage5 = self.d_4x_stage5(d_4x_stage5_fuse)

        # last conv
        h, w = d_1x_stage5.size(2), d_1x_stage5.size(3)
        x_2x = F.interpolate(d_2x_stage5, size=[h, w], mode='bilinear')
        x_4x = F.interpolate(d_4x_stage5, size=[h, w], mode='bilinear')
        x = torch.cat([d_1x_stage5, x_2x, x_4x], dim= 1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=[self.input_size[0], self.input_size[1]], mode='bilinear')
        return torch.sigmoid(x)


    def _make_layer(self, block, inplane, block_num):
        layers = []
        for i in range(0, block_num):
            layers.append(block(inplane))
        return nn.Sequential(*layers)

class CMNet_TwoPath(nn.Module):
    def __init__(self, IN_CHANNEL, NUM_CLASS, INPUT_SIZE, layers):
        super(CMNet_TwoPath, self).__init__()
        self.input_size = INPUT_SIZE

        # stem net
        self.d_conv1 = nn.Conv2d(IN_CHANNEL, 64, kernel_size=3, stride=2, padding=1,
                                 bias=False)
        self.d_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.d_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                 bias=False)
        self.d_bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.d_relu = nn.ReLU(inplace=True)

        self.f_conv1 = nn.Conv2d(IN_CHANNEL, 64, kernel_size=3, stride=2, padding=1,
                                 bias=False)
        self.f_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.f_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                 bias=False)
        self.f_bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.f_relu = nn.ReLU(inplace=True)

        # stage 1
        self.d_1x_stage1 = self._make_layer(Bottleneck, 64, layers[0])  # out:64
        self.f_1x_stage1 = self._make_layer(Bottleneck, 64, layers[0])  # out:64

        # stage 2
        self.d_1x_stage2 = self._make_layer(Bottleneck, 64, layers[1])  # out:64
        self.d_2x_fuse1_stage2 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.d_2x_fuse2_stage2 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.d_2x_stage2 = self._make_layer(Bottleneck, 128, layers[1])  # out 128

        self.f_1x_stage2 = self._make_layer(Bottleneck, 64, layers[1])  # out:64
        self.f_2x_fuse1_stage2 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.f_2x_fuse2_stage2 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.f_2x_stage2 = self._make_layer(Bottleneck, 128, layers[1])  # out 128

        # stage 3
        self.d_1x_fuse1_stage3 = conv3x3(128, 64)
        self.d_1x_fuse2_stage3 = conv3x3(128, 64)
        self.d_1x_stage3 = self._make_layer(Bottleneck, 64, layers[2])
        self.d_2x_fuse1_stage3 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.d_2x_fuse2_stage3 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.d_2x_stage3 = self._make_layer(Bottleneck, 128, layers[2])
        self.d_4x_fuse1_stage3 = conv3x3(64, 256, stride=4)
        self.d_4x_fuse2_stage3 = conv3x3(128, 256, stride=2)
        self.d_4x_fuse3_stage3 = conv3x3(64, 256, stride=4)
        self.d_4x_fuse4_stage3 = conv3x3(128, 256, stride=2)
        self.d_4x_stage3 = self._make_layer(Bottleneck, 256, layers[2])

        self.f_1x_fuse1_stage3 = conv3x3(128, 64)
        self.f_1x_fuse2_stage3 = conv3x3(128, 64)
        self.f_1x_stage3 = self._make_layer(Bottleneck, 64, layers[2])
        self.f_2x_fuse1_stage3 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.f_2x_fuse2_stage3 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.f_2x_stage3 = self._make_layer(Bottleneck, 128, layers[2])
        self.f_4x_fuse1_stage3 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse2_stage3 = conv3x3(128, 256, stride=2)
        self.f_4x_fuse3_stage3 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse4_stage3 = conv3x3(128, 256, stride=2)
        self.f_4x_stage3 = self._make_layer(Bottleneck, 256, layers[2])

        # stage 4
        self.d_1x_fuse1_stage4 = conv3x3(128, 64)
        self.d_1x_fuse2_stage4 = conv3x3(256, 64)
        self.d_1x_fuse3_stage4 = conv3x3(128, 64)
        self.d_1x_fuse4_stage4 = conv3x3(256, 64)
        self.d_1x_stage4 = self._make_layer(Bottleneck, 64, layers[3])
        self.d_2x_fuse1_stage4 = conv3x3(64, 128, stride= 2)
        self.d_2x_fuse2_stage4 = conv3x3(256, 128)
        self.d_2x_fuse3_stage4 = conv3x3(64, 128, stride=2)
        self.d_2x_fuse4_stage4 = conv3x3(256, 128)
        self.d_2x_stage4 = self._make_layer(Bottleneck, 128, layers[3])
        self.d_4x_fuse1_stage4 = conv3x3(64, 256, stride= 4)
        self.d_4x_fuse2_stage4= conv3x3(128, 256, stride= 2)
        self.d_4x_fuse3_stage4 = conv3x3(64, 256, stride= 4)
        self.d_4x_fuse4_stage4 = conv3x3(128, 256, stride= 2)
        self.d_4x_stage4 = self._make_layer(Bottleneck, 256, layers[3])

        self.f_1x_fuse1_stage4 = conv3x3(128, 64)
        self.f_1x_fuse2_stage4 = conv3x3(256, 64)
        self.f_1x_fuse3_stage4 = conv3x3(128, 64)
        self.f_1x_fuse4_stage4 = conv3x3(256, 64)
        self.f_1x_stage4 = self._make_layer(Bottleneck, 64, layers[3])
        self.f_2x_fuse1_stage4 = conv3x3(64, 128, stride=2)
        self.f_2x_fuse2_stage4 = conv3x3(256, 128)
        self.f_2x_fuse3_stage4 = conv3x3(64, 128, stride=2)
        self.f_2x_fuse4_stage4 = conv3x3(256, 128)
        self.f_2x_stage4 = self._make_layer(Bottleneck, 128, layers[3])
        self.f_4x_fuse1_stage4 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse2_stage4 = conv3x3(128, 256, stride=2)
        self.f_4x_fuse3_stage4 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse4_stage4 = conv3x3(128, 256, stride=2)
        self.f_4x_stage4 = self._make_layer(Bottleneck, 256, layers[3])

        # stage 5
        self.d_1x_fuse1_stage5 = conv3x3(128, 64)
        self.d_1x_fuse2_stage5 = conv3x3(256, 64)
        self.d_1x_fuse3_stage5 = conv3x3(128, 64)
        self.d_1x_fuse4_stage5 = conv3x3(256, 64)
        self.d_1x_stage5 = self._make_layer(BasicBlock, 64, layers[4])
        self.d_2x_fuse1_stage5 = conv3x3(64, 128, stride=2)
        self.d_2x_fuse2_stage5 = conv3x3(256, 128)
        self.d_2x_fuse3_stage5 = conv3x3(64, 128, stride=2)
        self.d_2x_fuse4_stage5 = conv3x3(256, 128)
        self.d_2x_stage5 = self._make_layer(BasicBlock, 128, layers[4])
        self.d_4x_fuse1_stage5 = conv3x3(64, 256, stride=4)
        self.d_4x_fuse2_stage5 = conv3x3(128, 256, stride=2)
        self.d_4x_fuse3_stage5 = conv3x3(64, 256, stride=4)
        self.d_4x_fuse4_stage5 = conv3x3(128, 256, stride=2)
        self.d_4x_stage5 = self._make_layer(BasicBlock, 256, layers[4])

        self.f_1x_fuse1_stage5 = conv3x3(128, 64)
        self.f_1x_fuse2_stage5 = conv3x3(256, 64)
        self.f_1x_fuse3_stage5 = conv3x3(128, 64)
        self.f_1x_fuse4_stage5 = conv3x3(256, 64)
        self.f_1x_stage5 = self._make_layer(BasicBlock, 64, layers[4])
        self.f_2x_fuse1_stage5 = conv3x3(64, 128, stride=2)
        self.f_2x_fuse2_stage5 = conv3x3(256, 128)
        self.f_2x_fuse3_stage5 = conv3x3(64, 128, stride=2)
        self.f_2x_fuse4_stage5 = conv3x3(256, 128)
        self.f_2x_stage5 = self._make_layer(BasicBlock, 128, layers[4])
        self.f_4x_fuse1_stage5 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse2_stage5 = conv3x3(128, 256, stride=2)
        self.f_4x_fuse3_stage5 = conv3x3(64, 256, stride=4)
        self.f_4x_fuse4_stage5 = conv3x3(128, 256, stride=2)
        self.f_4x_stage5 = self._make_layer(BasicBlock, 256, layers[4])

        # last conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels= 896, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=NUM_CLASS, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x, y):
        # stem net***********************************************
        d_stem = self.d_conv1(x)
        d_stem = self.d_bn1(d_stem)
        d_stem = self.d_relu(d_stem)
        d_stem = self.d_conv2(d_stem)
        d_stem = self.d_bn2(d_stem)
        d_stem = self.d_relu(d_stem)
        f_stem = self.f_conv1(y)
        f_stem = self.f_bn1(f_stem)
        f_stem = self.f_relu(f_stem)
        f_stem = self.f_conv2(f_stem)
        f_stem = self.f_bn2(f_stem)
        f_stem = self.f_relu(f_stem)

        # stage 1 *********************************************
        d_1x_stage1 = self.d_1x_stage1(d_stem)
        f_1x_stage1 = self.f_1x_stage1(f_stem)

        # stage 2 *********************************************
        d_1x_stage2_fuse = d_1x_stage1 + f_1x_stage1
        d_1x_stage2 = self.d_1x_stage2(d_1x_stage2_fuse)
        d_2x_stage2_fuse1 = self.d_2x_fuse1_stage2(d_1x_stage1)
        d_2x_stage2_fuse2 = self.d_2x_fuse2_stage2(f_1x_stage1)
        d_2x_stage2_fuse = d_2x_stage2_fuse1 + d_2x_stage2_fuse2
        d_2x_stage2 = self.d_2x_stage2(d_2x_stage2_fuse)
        f_1x_stage2_fuse = d_1x_stage1 + f_1x_stage1
        f_1x_stage2 = self.f_1x_stage2(f_1x_stage2_fuse)
        f_2x_stage2_fuse1 = self.f_2x_fuse1_stage2(d_1x_stage1)
        f_2x_stage2_fuse2 = self.f_2x_fuse2_stage2(f_1x_stage1)
        f_2x_stage2_fuse = f_2x_stage2_fuse1 + f_2x_stage2_fuse2
        f_2x_stage2 = self.f_2x_stage2(f_2x_stage2_fuse)

        # stage 3***************************************************
        d_1x_stage3_fuse1 = F.interpolate(d_2x_stage2, scale_factor=2,
                                         mode='bilinear', align_corners= True)
        d_1x_stage3_fuse1 = self.d_1x_fuse1_stage3(d_1x_stage3_fuse1)
        d_1x_stage3_fuse2 = F.interpolate(f_2x_stage2, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage3_fuse2 = self.d_1x_fuse2_stage3(d_1x_stage3_fuse2)
        d_1x_stage3_fuse = d_1x_stage3_fuse1 + d_1x_stage3_fuse2 + d_1x_stage2 + f_1x_stage2
        d_1x_stage3 = self.d_1x_stage3(d_1x_stage3_fuse)
        d_2x_stage3_fuse1 = self.d_2x_fuse1_stage3(d_1x_stage2)
        d_2x_stage3_fuse2 = self.d_2x_fuse2_stage3(f_1x_stage2)
        d_2x_stage3_fuse = d_2x_stage3_fuse1 + d_2x_stage3_fuse2 + d_2x_stage2 + f_2x_stage2
        d_2x_stage3 = self.d_2x_stage3(d_2x_stage3_fuse)
        d_4x_stage3_fuse1 = self.d_4x_fuse1_stage3(d_1x_stage2)
        d_4x_stage3_fuse2 = self.d_4x_fuse2_stage3(d_2x_stage2)
        d_4x_stage3_fuse3 = self.d_4x_fuse3_stage3(f_1x_stage2)
        d_4x_stage3_fuse4 = self.d_4x_fuse4_stage3(f_2x_stage2)
        d_4x_stage3_fuse = d_4x_stage3_fuse1 + d_4x_stage3_fuse2 + d_4x_stage3_fuse3 + d_4x_stage3_fuse4
        d_4x_stage3 = self.d_4x_stage3(d_4x_stage3_fuse)

        f_1x_stage3_fuse1 = F.interpolate(d_2x_stage2, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage3_fuse1 = self.f_1x_fuse1_stage3(f_1x_stage3_fuse1)
        f_1x_stage3_fuse2 = F.interpolate(f_2x_stage2, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage3_fuse2 = self.f_1x_fuse2_stage3(f_1x_stage3_fuse2)
        f_1x_stage3_fuse = f_1x_stage3_fuse1 + f_1x_stage3_fuse2 + d_1x_stage2 + f_1x_stage2
        f_1x_stage3 = self.f_1x_stage3(f_1x_stage3_fuse)
        f_2x_stage3_fuse1 = self.f_2x_fuse1_stage3(d_1x_stage2)
        f_2x_stage3_fuse2 = self.f_2x_fuse2_stage3(f_1x_stage2)
        f_2x_stage3_fuse = f_2x_stage3_fuse1 + f_2x_stage3_fuse2 + d_2x_stage2 + f_2x_stage2
        f_2x_stage3 = self.f_2x_stage3(f_2x_stage3_fuse)
        f_4x_stage3_fuse1 = self.f_4x_fuse1_stage3(d_1x_stage2)
        f_4x_stage3_fuse2 = self.f_4x_fuse2_stage3(d_2x_stage2)
        f_4x_stage3_fuse3 = self.f_4x_fuse3_stage3(f_1x_stage2)
        f_4x_stage3_fuse4 = self.f_4x_fuse4_stage3(f_2x_stage2)
        f_4x_stage3_fuse = f_4x_stage3_fuse1 + f_4x_stage3_fuse2 + f_4x_stage3_fuse3 + f_4x_stage3_fuse4
        f_4x_stage3 = self.f_4x_stage3(f_4x_stage3_fuse)

        # stage 4 ***************************************************************************************
        d_1x_stage4_fuse1 = F.interpolate(d_2x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage4_fuse1 = self.d_1x_fuse1_stage4(d_1x_stage4_fuse1)
        d_1x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage4_fuse2 = self.d_1x_fuse2_stage4(d_1x_stage4_fuse2)
        d_1x_stage4_fuse3 = F.interpolate(f_2x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage4_fuse3 = self.d_1x_fuse3_stage4(d_1x_stage4_fuse3)
        d_1x_stage4_fuse4 = F.interpolate(f_4x_stage3, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage4_fuse4 = self.d_1x_fuse4_stage4(d_1x_stage4_fuse4)
        d_1x_stage4_fuse = d_1x_stage4_fuse1 + d_1x_stage4_fuse2 + d_1x_stage4_fuse3 + d_1x_stage4_fuse4 + \
                           d_1x_stage3 + f_1x_stage3
        d_1x_stage4 = self.d_1x_stage4(d_1x_stage4_fuse)
        d_2x_stage4_fuse1 = self.d_2x_fuse1_stage4(d_1x_stage3)
        d_2x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_2x_stage4_fuse2 = self.d_2x_fuse2_stage4(d_2x_stage4_fuse2)
        d_2x_stage4_fuse3 = self.d_2x_fuse3_stage4(f_1x_stage3)
        d_2x_stage4_fuse4 = F.interpolate(f_4x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_2x_stage4_fuse4 = self.d_2x_fuse4_stage4(d_2x_stage4_fuse4)
        d_2x_stage4_fuse = d_2x_stage3 + f_2x_stage3 + d_2x_stage4_fuse1 + d_2x_stage4_fuse2 + \
                           d_2x_stage4_fuse3 + d_2x_stage4_fuse4
        d_2x_stage4 = self.d_2x_stage4(d_2x_stage4_fuse)
        d_4x_stage4_fuse1 = self.d_4x_fuse1_stage4(d_1x_stage3)
        d_4x_stage4_fuse2 = self.d_4x_fuse2_stage4(d_2x_stage3)
        d_4x_stage4_fuse3 = self.d_4x_fuse3_stage4(f_1x_stage3)
        d_4x_stage4_fuse4 = self.d_4x_fuse4_stage4(f_2x_stage3)
        d_4x_stage4_fuse = d_4x_stage3 + f_4x_stage3 + d_4x_stage4_fuse1 + d_4x_stage4_fuse2 + \
                           d_4x_stage4_fuse3 + d_4x_stage4_fuse4
        d_4x_stage4 = self.d_4x_stage4(d_4x_stage4_fuse)

        f_1x_stage4_fuse1 = F.interpolate(d_2x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage4_fuse1 = self.f_1x_fuse1_stage4(f_1x_stage4_fuse1)
        f_1x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        f_1x_stage4_fuse2 = self.f_1x_fuse2_stage4(f_1x_stage4_fuse2)
        f_1x_stage4_fuse3 = F.interpolate(f_2x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage4_fuse3 = self.f_1x_fuse3_stage4(f_1x_stage4_fuse3)
        f_1x_stage4_fuse4 = F.interpolate(f_4x_stage3, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        f_1x_stage4_fuse4 = self.f_1x_fuse4_stage4(f_1x_stage4_fuse4)
        f_1x_stage4_fuse = f_1x_stage4_fuse1 + f_1x_stage4_fuse2 + f_1x_stage4_fuse3 + f_1x_stage4_fuse4 + \
                           d_1x_stage3 + f_1x_stage3
        f_1x_stage4 = self.f_1x_stage4(f_1x_stage4_fuse)

        f_2x_stage4_fuse1 = self.f_2x_fuse1_stage4(d_1x_stage3)
        f_2x_stage4_fuse2 = F.interpolate(d_4x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_2x_stage4_fuse2 = self.f_2x_fuse2_stage4(f_2x_stage4_fuse2)
        f_2x_stage4_fuse3 = self.f_2x_fuse3_stage4(f_1x_stage3)
        f_2x_stage4_fuse4 = F.interpolate(f_4x_stage3, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_2x_stage4_fuse4 = self.f_2x_fuse4_stage4(f_2x_stage4_fuse4)
        f_2x_stage4_fuse = d_2x_stage3 + f_2x_stage3 + f_2x_stage4_fuse1 + f_2x_stage4_fuse2 + \
                           f_2x_stage4_fuse3 + f_2x_stage4_fuse4
        f_2x_stage4 = self.f_2x_stage4(f_2x_stage4_fuse)

        f_4x_stage4_fuse1 = self.f_4x_fuse1_stage4(d_1x_stage3)
        f_4x_stage4_fuse2 = self.f_4x_fuse2_stage4(d_2x_stage3)
        f_4x_stage4_fuse3 = self.f_4x_fuse3_stage4(f_1x_stage3)
        f_4x_stage4_fuse4 = self.f_4x_fuse4_stage4(f_2x_stage3)
        f_4x_stage4_fuse = d_4x_stage3 + f_4x_stage3 + f_4x_stage4_fuse1 + f_4x_stage4_fuse2 +\
                           f_4x_stage4_fuse3 + f_4x_stage4_fuse4
        f_4x_stage4 = self.d_4x_stage4(f_4x_stage4_fuse)

        # stage 5 ***************************************************************************************
        d_1x_stage5_fuse1 = F.interpolate(d_2x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse1 = self.d_1x_fuse1_stage5(d_1x_stage5_fuse1)
        d_1x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse2 = self.d_1x_fuse2_stage5(d_1x_stage5_fuse2)
        d_1x_stage5_fuse3 = F.interpolate(f_2x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse3 = self.d_1x_fuse3_stage5(d_1x_stage5_fuse3)
        d_1x_stage5_fuse4 = F.interpolate(f_4x_stage4, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        d_1x_stage5_fuse4 = self.d_1x_fuse4_stage5(d_1x_stage5_fuse4)
        d_1x_stage5_fuse = d_1x_stage5_fuse1 + d_1x_stage5_fuse2 + d_1x_stage5_fuse3 + d_1x_stage5_fuse4 + \
                           d_1x_stage4 + f_1x_stage4
        d_1x_stage5 = self.d_1x_stage5(d_1x_stage5_fuse)
        d_2x_stage5_fuse1 = self.d_2x_fuse1_stage5(d_1x_stage4)
        d_2x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_2x_stage5_fuse2 = self.d_2x_fuse2_stage5(d_2x_stage5_fuse2)
        d_2x_stage5_fuse3 = self.d_2x_fuse3_stage5(f_1x_stage4)
        d_2x_stage5_fuse4 = F.interpolate(f_4x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        d_2x_stage5_fuse4 = self.d_2x_fuse4_stage5(d_2x_stage5_fuse4)
        d_2x_stage5_fuse = d_2x_stage4 + f_2x_stage4 + d_2x_stage5_fuse1 + d_2x_stage5_fuse2 + \
                           d_2x_stage5_fuse3 + d_2x_stage5_fuse4
        d_2x_stage5 = self.d_2x_stage5(d_2x_stage5_fuse)
        d_4x_stage5_fuse1 = self.d_4x_fuse1_stage5(d_1x_stage4)
        d_4x_stage5_fuse2 = self.d_4x_fuse2_stage5(d_2x_stage4)
        d_4x_stage5_fuse3 = self.d_4x_fuse3_stage5(f_1x_stage4)
        d_4x_stage5_fuse4 = self.d_4x_fuse4_stage5(f_2x_stage4)
        d_4x_stage5_fuse = d_4x_stage4 + f_4x_stage4 + d_4x_stage5_fuse1 + d_4x_stage5_fuse2 + \
                           d_4x_stage5_fuse3 + d_4x_stage5_fuse4
        d_4x_stage5 = self.d_4x_stage5(d_4x_stage5_fuse)

        f_1x_stage5_fuse1 = F.interpolate(d_2x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage5_fuse1 = self.f_1x_fuse1_stage5(f_1x_stage5_fuse1)
        f_1x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        f_1x_stage5_fuse2 = self.f_1x_fuse2_stage5(f_1x_stage5_fuse2)
        f_1x_stage5_fuse3 = F.interpolate(f_2x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_1x_stage5_fuse3 = self.f_1x_fuse3_stage5(f_1x_stage5_fuse3)
        f_1x_stage5_fuse4 = F.interpolate(f_4x_stage4, scale_factor=4,
                                          mode='bilinear', align_corners=True)
        f_1x_stage5_fuse4 = self.f_1x_fuse4_stage5(f_1x_stage5_fuse4)
        f_1x_stage5_fuse = f_1x_stage5_fuse1 + f_1x_stage5_fuse2 + f_1x_stage5_fuse3 + f_1x_stage5_fuse4 + \
                           d_1x_stage4 + f_1x_stage4
        f_1x_stage5 = self.f_1x_stage5(f_1x_stage5_fuse)

        f_2x_stage5_fuse1 = self.f_2x_fuse1_stage5(d_1x_stage4)
        f_2x_stage5_fuse2 = F.interpolate(d_4x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_2x_stage5_fuse2 = self.f_2x_fuse2_stage5(f_2x_stage5_fuse2)
        f_2x_stage5_fuse3 = self.f_2x_fuse3_stage5(f_1x_stage4)
        f_2x_stage5_fuse4 = F.interpolate(f_4x_stage4, scale_factor=2,
                                          mode='bilinear', align_corners=True)
        f_2x_stage5_fuse4 = self.f_2x_fuse4_stage5(f_2x_stage5_fuse4)
        f_2x_stage5_fuse = d_2x_stage4 + f_2x_stage4 + f_2x_stage5_fuse1 + f_2x_stage5_fuse2 + \
                           f_2x_stage5_fuse3 + f_2x_stage5_fuse4
        f_2x_stage5 = self.f_2x_stage5(f_2x_stage5_fuse)

        f_4x_stage5_fuse1 = self.f_4x_fuse1_stage5(d_1x_stage4)
        f_4x_stage5_fuse2 = self.f_4x_fuse2_stage5(d_2x_stage4)
        f_4x_stage5_fuse3 = self.f_4x_fuse3_stage5(f_1x_stage4)
        f_4x_stage5_fuse4 = self.f_4x_fuse4_stage5(f_2x_stage4)
        f_4x_stage5_fuse = d_4x_stage4 + f_4x_stage4 + f_4x_stage5_fuse1 + f_4x_stage5_fuse2 + \
                           f_4x_stage5_fuse3 + f_4x_stage5_fuse4
        f_4x_stage5 = self.f_4x_stage5(f_4x_stage5_fuse)

        # last conv
        h, w = d_1x_stage5.size(2), d_1x_stage5.size(3)
        x_2x = F.interpolate(d_2x_stage5, size=[h, w], mode='bilinear')
        x_4x = F.interpolate(d_4x_stage5, size=[h, w], mode='bilinear')
        y_2x = F.interpolate(f_2x_stage5, size=[h, w], mode='bilinear')
        y_4x = F.interpolate(f_4x_stage5, size=[h, w], mode='bilinear')
        x = torch.cat([d_1x_stage5, x_2x, x_4x, f_1x_stage5, y_2x, y_4x], dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=[self.input_size[0], self.input_size[1]], mode='bilinear')
        return torch.sigmoid(x)

    def _make_layer(self, block, inplane, block_num):
        layers = []
        for i in range(0, block_num):
            layers.append(block(inplane))
        return nn.Sequential(*layers)

if __name__ == '__main__':
    input = torch.randn([32, 1, 224, 224]).cuda()
    model = CMNet_OnePath(IN_CHANNEL = 1,  NUM_CLASS= 2, layers= [3, 3, 3, 3, 1], INPUT_SIZE= [224, 224])
    model.cuda()
    res = model(input)
    print(model)