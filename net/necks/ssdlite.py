# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.16 23:48
@file    : ssdlite.py
@desc    : 深度可分离卷积构成的SSD
"""
from torch import nn
from net.backbones.mobilenet_v2 import ConvBNReLU


class SSDLiteResidual(nn.Module):
    def __init__(self, inp, oup, stride=2, norm_layer=None):
        super(SSDLiteResidual, self).__init__()
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(oup // 2)

        layers = []
        layers.extend([
            # pw
            ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer),
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
            nn.ReLU6(inplace=False)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
            return self.conv(x)


class SSDLite(nn.Module):
    def __init__(self):
        super(SSDLite, self).__init__()
        input_channel = 1280
        channels = [512, 256, 256, 128]
        layers = []
        for i, c in enumerate(channels):
            if i == 0:
                i_channel = 1280
            else:
                i_channel = channels[i-1]
            layers.append(SSDLiteResidual(i_channel, c))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        return x


def ssd_body():
    ssd_layers = nn.ModuleList([
        SSDLiteResidual(1280, 512),
        SSDLiteResidual(512,  256),
        SSDLiteResidual(256,  256),
        SSDLiteResidual(256,  128)
    ])
    return ssd_layers