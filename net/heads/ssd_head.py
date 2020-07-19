# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.19 16:17
@file    : ssd_head.py
@desc    : ssd输出分支
"""
from torch import nn


def SeperableConv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = in_channels,
                  kernel_size = kernel_size,
                  groups = in_channels,
                  stride = stride,
                  padding = padding,
                  bias = False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace = False),
        nn.Conv2d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = 1,
                  bias = True),
    )


def multibox(n_classes, width_mult=1.0, ):
    '''each output featureMap produce 6 result, >> mbox = 6 in each layer
    '''
    loc_layers = nn.ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult),
                        # out_channels = 6 * 4,
                        out_channels=3 * 4,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
        # nn.Conv2d(in_channels = 64, out_channels = 6 * 4, kernel_size = 1),
    ])

    conf_layers = nn.ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult),
                        # out_channels = 6 * n_classes,
                        out_channels=3 * n_classes,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * n_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * n_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * n_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * n_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * n_classes, kernel_size=3, padding=1),
        # nn.Conv2d(in_channels = 64, out_channels = 6 * n_classes, kernel_size = 1),
    ])
    return loc_layers, conf_layers