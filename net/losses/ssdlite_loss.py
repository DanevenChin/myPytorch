# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.20 22:29
@file    : ssdlite_loss.py
@desc    : ssdlite 损失函数
"""

import torch


def GIOU_loss():
    pass


def DIOU_loss():
    pass


def CIOU_loss():
    pass


def classification_loss():
    """
    分类loss的计算：1）先计算所有的分类loss；
                    2）去掉loss中的正样本；
                    3）将loss按大到小排序；
                    4）选择负样本的数量，默认为正样本的3倍；
                    5）选择前3倍正样本数量的负样本以及正样本作为分类loss的计算
    :return:
    """
    pass


def localization_loss():
    """
    定位loss的计算：匹配后的样本，定义置信度大于0的样本为正样本，并计算正样本的定位loss
    :return:
    """
    pass