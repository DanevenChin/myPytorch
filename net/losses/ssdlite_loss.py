# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.20 22:29
@file    : ssdlite_loss.py
@desc    : ssdlite 损失函数
"""

import torch
from net.anchor.ssd_prior import get_prior_box
from net.anchor.ssd_match import match
from torch.autograd import Variable
import torch.nn.functional as F


def GIOU_loss():
    pass


def DIOU_loss():
    pass


def CIOU_loss():
    pass


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def classification_loss(conf_data, conf_t, pos, num_classes):
    """
    分类loss的计算：1）先计算所有的分类loss；
                    2）去掉loss中的正样本；
                    3）将loss按大到小排序；
                    4）选择负样本的数量，默认为正样本的3倍；
                    5）选择前3倍正样本数量的负样本以及正样本作为分类loss的计算
    :param pred: 预测值，
    :param tar: 真实值，
    :param pos_idx: 正样本对应的索引
    :return:
    """
    # 计算最大的置信度, 以进行难负样本挖掘
    # conf_data: [batch, num_priors, num_classes]
    # batch_conf: [batch*num_priors, num_classes]
    negpos_ratio = 3
    num_batch = conf_data.size(0)
    batch_conf = conf_data.view(-1, num_classes)  # 预测值

    # conf_t: [batch, num_priors]
    # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

    # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
    loss_c[pos.view(-1, 1)] = 0  # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
    # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
    loss_c = loss_c.view(num_batch, -1)  # reshape
    # 进行降序排序, 并获取到排序的下标
    _, loss_idx = loss_c.sort(1, descending=True)
    # 将下标进行升序排序, 并获取到下标的下标
    _, idx_rank = loss_idx.sort(1)
    # num_pos: [batch, 1], 统计每个样本中的obj个数
    num_pos = pos.long().sum(1, keepdim=True)
    # 根据obj的个数, 确定负样本的个数(正样本的3倍)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=pos.size(1) - 1)
    # 获取到负样本的下标
    neg = idx_rank < num_neg.expand_as(idx_rank)

    # 计算包括正样本和负样本的置信度损失
    # pos: [batch, num_priors]
    # pos_idx: [batch, num_priors, num_classes]
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    # neg: [batch, num_priors]
    # neg_idx: [batch, num_priors, num_classes]
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
    conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
    # 按照pos_idx和neg_idx筛选目标数据
    targets_weighted = conf_t[(pos + neg).gt(0)]
    # 计算二者的交叉熵
    cls_loss = F.cross_entropy(conf_p, targets_weighted, size_average=False)

    return cls_loss


def localization_loss(pred, tar):
    """
    定位loss的计算：匹配后的样本，定义置信度大于0的样本为正样本，并计算正样本的定位loss
    :param pred: 预测值，[bz*object_num, 4]
    :param tar: 真实值，[bz*object_num, 4]
    :return:
    """
    loc_loss = F.smooth_l1_loss(pred, tar, reduction="sum")

    return loc_loss


def compute_loss(pred, tar):
    num_classes = 1
    loc_data, conf_data = pred
    batch_size = loc_data.size[0]
    priors = get_prior_box().data
    priors_num = priors.size()[0]

    loc_t = torch.Tensor(batch_size, priors_num, 4)
    conf_t = torch.LongTensor(batch_size, priors_num)
    for bz in range(batch_size):
        gt_box = tar[bz][:, :-1].data
        gt_label = tar[bz][:, -1].data
        loc_t, conf_t = match(threshold=0.5,
                              truths=gt_box,
                              priors=priors,
                              variances=[0.1,0.2],
                              labels=gt_label,
                              loc_t=loc_t,
                              conf_t=conf_t,
                              idx=bz)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()

    # wrap targets
    loc_t = Variable(loc_t, requires_grad=False)
    conf_t = Variable(conf_t, requires_grad=False)

    pos = conf_t > 0  # 正样本数量, bool

    # 计算定位loss
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_t = loc_t[pos_idx].view(-1, 4)  # [bz, prior_num, 4] -> [bz*prior_num, 4]
    loc_p = loc_data[pos_idx.view(-1, 4)]  # [bz, prior_num, 4] -> [bz*prior_num, 4]

    loc_loss = localization_loss(loc_p, loc_t)

    # 计算分类loss
    cls_loss = classification_loss(conf_data, conf_t, pos, num_classes)

    # 总loss
    num_pos = pos.long().sum(1, keepdim=True)  # 正样本的数量
    N = num_pos.data.sum().float()
    loc_loss /= N
    cls_loss /= N
    total_loss = loc_loss + cls_loss

    return total_loss, loc_loss, cls_loss