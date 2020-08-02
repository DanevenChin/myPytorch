# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.30 22:14
@file    : ssd_loss.py
@desc    : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.anchor.ssd_match import match

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, negpos_ratio, use_gpu=False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu =  use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = negpos_ratio
        self.variance = [0.1, 0.2]

    def forward(self, pred, targets):
        '''
        Args:
            pred: A tuple, 包含 loc(编码钱的位置信息), conf(类别), priors(先验框);
                  loc_pred_data: shape[b,M,4];
                  conf_pred_data: shape[b,M,num_classes];
                  priors: shape[M,4];

            targets: 真实的boxes和labels,shape[b,num_objs,5];
        '''
        loc_pred_data, conf_pred_data, priors = pred
        batch = loc_pred_data.size(0)  # batch
        num_priors = priors[:loc_pred_data.size(1), :].size(0)  # 先验框个数

        # 获取匹配每个prior box的 ground truth
        # 创建 loc_t 和 conf_t 保存真实box的位置和类别
        loc_t = torch.Tensor(batch, num_priors, 4)
        conf_t = torch.LongTensor(batch, num_priors)

        for idx in range(batch):
            truths = targets[idx][:, :-1].detach()  # ground truth box信息
            labels = targets[idx][:, -1].detach()  # ground truth conf信息
            defaults = priors.detach()     # priors的 box 信息

            # 匹配 ground truth
            match(self.threshold, truths, defaults,
                  self.variance, labels, loc_t, conf_t, idx)

        # use gpu
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0 # 匹配中所有的正样本mask,shape[b,M]

        # Localization Loss,使用 Smooth L1
        # shape[b,M]-->shape[b,M,4]
        pos_idx = pos.unsqueeze(2).expand_as(loc_pred_data)
        loc_p = loc_pred_data[pos_idx].view(-1,4)  # 预测的正样本box信息
        loc_t = loc_t[pos_idx].view(-1,4)     # 真实的正样本box信息
        loss_l = F.smooth_l1_loss(loc_p, loc_t) # Smooth L1 损失

        '''
        Target；
            下面进行hard negative mining
        过程:
            1、 针对所有batch的conf，按照置信度误差(预测背景的置信度越小，误差越大)进行降序排列;
            2、 负样本的label全是背景，那么利用log softmax 计算出logP,
               logP越大，则背景概率越低,误差越大;
            3、 选取误差交大的top_k作为负样本，保证正负样本比例接近1:3;
        '''
        # shape[b*M,num_classes]
        batch_conf = conf_pred_data.view(-1, self.num_classes)
        # 使用logsoftmax，计算置信度,shape[b*M, 1]
        conf_logP = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # hard Negative Mining
        conf_logP = conf_logP.view(batch, -1)  # shape[b, M]
        conf_logP[pos] = 0 # 把正样本排除，剩下的就全是负样本，可以进行抽样

        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, index = conf_logP.sort(1, descending=True)
        _, idx_rank = index.sort(1)

        # 抽取负样本
        # 每个batch中正样本的数目，shape[b,1]
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max= pos.size(1)-1)
        neg = idx_rank < num_neg # 抽取前top_k个负样本，shape[b, M]

        # shape[b,M] --> shape[b,M,num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred_data)

        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_pred_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        conf_target = conf_t[(pos+neg).gt(0)]

        # 计算conf交叉熵
        loss_c = F.cross_entropy(conf_p, conf_target)

        # 正样本个数
        N = num_pos.detach().sum().float()

        loss_l /= N
        loss_c /= N

        return loss_l, loss_c


# 调试代码使用
if __name__ == "__main__":
    batch_size = 1
    loss = MultiBoxLoss(2, 0.5, 3)
    from net.anchor.ssd_prior import get_prior_box
    prior = get_prior_box().data
    p_bbox = torch.randint(20, (batch_size,1917,4)).float()/20
    p_bbox = torch.cat((p_bbox[...,:2], p_bbox[...,2:]+p_bbox[...,:2]),2)
    p_conf = torch.randn(batch_size,1917,2)
    p = (p_bbox, p_conf)
    p1 = (p_bbox, p_conf,prior)
    t = torch.randint(20,(batch_size, 10, 4)).float()/20
    t = torch.cat((t[..., :2], t[..., 2:] + t[..., :2]), 2)
    tt = torch.randint(1, (batch_size,10,1))
    t = torch.cat((t,tt.float()), dim=2)

    print(p1[0].shape, p1[1].shape, t.shape)
    l, c = loss(p1, t)
    print('loc loss:', l)
    print('conf loss:', c)
