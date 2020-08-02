# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.24 00:28
@file    : ssd_match.py
@desc    : ssd的匹配策略
"""
import torch

"""
#####################　匹配机制　#####################
(1)对于每个ground truth box，首先将它匹配给和它有最大的IoU(jaccard overlap)的anchor。
这样可以保证对于每一个ground truth box都有一个anchor来对应。这些和ground truth对应的
anchor为正样本，然后没有匹配到的anchor为负样本，由于一张图中ground truth较少，而最后得
到的anchor数量又很多，所以这种方法就会导致正负样本的极不平衡，所以需要第二种匹配策略来缓解。
(2)第二种匹配策略就是对于剩余的anchor，如果它和某个ground truth box的IoU大于某个阈值(0.5)，
那么将这个ground truth box匹配给这个anchor，如果某个anchor和多个ground truth box的IoU大于
阈值，那么选择IoU最大的ground truth进行匹配。这样一个ground truth就对应多个anchor，但是每
个anchor只能对应一个ground truth。
"""


def point_form(boxes):
    # 将(cx, cy, w, h) 形式的box坐标转换成 (xmin, ymin, xmax, ymax) 形式
    return torch.cat(((boxes[:, :2] - boxes[:, 2:]/2),  # xmin, ymin
                    (boxes[:, :2] + boxes[:, 2:]/2)), 1)  # xmax, ymax


def intersect(box_a, box_b):
    """
    计算两个框的交集，两个框可以维度不一致
    思路: 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    :param box_a: tensor，(truths), shape:[num_objs, 4]
    :param box_b: tensor，(priors), shape:[num_priors, 4]
    :return: shape为[num_obj, num_priors]，box_a 与 box_b的交集, 其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.
    """
    # 两个box的交集可以看成是一个新的box, 该box的左上角坐标是box_a和box_b左上角坐标的较大值, 右下角坐标是box_a和box_b的右下角坐标的较小值
    A = box_a.size(0)
    B = box_b.size(0)

    # box_a 左上角/右下角坐标 expand以后, 维度会变成(A,B,2), 其中, 具体可看 expand 的相关原理.
    # box_b也是同理, 这样做是为了得到a中某个box与b中某个box的左上角(min_xy)的较大者(max).
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A,B,2),
                       box_b[:, :2].unsqueeze(0).expand(A,B,2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A,B,2),
                       box_b[:, 2:].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy-min_xy), min=0)  # 右下角减去左上角, 如果为负值, 说明没有交集, 置为0
    return inter[:, :, 0] * inter[:, :, 1]  # 高×宽, 返回交集的面积, shape 刚好为 [A, B]


def jaccard(box_a, box_b):
    """
    计算两个框直接的iou，支持两个框的维度不一致
    公式：A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    :param box_a: tensor，(truths), shape:[num_objs, 4]
    :param box_b: tensor，(priors), shape:[num_priors, 4]
    :return: 输出交并比，tensor, shape为[num_objs, num_priors]
    """
    inter = intersect(box_a, box_b) # 求任意两个box的交集面积, shape为[A, B], 即[num_obj, num_priors]
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def encode(matched, priors, variances):
    """
    边框坐标进行编码
    原文公式如下：
        tx = (cx - cxa) / wa * x_scale
        ty = (cy - cya) / ha * y_scale
        tw = log(w / wa) * w_scale
        th = log(h / ha) * h_scale
        其中，[cx, cy, w, h]为归一化的真实坐标，
              [cxa, cya, wa, ha]为归一化的先验框坐标，
              [tx, ty, tw, th]为编码后的坐标
        注意：只编码与 gt 匹配度最高的先验框，这一步详见match函数
    :param matched: tensor,[num_priors, 4], 存储的是与priorbox匹配的gtbox的坐标. 形式为(xmin, ymin, xmax, ymax)
    :param priors: tensor, [num_priors, 4], 存储的是priorbox的坐标. 形式为(cx, cy, w, h)
    :param variances: 默认[0.1, 0.2]
    :return: 编码后的匹配框: tensor, [num_priors, 4], [cx, cy, w, h]
    """
    g_cxy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]  # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标, 获得中心坐标的偏移量
    g_cxy /= (variances[0]*priors[:, 2:])  # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 令互相匹配的gtbox的宽高除以priorbox的宽高.
    g_wh = torch.log(g_wh) / variances[1]  # 这里这个variances[1]=0.2 不太懂是为什么.
    return torch.cat([g_cxy, g_wh], 1)  # 将编码后的中心坐标和宽高``连接起来, 返回 [num_priors, 4]


def decode(loc, priors, variances):
    """
    网络预测边框解码
    :param loc   : tensor, 网络预测的边框，Shape为[num_priors, 4]，边框格式为归一化的[cx, cy, w, h]
    :param priors: tensor, 先验框的边框，Shape为[num_priors, 4]，边框格式为归一化的[cx, cy, w, h]
    :param variances: 默认为[0, 1]
    :return:
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    真实框与先验框的匹配
    :param threshold: 真实框与先验框的iou阈值，大于该阈值代表匹配
    :param truths: tensor, 真实框bbox，shape为[num_objs, 4], 其中坐标格式为归一化的[xmin, ymin, xmax, ymax]
    :param priors: tensor, 先验框bbox，shape为[num_priors, 4]，其中坐标格式为归一化的[cx, cy, w, h]
    :param variances: 编码时的中点，宽高尺度，默认为[0.1, 0.2]
    :param labels: tensor, 真实类别cls，shape为[num_objs, 1]
    :param loc_t: tensor，用来保存匹配后的bbox信息，shape为[batch_size, num_priors, 4]，边框格式为[cx, cy, w, h]
    :param conf_t: tensor，用来保存匹配后的cls信息，shape为[batch_size, num_priors]
    :param idx: batch_size索引
    :return:
    """
    # 名校（共有num_priors所名校，先验框）提前批考试，挑选优等生（num_objs个优等生，gt）
    overlaps = jaccard(truths, point_form(priors))  # [num_objs, num_priors]

    # 优等生与名校的统计分析
    # 与优等生匹配度最好的名校
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # keepdim=True, 因此shape为[num_objs,1]
    # 与名校匹配度最好的学生（可能没有优等生）
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # shape为[1, num_priors]
    best_prior_idx.squeeze_(1)  # [num_objs, 1] -> [num_objs]
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)  # [1, num_priors] -> [num_priors]
    best_truth_overlap.squeeze_(0)

    # 提前批：面向对象：优等生（与 gt 匹配度最好的先验框）
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 将0维的与gt匹配最高的先验框的位置iou置为2（因为iou不可能大于2），
                                                          # best_prior_idx保存的就是先验框的位置，因为先让匹配度最好的占个坑位，
                                                          # 同时防止优等生漏选了，然后再选择次要的。多么像这个真实的世界啊！

    # 给优等生（gt）改成绩，可能会有多个名校(先验配)看中同一个优等生（gt），这里操作后，
    # 可以保证每一个优等生（每个gt都匹配了一个先验框） 都有名校可读，因为一个名校只有一个名额，总共有num_priors个名校。
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # 先验框匹配坑位，针对bbox
    conf = labels[best_truth_idx] + 1  # 先验框匹配坑位，针对cls
    conf[best_truth_overlap < threshold] = 0  # 将与gtbox的交并比小于阈值的置为0 , 淘汰掉太差的学生，因此每所名校不一定都能找到普通学生或优等生，但优等生一定有名校读
    loc = encode(matches, priors, variances)  # 返回编码后的中心坐标和宽高
    loc_t[idx] = loc  # 设置第idx张图片的gt编码坐标信息
    conf_t[idx] = conf  # 设置第idx张图片的编号信息.


if __name__ == '__main__':
    threshold = 0.5
    truths = torch.rand(5,4)
    priors = torch.rand(10,4)
    variances = [0.1,0.2]
    labels = torch.rand(5, 1)
    loc_t = torch.Tensor(2, 10, 4)
    conf_t = torch.LongTensor(2, 10)
    idx = 0
    loc_t, conf_t = match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx)