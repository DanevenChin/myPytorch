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
    return torch.cat( (boxes[:2] - boxes[2:]/2), # xmin, ymin
                    (boxes[:2] + boxes[2:]/2), 1) # xmax, ymax


def intersect(box_a, box_b):
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]) box_a 与 box_b 两个集合中任意两个 box 的交集, 其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.(非对称矩阵)
    # 思路: 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    # 两个box的交集可以看成是一个新的box, 该box的左上角坐标是box_a和box_b左上角坐标的较大值, 右下角坐标是box_a和box_b的右下角坐标的较小值
    A = box_a.size(0)
    B = box_b.size(0)
    # box_a 左上角/右下角坐标 expand以后, 维度会变成(A,B,2), 其中, 具体可看 expand 的相关原理. box_b也是同理, 这样做是为了得到a中某个box与b中某个box的左上角(min_xy)的较大者(max)
    # unsqueeze 为增加维度的数量, expand 为扩展维度的大小
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A,B,2),
                        box_b[:, :2].unsqueeze(0).expand(A,B,2)) # 在box_a的 A 和 2 之间增加一个维度, 并将维度扩展到 B. box_b 同理
    # 求右下角(max_xy)的较小者(min)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A,B,2),
                        box_b[:, 2:].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy, min_xy), min=0) # 右下角减去左上角, 如果为负值, 说明没有交集, 置为0
    return inter[:, :, 0] * inter[:, :, 0] # 高×宽, 返回交集的面积, shape 刚好为 [A, B]


def jaccard(box_a, box_b):
    # A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]), 代表了 box_a 和 box_b 两个集合中任意两个 box之间的交并比
    inter = intersect(box_a, box_b) # 求任意两个box的交集面积, shape为[A, B], 即[num_obj, num_priors]
    area_a = ((box_a[:,2]-box_a[:,0]) * (box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:,2]-box_b[:,0]) * (box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter) # [A,B], 这里会将A中的元素复制B次
    union = area_a + area_b - inter
    return inter / union # [A, B], 返回任意两个box之间的交并比, res[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.


def encode(matched, priors, variances):
    # 对边框坐标进行编码, 需要宽度方差和高度方差两个参数, 具体公式可以参见原文公式(2)
    # matched: [num_priors,4] 存储的是与priorbox匹配的gtbox的坐标. 形式为(xmin, ymin, xmax, ymax)
    # priors: [num_priors, 4] 存储的是priorbox的坐标. 形式为(cx, cy, w, h)
    # return : encoded boxes: [num_priors, 4]
    g_cxy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标, 获得中心坐标的偏移量
    g_cxy /= (variances[0]*priors[:, 2:]) # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示
    #variances[0]为0.1, 令其分别乘以w和h, 得到d_i^w 和 d_i^h
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:] # 令互相匹配的gtbox的宽高除以priorbox的宽高.
    g_wh = torch.log(g_wh) / variances[1] # 这里这个variances[1]=0.2 不太懂是为什么.
    return torch.cat([g_cxy, g_wh], 1) # 将编码后的中心坐标和宽高``连接起来, 返回 [num_priors, 4]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # threshold: (float) 确定是否匹配的交并比阈值
    # truths: (tensor: [num_obj, 4]) 存储真实 box 的边框坐标
    # priors: (tensor: [num_priors, 4], 即[8732, 4]), 存储推荐框的坐标, 注意, 此时的框是 default box, 而不是 SSD 网络预测出来的框的坐标, 预测的结果存储在 loc_data中, 其 shape 为[num_obj, 8732, 4].
    # variances: cfg['variance'], [0.1, 0.2], 用于将坐标转换成方便训练的形式(参考RCNN系列对边框坐标的处理)
    # labels: (tensor: [num_obj]), 代表了每个真实 box 对应的类别的编号
    # loc_t: (tensor: [batches, 8732, 4]),
    # conf_t: (tensor: [batches, 8732]),
    # idx: batches 中图片的序号, 标识当前正在处理的 image 在 batches 中的序号
    overlaps = jaccard(truths, point_form(priors)) # [A, B], 返回任意两个box之间的交并比, overlaps[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.

    # 二部图匹配(Bipartite Matching)
    # [num_objs,1], 得到对于每个 gt box 来说的匹配度最高的 prior box, 前者存储交并比, 后者存储prior box在num_priors中的索引位置
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # keepdim=True, 因此shape为[num_objs,1]
    # [1, num_priors], 即[1,8732], 同理, 得到对于每个 prior box 来说的匹配度最高的 gt box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_prior_idx.squeeze_(1) # 上面特意保留了维度(keepdim=True), 这里又都把维度 squeeze/reduce 了, 实际上只需用默认的 keepdim=False 就可以自动 squeeze/reduce 维度.
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 维度压缩后变为[num_priors], best_prior_idx 维度为[num_objs],
    # 该语句会将与gt box匹配度最好的prior box 的交并比置为 2, 确保其最大, 以免防止某些 gtbox 没有匹配的 priorbox.

    # 假想一种极端情况, 所有的priorbox与某个gtbox(标记为G)的交并比为1, 而其他gtbox分别有一个交并比
    # 最高的priorbox, 但是肯定小于1(因为其他的gtbox与G的交并比肯定小于1), 这样一来, 就会使得所有
    # 的priorbox都与G匹配, 为了防止这种情况, 我们将那些对gtbox来说, 具有最高交并比的priorbox,
    # 强制进行互相匹配, 即令best_truth_idx[best_prior_idx[j]] = j, 详细见下面的for循环

    # 注意!!: 因为 gt box 的数量要远远少于 prior box 的数量, 因此, 同一个 gt box 会与多个 prior box 匹配.
    for j in range(best_prior_idx.size(0)): # range:0~num_obj-1
        best_truth_idx[best_prior_idx[j]] = j
        # best_prior_idx[j] 代表与box_a的第j个box交并比最高的 prior box 的下标, 将与该 gtbox
        # 匹配度最好的 prior box 的下标改为j, 由此,完成了该 gtbox 与第j个 prior box 的匹配.
        # 这里的循环只会进行num_obj次, 剩余的匹配为 best_truth_idx 中原本的值.
        # 这里处理的情况是, priorbox中第i个box与gtbox中第k个box的交并比最高,
        # 即 best_truth_idx[i]= k
        # 但是对于best_prior_idx[k]来说, 它却与priorbox的第l个box有着最高的交并比,
        # 即best_prior_idx[k]=l
        # 而对于gtbox的另一个边框gtbox[j]来说, 它与priorbox[i]的交并比最大,
        # 即但是对于best_prior_idx[j] = i.
        # 那么, 此时, 我们就应该将best_truth_idx[i]= k 修改成 best_truth_idx[i]= j.
        # 即令 priorbox[i] 与 gtbox[j]对应.
        # 这样做的原因: 防止某个gtbox没有匹配的 prior box.
    matches = truths[best_truth_idx]
    # truths 的shape 为[num_objs, 4], 而best_truth_idx是一个指示下标的列表, 列表长度为 8732,
    # 列表中的下标范围为0~num_objs-1, 代表的是与每个priorbox匹配的gtbox的下标
    # 上面的表达式会返回一个shape为 [num_priors, 4], 即 [8732, 4] 的tensor, 代表的就是与每个priorbox匹配的gtbox的坐标值.
    conf = labels[best_truth_idx]+1 # 与上面的语句道理差不多, 这里得到的是每个prior box匹配的类别编号, shape 为[8732]
    conf[best_truth_overlap < threshold] = 0 # 将与gtbox的交并比小于阈值的置为0 , 即认为是非物体框
    loc = encode(matches, priors, variances) # 返回编码后的中心坐标和宽高.
    loc_t[idx] = loc  # 设置第idx张图片的gt编码坐标信息
    conf_t[idx] = conf  # 设置第idx张图片的编号信息.(大于0即为物体编号, 认为有物体, 小于0认为是背景)

    return loc_t, conf_t