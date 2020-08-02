# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.25 23:01
@file    : ssd_prior.py
@desc    : ssd先验框设置
"""
from itertools import product as product
from math import sqrt
import torch


class PriorBox(object):
    # 所谓priorbox实际上就是网格中每一个cell推荐的box
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=21]
        # coco, voc的相关配置都来自于data/cfg.py 文件
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        # self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # 存放的是feature map的尺寸:38,19,10,5,3,1
            # from itertools import product as product
            for i, j in product(range(f), repeat=2):
                # 这里实际上可以用最普通的for循环嵌套来代替, 主要目的是产生anchor的坐标(i,j)

                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.

                # aspect_ratios 为1时对应的box
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改.

        output = torch.Tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1, min=0) # clamp_ 是clamp的原地执行版本
        return output # 输出default box坐标(可以理解为anchor box)


def get_prior_box(feature_map = [19, 10, 5, 3, 2, 1],
                  smin=0.2,
                  smax=0.95,
                  aspect_ratios=[1, 2, 3, 1/2, 1/3]):
    """
    计算先验框，先验框设置第一个特征图的先验框个数为3个，其尺度对应长宽比为[(0.1,1),(0.2,2),(0.2,1/2)]
    其余特征图的先验框个数为6个(顺序为[1, 1', 2, 3, 1/2, 1/3])，其中最后一个特征图计算另外一个长宽比为1的尺度时，为防止溢出，最大为1
    :param feature_map: 列表，其中包含特征图的size
    :param smin:  最小尺度，0~1之间
    :param smax:  最大尺度，0~1之间
    :param aspect_ratios:  长宽比设置
    :return:  numpy， 返回每个先验框位置的坐标，[先验框总数量，4]
    """
    # feature_map = [32, 16, 8, 4, 2, 1]  # input_shape:512
    feature_map_num = len(feature_map)

    anchor_box = []
    add_value = (smax - smin) / (feature_map_num - 1)
    for k, f in enumerate(feature_map):
        for i, j in product(range(f), repeat=2):
            # 中点坐标
            cx = (i + 0.5) / f
            cy = (j + 0.5) / f

            # 尺度
            sk = smin + add_value*(k)

            # 长宽比为1
            if k == 0:
                anchor_box.append((cx, cy, 0.1, 0.1))
                ars = [1, 2, 1/2]
            else:
                anchor_box.append((cx, cy, sk, sk))
                ars = aspect_ratios

                sk_next = sk+add_value
                if sk_next > 1:
                    sk_next=1
                sk_ = sqrt(sk*sk_next)
                anchor_box.append((cx, cy, sk_, sk_))

            # 长宽比为其他时
            for ar in ars[1:]:
                w = sk * sqrt(ar)
                h = sk / sqrt(ar)
                anchor_box.append((cx, cy, w, h))

    anchor_box = torch.tensor(anchor_box)
    xmin_ymin = anchor_box[:, :2] - anchor_box[:, 2:] / 2
    xmax_ymax = anchor_box[:, :2] + anchor_box[:, 2:] / 2
    anchor_box = torch.cat((xmin_ymin, xmax_ymax), 1)

    return anchor_box.clamp(min=0, max=1)


if __name__ == '__main__':
    # MOBILEV2_512 = {
    #     "feature_maps": [32, 16, 8, 4, 2, 1],
    #     "min_dim": 512,
    #     "steps": [16, 32, 64, 128, 256, 512],
    #     "min_sizes": [102.4, 174.08, 245.76, 317.44, 389.12, 460.8],
    #     "max_sizes": [174.08, 245.76, 317.44, 389.12, 460.8, 512],
    #     "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    #     "variance": [0.1, 0.2],
    #     "clip": True,
    # }
    #
    # priorbox = PriorBox(MOBILEV2_512)
    # output = priorbox.forward()
    # print("output shape: ", output)

    anchor_box = get_prior_box()
    # print(anchor_box)