# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.19 14:42
@file    : ssdlite_mobilenet_v2.py
@desc    : 
"""
import torch
from torch import nn
from net.backbones.mobilenet_v2 import MobileNetV2
from net.necks.ssdlite import ssd_body
from net.heads.ssd_head import multibox


class ssdlite_mobilenet_v2(nn.Module):
    def __init__(self, num_class, width_ratio=1.0, is_train=True):
        super(ssdlite_mobilenet_v2, self).__init__()
        # 参数
        self.is_train = is_train
        self.num_classes = num_class

        # 网络结构
        self.features = MobileNetV2(width_mult=width_ratio).features
        self.ssdlite = ssd_body()
        self.loc_layers, self.conf_layers = multibox(num_class, width_mult=width_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        source = []
        for layer in self.features[0:14]:
            x = layer(x)

        sub = getattr(self.features[14], "conv")
        x = sub[0](x)
        source.append(x)

        for layer in sub[1:]:
            x = layer(x)

        for layer in self.features[15:]:
            x = layer(x)
        source.append(x)

        for layer in self.ssdlite:
            x = layer(x)
            source.append(x)

        loc = []
        conf = []
        for (x, l, c) in zip(source, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.is_train:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        else:
            output = (
                loc,
                torch.max(self.sigmoid(conf.view(-1, self.num_classes)), dim=1)[0],
                torch.max(self.sigmoid(conf.view(-1, self.num_classes)), dim=1)[1],
            )

        return output

if __name__ == '__main__':
    import torchsummary as summary
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url

    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    }
    state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                          progress=True)
    # 读取参数
    mobv2 = MobileNetV2()
    mobv2.load_state_dict(state_dict)
    pretrained_dict = mobv2.state_dict()
    model = ssdlite_mobilenet_v2(num_class=1)
    model_dict = model.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for k,v in pretrained_dict.items():
        # print("-->", k, v)
        if k in model_dict:
            pretrained_dict = {k: v}
            print(k)

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)
    model = model.to("cuda")
    print(model)
