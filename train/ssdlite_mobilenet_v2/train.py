# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.19 22:18
@file    : train.py
@desc    : ssdlite_mobilenet_v2 训练代码
"""
from net.ssdlite_mobilenet_v2 import ssdlite_mobilenet_v2
from net.backbones.mobilenet_v2 import MobileNetV2
import torch
from torch.autograd import Variable
from net.losses.ssdlite_loss import compute_loss


def load_model(pretrain_model=True):
    if pretrain_model:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url

        model_urls = {
            'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        }
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=True)
        mobv2 = MobileNetV2()
        mobv2.load_state_dict(state_dict)
        pretrained_dict = mobv2.state_dict()
        model = ssdlite_mobilenet_v2(num_class=1)
        model_dict = model.state_dict()

        for k, v in pretrained_dict.items():
            # print("-->", k, v)
            if k in model_dict:
                pretrained_dict = {k: v}
                # print(k)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to("cuda")
    return model

def train():
    epoch = 10
    model = load_model()
    # print(model)

    for epo in range(epoch):
        model.train()
        input_tensor = torch.rand(8, 3, 300, 300).cuda()
        target_cls = torch.rand(8, 1917, 1)
        target_loc = torch.rand(8, 1917, 4)
        target = torch.cat((target_loc, target_cls), 2)
        input = Variable(input_tensor.cuda())
        print(input_tensor.shape)
        output = model(input)

        total_loss, loc_loss, cls_loss = compute_loss(output, target)
        print(total_loss)
        total_loss.backward()


if __name__ == '__main__':
    train()
