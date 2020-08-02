# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.19 22:18
@file    : train.py
@desc    : ssdlite_mobilenet_v2 训练代码
"""
import torch
import datetime, time
import numpy as np
from net.ssdlite_mobilenet_v2 import ssdlite_mobilenet_v2
from net.backbones.mobilenet_v2 import MobileNetV2
from torch.autograd import Variable
from net.losses.ssd_loss import MultiBoxLoss
from net.anchor.ssd_prior import get_prior_box
from data.dataset import Dataset_ori, detection_collate
from data.data_augment import SSDAugmentation
from torch.utils.data import DataLoader
import torch.optim as optim


def load_model(num_classes, pretrain_model=True):
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
        model = ssdlite_mobilenet_v2(num_class=num_classes)
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
    # 参数设置
    img_files = r'H:\deep_learning\datasets\coco2014\train2014\train.txt'
    epoch = 10
    batch_size = 2
    num_workers = 1
    num_classes = 80
    overlap_thresh = 0.5
    negpos_ratio = 3
    img_size = 300
    learning_rate = 0.1
    weight_decay = 1e-4

    train_dataset = Dataset_ori(img_files=img_files,
                                input_size=(img_size, img_size),
                                augment=SSDAugmentation()
                                )
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=detection_collate)

    model = load_model(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    loss = MultiBoxLoss(num_classes, overlap_thresh, negpos_ratio, True)
    prior = Variable(get_prior_box().data.cuda())
    print(len(train_dataloader))

    train_time = []
    for epo in range(epoch):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            ctime = time.time()
            optimizer.zero_grad()

            target = [Variable(anno.cuda()) for anno in target]
            input = Variable(img.cuda())
            output = model(input)
            p1 = (output[0], output[1], prior)

            l, c = loss(p1, target)
            total_loss = l + c

            cost_time = time.time() - ctime
            train_time.append(cost_time)
            mean_time = np.mean(train_time)
            print("{} epoch:{} | batch:{}/{} | loc_loss:{} | conf_loss:{} | loss:{} | time:{}s | real_time:{}h".format(
                str(datetime.datetime.now()).split('.')[0], epoch, i+1, len(train_dataloader), l, c, total_loss,
                round(cost_time, 2), round((len(train_dataloader)-1-i+(epoch-i-epo)*len(train_dataloader))*mean_time/3600, 2)
            ))
            print("1")
            total_loss.backward()
            print("2")

            optimizer.step()
            print("3")


if __name__ == '__main__':
    train()
