# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020.7.26 20:55
@file    : dataset.py
@desc    : 数据集
"""
import os
import cv2
import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import jpeg4py as jpeg


class Dataset_ori(Dataset):
    def __init__(self, img_files, input_size, augment=None):
        with open(img_files, 'r')as f:
            img_list = f.read().splitlines()
        self.img_list = [img for img in img_list]
        self.input_size = input_size
        self.augment = augment

    def __getitem__(self, item):
        img_path = self.img_list[item].split(' ')[0]
        img_path = img_path.replace('/home', 'z:')

        # img = cv2.imread(img_path)
        # img = cv2.resize(img, self.input_size)
        # img = img.transpose(2, 0, 1)

        img = jpeg.JPEG(img_path).decode()
        print(img.shape)

        if self.augment:
            return self.augment(torch.from_numpy(img))
        else:
            return torch.from_numpy(img)

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    img_files = r'Z:\qindanfeng\work\YOLOv3\yolov3_voc\data\test_annotation.txt'
    num_work = 7
    train_dataset = Dataset_ori(img_files, (300, 300))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8,
                                  num_workers=num_work,
                                  shuffle=True)
    print(len(train_dataloader))
    time_list = []
    for epoch in range(5):
        ctime = time.time()
        for i, batch in enumerate(train_dataloader):
            print('[{}/{}]'.format(i, len(train_dataloader)))
            pass
        time_list.append(time.time() - ctime)
    print("共{}个进程, 平均耗时:{}s".format(num_work, np.mean(np.array(time_list))))

'''
默认的数据加载：
num_work  time(s)
	0      38.90
	4      22.40
	5      21.79
	6      22.49
	7      21.74
	8      22.56
	12     26.95

jpeg4py:


'''
