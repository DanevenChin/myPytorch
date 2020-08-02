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
from data.data_augment import SSDAugmentation
from torch.utils.data import Dataset, DataLoader
import jpeg4py as jpeg


class Dataset_ori(Dataset):
    def __init__(self, img_files, input_size, annotation_type="yolo", augment=None):
        with open(img_files, 'r')as f:
            img_list = f.read().splitlines()
        self.img_list = img_list
        self.input_size = input_size
        self.augment = augment
        self.annotation_type = annotation_type

    def __getitem__(self, item):
        img_path = self.img_list[item]

        img = cv2.imread(img_path)
        h, w, c = img.shape

        # img = jpeg.JPEG(img_path).decode()

        # 读取标注文件，提取bbox和cls信息
        coods = []
        if self.annotation_type == "yolo":
            label_path = img_path.replace("JPEGImages", "labels").replace("jpg", "txt")
            with open(label_path, "r")as f:
                label_list = f.read().splitlines()

            for label in label_list:
                label_info = label.split(" ")
                cls = label_info[0]
                bbox = label_info[1:]

                # 将yolo格式(归一化的[cx, cy, w, h])转为归一化的[xmin, ymin, xmax, ymax]
                xmin = max(float(bbox[0]) - float(bbox[2]) / 2, 0)
                ymin = max(float(bbox[1]) - float(bbox[3]) / 2, 0)
                xmax = min(float(bbox[0]) + float(bbox[2]) / 2, w)
                ymax = min(float(bbox[1]) + float(bbox[3]) / 2, h)
                coods.append([xmin, ymin, xmax, ymax, int(cls)])

        target = np.array(coods)

        if self.augment:
            img, boxes, labels = self.augment(img,
                                                target[:, :4],
                                                target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img.transpose(2, 0, 1)), target

    def __len__(self):
        return len(self.img_list)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    img_files = r'H:\deep_learning\datasets\coco2014\train2014\train.txt'
    num_work = 7
    train_dataset = Dataset_ori(img_files=img_files,
                                input_size=(300, 300),
                                augment=SSDAugmentation()
                                )
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  collate_fn=detection_collate)
    for i, (img, target) in enumerate(train_dataset):
        print("train_dataset:", i, target)
        if i == 10:
            break

    for i, (img, target) in enumerate(train_dataloader):
        print("train_dataloader:", i, target)
        if i == 10:
            break

