# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:50
# @Author  : Xiaoyu Liu
# @Software: PyCharm
import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from skimage import io
from torchvision import transforms as tfs
from .data_affinity import *
import random
from .augmentation import Flip
from .augmentation import Elastic
from .augmentation import Grayscale
from .augmentation import Rotate
from .augmentation import Rescale


class RandomCrop(object):
    """随机裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, label, seed=None):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if seed is not None:
            random.seed(seed)
        top = np.random.randint(0, h - new_h)
        if seed is not None:
            random.seed(seed)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        label = label[top: top + new_h,
                left: left + new_w]

        return image, label




class FIB25(Dataset):
    def __init__(self, dir, mode, size):
        self.size = size  # img size after crop
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        self.path_i1 = os.path.join(dir, 'tstvol-520-1_inputs')
        self.path_i2 = os.path.join(dir, 'tstvol-520-2_inputs')
        self.path_l1 = os.path.join(dir, 'tstvol-520-1_labels')
        self.path_l2 = os.path.join(dir, 'tstvol-520-2_labels')
        id_i1 = os.listdir(self.path_i1)
        id_i1.sort(key=lambda x: int(x[-8:-4]))
        self.data_fib1 = [os.path.join(self.path_i1, x) for x in id_i1]
        self.label_fib1 = [os.path.join(self.path_l1, x.replace('png', 'tif')) for x in id_i1]

        id_i2 = os.listdir(self.path_i2)
        id_i2.sort(key=lambda x: int(x[-8:-4]))
        self.data_fib2 = [os.path.join(self.path_i2, x) for x in id_i2]
        self.label_fib2 = [os.path.join(self.path_l2, x.replace('png', 'tif')) for x in id_i2]

        self.crop = RandomCrop((self.size, self.size))

        if mode == "train":
            self.data = self.data_fib1[:-50]
            self.label = self.label_fib1[:-50]
        elif mode == "validation":
            self.data = self.data_fib1[-50:]
            self.label = self.label_fib1[-50:]
        elif mode == "test":
            self.data = self.data_fib2
            self.label = self.label_fib2

        # print(self.data)

        self.augs_init()

        self.padding = 0

    def __len__(self):
        return len(self.data)

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    def augs_mix(self, data):
        if random.random() > 0.5:
            data = self.aug_flip(data)
        if random.random() > 0.5:
            data = self.aug_rotation(data)
        # if random.random() > 0.5:
        #     data = self.aug_rescale(data)
        if random.random() > 0.5:
            data = self.aug_elastic(data)
        if random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data

    def __getitem__(self, id):
        data = io.imread(self.data[id])
        label = io.imread(self.label[id])

        # print(label.shape, label.dtype)
        # print(len(np.unique(label)))
        # if not self.mode == "test":
        #     if self.mode == "validation":
        #         data, label = self.crop(data, label,seed=123)
        #     else:
        #         data,label = self.crop(data,label)
        data = data.astype(np.float32) / 255.0

        if self.mode == "train":

            pack = {'image': data[np.newaxis, :], 'label': label[np.newaxis, :]}
            if np.random.rand() < 0.5:
                pack = self.augs_mix(pack)
            data = pack['image']
            label = pack['label']

            data, label = self.crop(data[0], label[0])

        inverse1, pack1 = np.unique(label, return_inverse=True)
        pack1 = pack1.reshape(label.shape)
        inverse1 = np.arange(0, inverse1.size)
        label = inverse1[pack1]

        # while label.max()>59:
        #     id = (id+1)%len(self.data)
        #     data = io.imread(self.data[id])
        #     label = io.imread(self.label[id])
        #     # print(label.shape, label.dtype)
        #     # print(len(np.unique(label)))
        #     data, label = self.crop(data, label)
        #     # print(label.shape,label.dtype)
        #     # print(len(np.unique(label)),label.max(),label.min())
        #     inverse1, pack1 = np.unique(label, return_inverse=True)
        #     pack1 = pack1.reshape(label.shape)
        #     inverse1 = np.arange(0, inverse1.size)
        #     label = inverse1[pack1]

        affs_yx = seg_to_aff(seg_widen_border(label+1)).astype(np.float32)

        fg = label > 0
        fg = fg.astype(np.uint8)
        # print(label.max(),label.min())
        data = torch.from_numpy(data.copy())
        label = torch.from_numpy(label.copy())
        fg = torch.from_numpy(fg.copy())
        affs_yx = torch.from_numpy(affs_yx.copy())

        return data.unsqueeze(0), label.unsqueeze(0), fg.unsqueeze(0), affs_yx.unsqueeze(0)





















