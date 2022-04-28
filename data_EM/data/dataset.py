# -*- coding: utf-8 -*-
# @Time    : 2021/4/27 14:50
# @Author  : Xiaoyu Liu
# @Email   : Xiaoyliu@mail.ustc.edu.cn
# @Software: PyCharm

import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from skimage import io
from torchvision import transforms as tfs
import PIL.Image as Image
import cv2
from torchvision import transforms
import numpy
import random


class ToLogits(object):
    def __init__(self, expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(numpy.array(pic, numpy.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(numpy.array(pic, numpy.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img


class CVPPP(Dataset):
    def __init__(self, dir, mode, size):
        self.size = size
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        self.dir = os.path.join(dir, mode)
        # self.path_labels = os.path.join(dir,'leftImg8bit',mode)
        self.id_num = os.listdir(self.dir)  # all file
        self.id_img = [f for f in self.id_num if 'rgb' in f]
        self.id_label = [f for f in self.id_num if 'label' in f]
        self.id_fg = [f for f in self.id_num if 'fg' in f]
        self.id_boundary = [f for f in self.id_num if 'boundary' in f]
        if not self.mode == 'validation':
            self.id_img.sort(key=lambda x: int(x[5:8]))
            self.id_label.sort(key=lambda x: int(x[5:8]))
            self.id_fg.sort(key=lambda x: int(x[5:8]))
            self.id_boundary.sort(key=lambda x: int(x[5:8]))
        else:
            self.id_img.sort(key=lambda x: int(x[21:23]))
            self.id_label.sort(key=lambda x: int(x[21:23]))
            self.id_fg.sort(key=lambda x: int(x[21:23]))
            self.id_boundary.sort(key=lambda x: int(x[21:23]))

        self.transform_val = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(448, scale=(0.7, 1.)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.target_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(448, scale=(0.7, 1.), interpolation=0),
             ToLogits()])

    def __len__(self):
        return len(self.id_img)

    def __getitem__(self, id):

        if self.mode == 'train':
            data = Image.open(os.path.join(self.dir, self.id_img[id])).convert('RGB')  #
            label = Image.open(os.path.join(self.dir, self.id_label[id]))
            fg = Image.open(os.path.join(self.dir, self.id_fg[id]))
            boundary = Image.open(os.path.join(self.dir, self.id_boundary[id]))

            seed = np.random.randint(2147483647)
            random.seed(seed)
            data = self.transform(data)

            random.seed(seed)
            label = self.target_transform(label)

            random.seed(seed)
            fg = self.target_transform(fg)

            random.seed(seed)
            boundary = self.target_transform(boundary)

            inverse1, pack1 = torch.unique(label, return_inverse=True)
            pack1 = pack1.reshape(label.shape)

            inverse1 = torch.arange(0, len(inverse1))
            label = inverse1[pack1]

            return data, label, fg, 1-boundary/255.0

        elif self.mode=='validation':

            data = np.array(Image.open(os.path.join(self.dir, self.id_img[id])).convert('RGB')).transpose((2, 0, 1))
            label = np.array(Image.open(os.path.join(self.dir, self.id_label[id])))
            boundary = np.array(Image.open(os.path.join(self.dir, self.id_boundary[id])))
            # data, label = self.crop(data, label)

            inverse1, pack1 = np.unique(label, return_inverse=True)
            pack1 = pack1.reshape(label.shape)
            inverse1 = np.arange(0, inverse1.size)
            label = inverse1[pack1]
            # print(label.max())

            # print(label.max(),label.min())
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            boundary = torch.from_numpy(boundary)

            data = self.transform_val(data.float())
            # print(data.shape,label.shape)
            return data[:,:,:-1], label.unsqueeze(0)[:,:,:-1],(1-boundary.unsqueeze(0)/255.0)[:,:,:-1]

        elif self.mode == 'test':

            data = Image.open(os.path.join(self.dir, self.id_img[id])).convert('RGB')  #
            data = self.transform_test(data)
            h, w = data.shape[-2], data.shape[-1]
            fg = np.array(Image.open(os.path.join(self.dir, self.id_fg[id])))

            inverse1, pack1 = np.unique(fg, return_inverse=True)
            pack1 = pack1.reshape(fg.shape)
            inverse1 = np.arange(0, inverse1.size)
            fg = inverse1[pack1]
            # print(label.max(),label.min())

            fg = torch.from_numpy(fg)
            # print(data.shape,label.shape)
            return data, fg.unsqueeze(0)


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

    def __call__(self, image, label, fg=None):

        h, w = image.shape[-2:]
        new_h, new_w = self.output_size
        if h < new_h or w < new_h:
            image_n = np.zeros((image.shape[0], new_h, new_w), dtype=image.dtype)

            for i in range(image.shape[0]):
                image_n[i] = cv2.resize(image[i], dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)
            return image_n, label
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h,
                left: left + new_w]
        label = label[top: top + new_h,
                left: left + new_w]
        if not fg is None:
            fg = fg[top: top + new_h,
                 left: left + new_w]
            return image, label, fg

        return image, label


class AC34(Dataset):
    def __init__(self, dir, mode, size):
        self.size = size  # img size after crop

        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        self.path_i3 = os.path.join(dir, 'AC3_inputs')
        self.path_i4 = os.path.join(dir, 'AC4_inputs')
        self.path_l3 = os.path.join(dir, 'AC3_labels')
        self.path_l4 = os.path.join(dir, 'AC4_labels')
        id_i3 = os.listdir(self.path_i3)
        id_i3.sort(key=lambda x: int(x[-8:-4]))
        self.data_AC3 = [os.path.join(self.path_i3, x) for x in id_i3]
        self.label_AC3 = [os.path.join(self.path_l3, x.replace('png', 'tif')) for x in id_i3]

        id_i4 = os.listdir(self.path_i4)
        id_i4.sort(key=lambda x: int(x[-8:-4]))
        self.data_AC4 = [os.path.join(self.path_i4, x) for x in id_i4]
        self.label_AC4 = [os.path.join(self.path_l4, x.replace('png', 'tif')) for x in id_i4]

        self.crop = RandomCrop((self.size, self.size))

        if mode == "train":
            self.data = self.data_AC4 + self.data_AC3[-116:]
            self.label = self.label_AC4 + self.label_AC3[-116:]
        elif mode == "validation":
            self.data = self.data_AC3[-156:-116]
            self.label = self.label_AC3[-156:-116]
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        data = io.imread(self.data[id])
        label = io.imread(self.label[id])
        # print(label.shape, label.dtype)
        # print(len(np.unique(label)))
        data, label = self.crop(data, label)
        # print(label.shape,label.dtype)
        # print(len(np.unique(label)),label.max(),label.min())
        inverse1, pack1 = np.unique(label, return_inverse=True)
        pack1 = pack1.reshape(label.shape)
        inverse1 = np.arange(0, inverse1.size)
        label = inverse1[pack1]
        # print(label.max())
        while label.max() > 21:
            id = (id + 1) % len(self.data)
            data = io.imread(self.data[id])
            label = io.imread(self.label[id])
            # print(label.shape, label.dtype)
            # print(len(np.unique(label)))
            data, label = self.crop(data, label)
            # print(label.shape,label.dtype)
            # print(len(np.unique(label)),label.max(),label.min())
            inverse1, pack1 = np.unique(label, return_inverse=True)
            pack1 = pack1.reshape(label.shape)
            inverse1 = np.arange(0, inverse1.size)
            label = inverse1[pack1]

        # print(label.max(),label.min())
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data.unsqueeze(0), label.unsqueeze(0)
