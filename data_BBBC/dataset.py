# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:50
# @Author  : Xiaoyu Liu
# @Software: PyCharm
import os
import cv2
import sys
import tifffile
import torch
import random
import numpy as np
from PIL import Image
import os
import torch
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

from .utils.utils import center_crop_2d
from .utils.affinity_ours import multi_offset, gen_affs_ours
from .data.data_segmentation import seg_widen_border, weight_binary_ratio
from .data.data_consistency import Filp_EMA
from .utils.utils import remove_list
from .utils.consistency_aug import tensor2img, img2tensor, add_gauss_noise
from .utils.consistency_aug import add_gauss_blur, add_intensity, add_mask

class ToLogits(object):
    def __init__(self, expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
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

class BBBC(Dataset):
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

    def __init__(self, dir, mode, size):
        self.size = size  # img size after crop
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")

        self.flip = True
        self.crop = True
        self.if_ema_flip = True
        self.if_ema_noise = False
        self.if_ema_blur = False
        self.if_ema_intensity = True
        self.if_ema_mask = True
        self.data_folder = self.dir
        self.padding = 30
        self.separate_weight = True

        self.dir_img = os.path.join(self.data_folder, 'images')
        # self.dir_lb = os.path.join(self.data_folder, 'masks')
        self.dir_lb = os.path.join(self.data_folder, 'label_instance')
        self.dir_meta = os.path.join(self.data_folder, 'metadata')

        # augmentation
        self.if_scale_aug = True
        self.if_filp_aug = True
        self.if_elastic_aug = True
        self.if_intensity_aug = True
        self.if_rotation_aug = True

        if self.mode == "train":
            f_txt = open(os.path.join(self.dir_meta, 'training.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "validation":
            # f_txt = open(os.path.join(self.dir_meta, 'validation.txt'), 'r')
            # valid_set = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            # f_txt.close()

            # use test set as valid set directly
            f_txt = open(os.path.join(self.dir_meta, 'test.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "test":
            f_txt = open(os.path.join(self.dir_meta, 'test.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        else:
            raise NotImplementedError
        print('The number of %s image is %d' % (self.mode, len(self.id_img)))
        self.ema_flip = Filp_EMA()

        # padding for random rotation
        self.crop_size = [256, 256]
        self.crop_from_origin = [0, 0]
        # self.padding = 30
        self.crop_from_origin[0] = self.crop_size[0] + 2 * self.padding
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.img_size = [520+2*self.padding, 696+2*self.padding]
        self.offsets = multi_offset([1], neighbor=4)
        # augmentation initoalization
        self.augs_init()

    def __len__(self):
        return len(self.id_img)

    def __getitem__(self, id):

        if self.mode == 'train':
            k = random.randint(0, len(self.id_img) - 1)
            # read raw image
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))
            # # strip the first channel
            # if len(label.shape) == 3:
            #     label = label[:,:,0]
            # # label the annotations nicely to prepare for future filtering operation
            # label = skimage.morphology.label(label)
            # # filter small objects, e.g. micronulcei
            # label = skimage.morphology.remove_small_objects(label, min_size=25)

            # raw images padding
            imgs = np.pad(imgs, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            label = np.pad(label, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

            random_x = random.randint(0, self.img_size[0] - self.crop_from_origin[0])
            random_y = random.randint(0, self.img_size[1] - self.crop_from_origin[1])
            imgs = imgs[random_x:random_x + self.crop_from_origin[0], \
                   random_y:random_y + self.crop_from_origin[1]]
            label = label[random_x:random_x + self.crop_from_origin[0], \
                    random_y:random_y + self.crop_from_origin[1]]

            data = {'image': imgs, 'label': label}
            if np.random.rand() < 0.8:
                data = self.augs_mix(data)
            imgs = data['image']
            label = data['label']
            imgs = center_crop_2d(imgs, det_shape=self.crop_size)
            label = center_crop_2d(label, det_shape=self.crop_size)
            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)  #input channels is 3

            label_numpy = label.copy()

            lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)

            if self.separate_weight:
                weightmap = np.zeros_like(lb_affs)
                # weightmap1 = np.zeros_like(lb_affs1)
                # weightmap2 = np.zeros_like(lb_affs2)
                # weightmap3 = np.zeros_like(lb_affs3)
                # weightmap4 = np.zeros_like(lb_affs4)
                for i in range(lb_affs.shape[0]):
                    weightmap[i] = weight_binary_ratio(lb_affs[i])
                # for i in range(lb_affs1.shape[0]):
                #     weightmap1[i] = weight_binary_ratio(lb_affs1[i])
                # for i in range(lb_affs2.shape[0]):
                #     weightmap2[i] = weight_binary_ratio(lb_affs2[i])
                # for i in range(lb_affs3.shape[0]):
                #     weightmap3[i] = weight_binary_ratio(lb_affs3[i])
                # for i in range(lb_affs4.shape[0]):
                #     weightmap4[i] = weight_binary_ratio(lb_affs4[i])
            else:
                weightmap = weight_binary_ratio(lb_affs)
                # weightmap1 = weight_binary_ratio(lb_affs1)
                # weightmap2 = weight_binary_ratio(lb_affs2)
                # weightmap3 = weight_binary_ratio(lb_affs3)
                # weightmap4 = weight_binary_ratio(lb_affs4)

            lb_affs = torch.from_numpy(lb_affs)
            weightmap = torch.from_numpy(weightmap)
            affs_mask = torch.from_numpy(affs_mask)
            # down1 = torch.from_numpy(np.concatenate([lb_affs1, weightmap1, affs_mask1], axis=0))
            # down2 = torch.from_numpy(np.concatenate([lb_affs2, weightmap2, affs_mask2], axis=0))
            # down3 = torch.from_numpy(np.concatenate([lb_affs3, weightmap3, affs_mask3], axis=0))
            # down4 = torch.from_numpy(np.concatenate([lb_affs4, weightmap4, affs_mask4], axis=0))

            ema_data = imgs.copy()
            if self.if_ema_noise:
                ema_data = add_gauss_noise(ema_data)

            if self.if_ema_blur:
                ema_data = add_gauss_blur(ema_data)

            if self.if_ema_intensity:
                ema_data = add_intensity(ema_data)

            if self.if_ema_mask:
                label_mask = label_numpy.copy()
                label_mask[label_mask != 0] = 1
                ema_data = add_mask(ema_data, label_mask)

            if self.if_ema_flip:
                ema_data, rule = self.ema_flip(ema_data)
                rule = torch.from_numpy(rule.astype(np.float32))
            else:
                rule = torch.from_numpy(np.asarray([0, 0, 0], dtype=np.float32))

            imgs = torch.from_numpy(imgs)
            label = label.astype(np.float32)
            fg = label > 0
            fg = fg.astype(np.uint8)
            fg = torch.from_numpy(fg[np.newaxis, :, :].copy())

            label = torch.from_numpy(label[np.newaxis, :, :])
            ema_data = torch.from_numpy(np.ascontiguousarray(ema_data, dtype=np.float32))


            return imgs, label, fg, lb_affs, weightmap, affs_mask  #loss_temp = criterion(affs_temp*affs_mask, target*affs_mask, weightmap)

            # return {'image': imgs,
            #         'affs': lb_affs,
            #         'wmap': weightmap,
            #         'seg': label,
            #         'mask': affs_mask,
            #         'down1': down1,
            #         'down2': down2,
            #         'down3': down3,
            #         'down4': down4,
            #         'ema_image': ema_data,
            #         'rules': rule}

        elif self.mode == 'validation':
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))

            # if self.padding:
            # imgs = np.pad(imgs, ((92,92),(4,4)), mode='reflect')  # [704, 704]
            # label = np.pad(label, ((92,92),(4,4)), mode='reflect')
            imgs = np.pad(imgs, ((92, 92), (4, 4)), mode='constant')  # [704, 704]
            label = np.pad(label, ((92, 92), (4, 4)), mode='constant')

            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)
            imgs = torch.from_numpy(imgs)
            rule = torch.from_numpy(np.asarray([0, 0, 0], dtype=np.float32))

            label_numpy = label.copy()
            lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
            if self.separate_weight:
                weightmap = np.zeros_like(lb_affs)
                for i in range(len(self.offsets)):
                    weightmap[i] = weight_binary_ratio(lb_affs[i])
            else:
                weightmap = weight_binary_ratio(lb_affs)

            lb_affs = torch.from_numpy(lb_affs)
            weightmap = torch.from_numpy(weightmap)
            affs_mask = torch.from_numpy(affs_mask)
            fg = label > 0
            fg = fg.astype(np.uint8)
            fg = torch.from_numpy(fg[np.newaxis, :, :].copy())
            label = torch.from_numpy(label[np.newaxis, :, :].astype(np.float32))

            return imgs, label, fg, lb_affs, weightmap, affs_mask
            # return {'image': imgs,
            #         'affs': lb_affs,
            #         'wmap': weightmap,
            #         'seg': label,
            #         'mask': affs_mask,
            #         'down1': label,
            #         'down2': label,
            #         'down3': label,
            #         'down4': label,
            #         'ema_image': imgs,
            #         'rules': rule}

        else:
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))

            # if self.padding:
            # imgs = np.pad(imgs, ((92,92),(4,4)), mode='reflect')  # [704, 704]
            # label = np.pad(label, ((92,92),(4,4)), mode='reflect')
            imgs = np.pad(imgs, ((92, 92), (4, 4)), mode='constant')  # [704, 704]
            label = np.pad(label, ((92, 92), (4, 4)), mode='constant')

            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)
            imgs = torch.from_numpy(imgs)
            rule = torch.from_numpy(np.asarray([0, 0, 0], dtype=np.float32))
            label = torch.from_numpy(label[np.newaxis, :, :].astype(np.float32))
            # fg = label > 0
            # fg = fg.astype(np.uint8)

            return imgs, label