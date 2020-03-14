#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 03:06:38 2019

@author: fayyaz
"""
import torch
import glob
import os
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from ICIAR2018_master.src.patch_extractor import PatchExtractor
class loader(Dataset):
    def __init__(self, path, stride,patch_size, augment=False):
        super().__init__()

        if os.path.isdir(path):
            names = [name for name in glob.glob(path + '/*.png')]
        else:
            names = [path]
        self.patch_size=patch_size
        self.path = path
        self.stride = stride
        self.augment = augment
        self.names = list(sorted(names))

    def __getitem__(self, index):
        file = self.names[index]
        with Image.open(file) as img:

            bins = 8 if self.augment else 1
            extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
            b = torch.zeros((bins, extractor.shape()[0] * extractor.shape()[1], 3, self.patch_size, self.patch_size))

            for k in range(bins):

                if k % 4 != 0:
                    img = img.rotate((k % 4) * 90)

                if k // 4 != 0:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
                patches = extractor.extract_patches()
                for i in range(len(patches)):
                    b[k, i] = transforms.ToTensor()(patches[i])

            return b, file

    def __len__(self):
        return len(self.names)
