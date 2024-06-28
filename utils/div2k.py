import os
import glob
import random
import pickle
import numpy as np
import imageio
import cv2 as cv
import torch
import torch.utils.data as data
import skimage.color as sc
from skimage.filters import rank
from skimage.morphology import square
import time
from utils.utils import ndarray2tensor, filter2D
from utils.generate_kernels import kernels


class DIV2K(data.Dataset):
    def __init__(self, opt, HR_folder, LR_folder, train=True, augment=True, scale=2, colors=1, patch_size=96):
        super(DIV2K, self).__init__()
        self.opt = opt
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment   = augment
        self.train     = train
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size

        self.nums_trainset = 0
        self.hr_filenames = []
        self.lr_filenames = []
        ## generate dataset
        start_idx = 0
        end_idx = 0
        if self.train: start_idx, end_idx = 1, 801
        else:          start_idx, end_idx = 801, 901

        if opt["use_degradation"]:
            img_name = os.listdir(HR_folder)
            for i in img_name:
                hr_filename = os.path.join(self.HR_folder, i)
                self.hr_filenames.append(hr_filename)
        else:
            for i in range(start_idx, end_idx):
                idx = str(i).zfill(4)
                hr_filename = os.path.join(self.HR_folder, idx + self.img_postfix)
                lr_filename = os.path.join(self.LR_folder, 'X{}'.format(self.scale), idx + 'x{}'.format(self.scale) + self.img_postfix)
                self.hr_filenames.append(hr_filename)
                self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)
 
    def __len__(self):
        if self.train:
            return self.nums_trainset
        else:
            return len(self.hr_filenames)

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        
        if self.opt["degradation"]:
            hr = imageio.imread(self.hr_filenames[idx], pilmode="RGB")
            # hr = cv.imread(self.hr_filenames[idx], cv.IMREAD_UNCHANGED) / 255.0
            if self.colors == 1:
                hr = sc.rgb2ycbcr(hr)[:, :, 0:1] / 255.0
                hr = np.array(hr)

            if self.train:
                # crop patch randomly
                hr_h, hr_w, _ = hr.shape
                hp = self.patch_size
                hx, hy = random.randrange(0, hr_w - hp + 1), random.randrange(0, hr_h - hp + 1)
                hr_patch = hr[hy:hy+hp, hx:hx+hp, :]
                # augment data
                if self.augment:
                    #print("data augmentation!")
                    hflip = random.random() > 0.5
                    vflip = random.random() > 0.5
                    rot90 = random.random() > 0.5
                    if hflip: hr_patch = hr_patch[:, ::-1, :]
                    if vflip: hr_patch = hr_patch[::-1, :, :]
                    if rot90: hr_patch = hr_patch.transpose(1,0,2)
                # generate kernels
                data_dict = kernels(hr_patch, self.opt)
                return data_dict

        else:
            lr, hr = imageio.imread(self.lr_filenames[idx], pilmode="RGB"), imageio.imread(self.hr_filenames[idx], pilmode="RGB")
            # lr, hr = cv.imread(self.lr_filenames[idx], cv.IMREAD_GRAYSCALE) / 255.0, cv.imread(self.hr_filenames[idx], cv.IMREAD_GRAYSCALE) / 255.0
            if self.colors == 1:
                lr, hr = sc.rgb2ycbcr(lr)[:, :, 0:1] / 255.0, sc.rgb2ycbcr(hr)[:, :, 0:1] / 255.0

            if self.train:
                # crop patch randomly
                lr_h, lr_w, _ = lr.shape
                hp = self.patch_size
                lp = self.patch_size // self.scale
                lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
                hx, hy = lx * self.scale, ly * self.scale
                lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
                # augment data
                if self.augment:
                    #print("data augmentation!")
                    hflip = random.random() > 0.5
                    vflip = random.random() > 0.5
                    rot90 = random.random() > 0.5
                    if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
                    if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
                    if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
                # numpy to tensor
                lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
                return lr_patch, hr_patch

if __name__ == '__main__':
    HR_folder = '/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_HR'
    LR_folder = '/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_LR_bicubic'
    argment   = True
    div2k = DIV2K(HR_folder, LR_folder, train=True, argment=True, scale=2, colors=1, patch_size=96, repeat=168, store_in_ram=False)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        lr, hr = div2k[idx]
        print(lr.shape, hr.shape)
    end = time.time()
    print(end - start)
