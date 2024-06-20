import cv2
import math
import numpy as np
import torch.nn.functional as F
import os.path as osp
import random
import torch
from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.utils import img2tensor, filter2D
from torch.utils import data as data

# ------------------------ 生成 kernels 给降质过程中的Blur使用 ------------------------ #
# Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
def kernels(img_gt, opt):
    blur_kernel_size = opt["degradation"]['blur_kernel_size']
    kernel_list = opt["degradation"]['kernel_list']
    kernel_prob = opt["degradation"]['kernel_prob']  # a list for each kernel probability
    blur_sigma = opt["degradation"]['blur_sigma']
    betag_range = opt["degradation"]['betag_range']  # betag used in generalized Gaussian blur kernels
    betap_range = opt["degradation"]['betap_range']  # betap used in plateau blur kernels
    sinc_prob = opt["degradation"]['sinc_prob']  # the probability for sinc filters

    # blur settings for the second degradation
    blur_kernel_size2 = opt["degradation"]['blur_kernel_size2']
    kernel_list2 = opt["degradation"]['kernel_list2']
    kernel_prob2 = opt["degradation"]['kernel_prob2']
    blur_sigma2 = opt["degradation"]['blur_sigma2']
    betag_range2 = opt["degradation"]['betag_range2']
    betap_range2 = opt["degradation"]['betap_range2']
    sinc_prob2 = opt["degradation"]['sinc_prob2']

    # a final sinc filter
    final_sinc_prob = opt["degradation"]['final_sinc_prob']

    kernel_range1 = [v for v in range(3, blur_kernel_size, 2)]
    kernel_range2 = [v for v in range(3, blur_kernel_size2, 2)]
    pulse_tensor = torch.zeros(blur_kernel_size-2, blur_kernel_size-2).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[(blur_kernel_size-3)//2, (blur_kernel_size-3)//2] = 1

    # crop or pad to 400
    # TODO: 400 is hard-coded. You may change it accordingly
    h, w = img_gt.shape[0:2]
    crop_pad_size = opt["patch_size"]
    # pad
    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    # crop
    if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        h, w = img_gt.shape[0:2]
        # randomly choose top and left coordinates
        top = random.randint(0, h - crop_pad_size)
        left = random.randint(0, w - crop_pad_size)
        img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range1)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
                                    kernel_list,
                                    kernel_prob,
                                    kernel_size,
                                    blur_sigma,
                                    blur_sigma, [-math.pi, math.pi],
                                    betag_range,
                                    betap_range,
                                    noise_range=None)
    # pad kernel
    pad_size = (blur_kernel_size-2 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range2)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
                                    kernel_list2,
                                    kernel_prob2,
                                    kernel_size,
                                    blur_sigma2,
                                    blur_sigma2, [-math.pi, math.pi],
                                    betag_range2,
                                    betap_range2,
                                    noise_range=None)

    # pad kernel
    pad_size = (blur_kernel_size2-2 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range1)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=blur_kernel_size-2)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    # BGR to RGB, HWC to CHW, numpy to tensor
    img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)

    return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
    return return_d
