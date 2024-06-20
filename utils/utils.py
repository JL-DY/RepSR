import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision.models as models
import datetime
import cv2 as cv
import sys

def calc_psnr(sr, hr):
    # diff = (sr - hr) / 255.00
    diff = (sr - hr)
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    # def ssim(
    #     X,
    #     Y,
    #     data_range=255,
    #     size_average=True,
    #     win_size=11,
    #     win_sigma=1.5,
    #     win=None,
    #     K=(0.01, 0.03),
    #     nonnegative_ssim=False,
    # )
    # ssim_val = ssim(sr, hr, data_range=255, size_average=True)
    ssim_val = ssim(sr, hr, data_range=1, size_average=True)
    return float(ssim_val)
    
def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor


def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'set5': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'set14': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'b100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'u100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        }
    }
    return stat_dict

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img)
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
    
# 感知损失
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 9
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
  
if __name__ == '__main__':
    timestamp = cur_timestamp_str()
    print(timestamp)