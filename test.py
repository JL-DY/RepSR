import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RepSRNet import RepSR_Net
from torch.utils.data import DataLoader
import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='RepSR')
parser.add_argument('--config', type=str, default="./configs/RepSR_m4c16.yml", help = 'pre-config file for training')
args = parser.parse_args()
if args.config:
    opt = vars(args)
    yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(yaml_args)

if __name__ == '__main__':
    device = torch.device('cuda')
    input_folder = "input_img"
    re_parameterized = 0  # 是否进行结构重新参数化,暂未实现
    model_channel = 1     # 模型的输入
    img_channel = 1 # 单通道图像or三通道
    img_range = 1 # 图像数据范围是0-1 or 0-255

    model_repsr = RepSR_Net(m=4, c=16, scale=4, colors=model_channel, opt=opt, device=device).to(device)

    model_repsr.load_state_dict(torch.load("./weights/Repsr-x4-m4c16-2024-0628-1054/models/model_x4_9.pt", map_location=device))
    model_repsr.eval()

    path = os.listdir(input_folder)
    for i in path:
        img_path = os.path.join(input_folder, i)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if model_channel==1 and img_channel==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y_channel = img[:, :, 0]
            CrCb = img[:, :, 1:]
            Y_channel = np.array(np.expand_dims(np.expand_dims(Y_channel, axis=0), axis=0), dtype=np.float32)/255
        
            Y_channel = torch.tensor(Y_channel).to(device)
            with torch.no_grad():
                result = model_repsr(Y_channel)
            result = result[0].cpu().numpy()
            result = np.clip(result, 0, img_range)
            result = np.array(result*(255/img_range), dtype=np.uint8)
            result = np.transpose(result, (1, 2, 0))
            # result = Edge_Addweight(result, 5, 0.7).edge_addweight()

            new_h, new_w, new_c = np.shape(result)
            new_CrCb = cv2.resize(CrCb, (new_w, new_h))
            new_result = cv2.merge((result, new_CrCb))
            new_result = cv2.cvtColor(new_result, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(img_path.replace(input_folder, "output_img"), result)

        if model_channel==1 and img_channel==1:
            img = np.array(np.expand_dims(np.expand_dims(img, axis=0), axis=0), dtype=np.float32)/255
        
            img = torch.tensor(img).to(device)
            with torch.no_grad():
                result = model_repsr(img)
            result = result[0].cpu().numpy()
            result = np.clip(result, 0, img_range)
            result = np.array(result*(255/img_range), dtype=np.uint8)
            result = np.transpose(result, (1, 2, 0))
            # result = Edge_Addweight(result, 5, 0.7).edge_addweight()
            cv2.imwrite(img_path.replace(input_folder, "output_img"), result)

        if model_channel == 3 and img_channel==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2)), dtype=np.float32)/255

            img = torch.tensor(img).to(device)
            with torch.no_grad():
                result = model_repsr(img)
            result = result[0].cpu().numpy()
            result = np.clip(result, 0, img_range)
            result = np.array(result*(255/img_range), dtype=np.uint8)
            result = np.squeeze(np.transpose(result, (1, 2, 0)))

            new_result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path.replace(input_folder, "output-img"), new_result)
