import os
import cv2 as cv
import numpy as np

# 未归一化时: mean=110.9529, var=3333.466
# 归一化时：mean=0.4351, var=0.05126

img_folder = "/media/Data/jl/sr_data/DIV2K/DIV2K_train_LR_bicubic/X4/"

mean = []
var = []

img_list = os.listdir(img_folder)
for name in img_list:
    img_path = os.path.join(img_folder, name)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = np.array(img)/255
    mean.append(np.mean(img))
    var.append(np.var(img))

print(np.mean(mean))
print(np.mean(var))