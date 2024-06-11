# **该仓库是对腾讯的RepSR的非官方实现**

原文链接：[RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization](https://arxiv.org/abs/2205.05671)

官方[git地址](https://github.com/TencentARC/RepSR)

## 1.Train

代码主体框架是以[ECBSR](https://github.com/xindongzhang/ECBSR)为基础进行的实现

根据自己的数据集实际存放路径和需求，修改./configs/***.yml中的配置参数，然后运行train.py即可，训练过程中的模型和日志等内容会保存在./weights中

使用的数据集和论文中提到的一致

## 2.Test

还没训练完，暂空

## 3.注意事项

DIV2K的800张图像没有区分训练集和测试集，全拿来训练了，另外四个数据集用来测试

论文中类似frozenBN的操作不知道复现的对不对，我是在0.9epoch时，冻结了所有的BN层，不更新均值和方差，但是BN的缩放和偏移还是正常更新的

## 4.结论

暂空
