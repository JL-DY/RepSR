# **该仓库是对腾讯的RepSR的非官方实现**

原文链接：[RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization](https://arxiv.org/abs/2205.05671)

官方[git地址](https://github.com/TencentARC/RepSR)

## 1.Train

代码主体框架是以[ECBSR](https://github.com/xindongzhang/ECBSR)为基础进行的实现

根据自己的数据集实际存放路径和需求，修改./configs/***.yml中的配置参数，然后运行train.py即可，训练过程中的模型和日志等内容会保存在./weights中

使用的数据集和论文中提到的一致

## 2.Test

运行test.py即可，测试代码很简单，有需要修改的地方自己看一下代码就知道了

## 3.注意事项

DIV2K的800张图像没有区分训练集和测试集，全拿来训练了，另外四个数据集用来测试

论文中类似frozenBN的操作不知道复现的对不对，我是在0.9epoch时，冻结了所有的BN层，不更新均值和方差，但是BN的缩放和偏移还是正常更新的

训练和推理本人是采用的单通道进行，将原始的RGB图像转为YCrCB后，仅使用Y通道进行训练和推理，loss为L1-loss

patch_size=256，epoch=1000，没有完全按照论文中的来

## 4.结论

目前来看，在我个人设备上训练出来的模型是有效的，各项指标差的也不算多，在我个人的测试集上目前也没有看到BN artifacts现象产生

但是这个算法和ECBSR相同，由于配对图像就是单纯的下采样得到的，因此不能对本身含有噪声的低分辨率图像进行超分(但是低分辨率图像很多时候又有噪声)，会产生噪声放大的现象

后续添加了二阶段降质算法进行下采样后，能对低分辨率图像的噪声进行明显的改善，[二阶段降质算法的官方来源](https://github.com/XPixelGroup/BasicSR)

由于个人不太习惯用BasicSR这套框架，因此将二阶段降质算法单独拧出来并合并到RepSR中。拧出来的二阶段降质算法和使用方法在另一个[仓库](https://github.com/JL-DY/Degradation)中可以了解并查看



