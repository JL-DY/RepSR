import torch.nn as nn
import torch.nn.functional as F


class RepSR_Plain(nn.Module):
    def __init__(self, m, c, scale, colors, device):
        super(RepSR_Plain, self).__init__()
        self.device = device
        self.m_repsr_block = m
        self.c_channel = c
        self.scale = scale
        self.colors = colors

        M_block = []
        M_block += [nn.Conv2d(self.colors, self.c_channel, 3, 1, 'same')]
        M_block += [nn.PReLU(self.c_channel)]
        for i in range(self.m_repsr_block):
            M_block += [nn.Conv2d(self.c_channel, self.c_channel, 3, 1, 'same')]
            M_block += [nn.PReLU(self.c_channel)]
        M_block += [nn.Conv2d(self.c_channel, self.colors*self.scale*self.scale, 3, 1, 'same')]
        self.M_block = nn.Sequential(*M_block)

        self.pixelshuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        resize = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        x = self.M_block(x)
        x = self.pixelshuffle(x)
        y = x + resize
        return y