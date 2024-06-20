import torch
import torch.nn as nn

class RepSR_Block(nn.Module):
    def __init__(self, c):
        super(RepSR_Block, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(c, 2*c, 3, 1, 'same'),
                                     nn.BatchNorm2d(2*c),
                                     nn.Conv2d(2*c, c, 1, 1))
        self.branch2 = nn.Sequential(nn.Conv2d(c, 2*c, 3, 1, 'same'),
                                     nn.BatchNorm2d(2*c),
                                     nn.Conv2d(2*c, c, 1, 1))
        
    def forward(self, x):
        repsr_block = x + self.branch1(x) + self.branch2(x)
        return repsr_block
