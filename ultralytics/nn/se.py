import numpy as np
import torch
from torch import nn
from torch.nn import init
 
 #详细改进流程和操作，请关注B站博主：Ai学术叫叫兽 er,畅享一对一指点迷津，已指导无数家人拿下学术硕果！！！
 
class SEAttention(nn.Module):
 
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )#详细改进流程和操作，请关注B站博主：Ai学术叫叫兽 er,畅享一对一指点迷津，已指导无数家人拿下学术硕果！！！
 
 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)#详细改进流程和操作，请关注B站博主：Ai学术叫叫兽 er,畅享一对一指点迷津，已指导无数家人拿下学术硕果！！！
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 