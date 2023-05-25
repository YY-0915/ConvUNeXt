'''
Author: your name
Date: 2022-02-12 18:05:36
LastEditTime: 2022-02-20 19:00:37
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /MRI-Segmentation/miccai2022project/pretask-classes/net.py
'''
import torch
import torchvision
import torch.nn as nn
from convnext import convnext_large

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.backbone = convnext_large()
        self.fc2 = nn.Linear(1000,512)
        self.Gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        down1 = self.backbone.downsample_layers[0](x)
        down2 = self.backbone.downsample_layers[1](down1)
        down3 = self.backbone.downsample_layers[2](down2)
        down4 = self.backbone.downsample_layers[3](down3)
        feature = self.backbone.stages[3](down4)
        x = self.backbone(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return (x,(down1,down2,down3,down4,feature))


print(net())