'''
Author: weifeng liu                                   
Date: 2022-02-21 01:40:05
LastEditTime: 2022-07-02 15:51:52
LastEditors: liuweifeng 1683723560@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /MRI-Segmentation/miccai2022project/segment_model/convnext_Decoder_tranconv.py
'''
from net import net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
from convnext import convnext_large
from net import net

class DoubleConv(nn.Module):
    """
    unet的编码器中，每一个level都会有两层卷积和Relu
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)

class  upsample(nn.Module):
    """
    upsample,  使用双线性插值或者反卷积
    """
    def __init__(self, in_channels,out_channels,bilinear = True):
        super(upsample,self).__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                        align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2,
                                               kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        """
        :param x1: decoder feature
        :param x2: encoder feature
        :return:
        """
        x1 = self.upsample(x1)

        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])

        #将x1与x2的特征图对齐后concat
        x1 = F.pad(x1, [diff_x//2,diff_x - diff_x//2,
                   diff_y//2,diff_y - diff_y // 2])
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class output_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(output_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,bilinear = False):
        """
        :param n_classes: segmentation classes
        :param bilinear: upsample tpye
        """
        super(UNET,self).__init__()
        # self.n_classes = n_classes
        self.bilinear = bilinear
        
        # device = torch.device("cpu")
        # self.encoder = net()
        # self.encoder = nn.DataParallel(self.encoder)
        # self.encoder.to(device)
        # self.encoder.load_state_dict(torch.load('/home/lwf/Project/MRI-Segmentation/miccai2022project/pretask-classes/best_model/best model epoch196:net.pth'))
        self.backbone = convnext_large(pretrained=True)
        

        self.upsample1 = upsample(1536,768,self.bilinear)
        self.upsample2 = upsample(768,384,self.bilinear)
        self.upsample3 = upsample(384,192,self.bilinear)
        # self.upsample4 = upsample(384,96,bilinear)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
        #                                 align_corners=True)
        self.upsample = nn.ConvTranspose2d(192, 192,
                                               kernel_size=2,stride=2)
        self.outconv = output_conv(192,1)
    def forward(self,x):
        down1 = self.backbone.downsample_layers[0](x)
        down1 = self.backbone.stages[0](down1)
        # print("down1:",down1.shape)
        # [,192,128,128]
        down2 = self.backbone.downsample_layers[1](down1)
        down2 = self.backbone.stages[1](down2)
        # print("down2:",down2.shape)
        # [,384,64,64]
        down3 = self.backbone.downsample_layers[2](down2)
        down3 = self.backbone.stages[2](down3)
        # print("down3:",down3.shape)
        # [,768,32,32]
        down4 = self.backbone.downsample_layers[3](down3)
        down4 = self.backbone.stages[3](down4)
        # print("down4:",down4.shape)
        # [,1536,16,16]
        # feature = self.backbone.stages[3](down4)
        # print("backbone feature:",feature.shape)
        # [,1536,16,16]
        # x = self.backbone(x)

        x = self.upsample1(down4,down3)
        # print("unsample1:",x.shape)
        x = self.upsample2(x,down2)
        # print("unsample2:",x.shape)
        x = self.upsample3(x,down1)
        # print("unsample3:",x.shape)
        # x = self.upsample2(x,down1)
        x = self.upsample(x)
        x = self.upsample(x)
        res = self.outconv(x)
        # print("res:",res.shape)
        return res

net = UNET()
print(net)
# x = torch.Tensor(4,3,512,512)
# y = net(x)