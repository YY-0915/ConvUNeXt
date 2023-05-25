import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class SpatialAtt(nn.Module):

    def __init__(self, kernel_size=49):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        # residual = x
        x = x * self.sa(x)
        return x

class mixSACA(nn.Module):
    def __init__(self, inp, kernel_size=7,reduction=32,layer_scale_init_value=1e-4):
        super().__init__()
        self.dim = inp // 2
        self.ca = CoordAtt(self.dim, self.dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.shuf = Channel_shuffle(self.dim)
    def forward(self, x):
        residual = x
        sp = torch.split(x, self.dim, dim=1)
        sp1 = sp[0]
        sp2 = sp[1]

        sp1 = self.ca(sp1)
        sp2 = sp2*self.sa(sp2)

        output = torch.cat((sp1, sp2),1)

        x = output + residual
        x = self.shuf(x)
        return x


class Channel_shuffle(nn.Module):
    def __init__(self,group):
        super(Channel_shuffle,self).__init__()
        self.groups=group

    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
                   channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used befogitre view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x