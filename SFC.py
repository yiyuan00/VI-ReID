# @Author: CZH
# coding=utf-8
# @FileName:fusion2.py
# @Time:2024/1/2
# @Author: CZH
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import BatchNorm2d
# Context and Spatial Feature Calibration for Real-Time Semantic Segmentation

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SFC_G2(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SFC_G2, self).__init__()
        self.conv1024to128 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, bias=False)
        self.conv1024to256 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, bias=False)

        self.conv_8 = ConvBNReLU(128, 128, 3, 1, 1)
        #self.cp1x1 = nn.Conv2d(128, 32, 1, bias=False)

        self.conv_32 = ConvBNReLU(128, 128, 3, 1, 1)

        #self.sp1x1 = nn.Conv2d(128, 32, 1, bias=False)

        self.groups = 2

        # print('groups', self.groups)

        self.conv_offset = nn.Sequential(
            ConvBNReLU(256, 256, 1, 1, 0),
            nn.Conv2d(256, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False))

        self.keras_init_weight()

        self.conv_offset[1].weight.data.zero_()

        self.conv128to1024 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.conv128to1024_2 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, bias=False) #考虑走同一个卷积升高维度

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, cp, sp):
        cp1 = self.conv1024to128(cp)
        sp1 = self.conv1024to256(sp)

        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  128
        #sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)
        # x_8
        cp = self.conv_8(cp)

        ## 将cp1x1/sp1x1和conv_offset 合并，将导致更低的计算参数
        #cp1x1 = self.cp1x1(cp)
        #sp1x1 = self.sp1x1(sp)

        conv_results = self.conv_offset(torch.cat([cp1, sp1], 1))

        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0:self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l, align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h, align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        #sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]
        cp=cp * att[:, 1:2, :, :]
        sp=sp * att[:, 0:1, :, :]
        #cp=self.conv128to1024(cp)
        #sp=self.conv128to1024_2(sp)
        x=torch.cat((cp ,sp ),0)

        return x


if __name__ == '__main__':
    input1 = torch.randn(7, 1024, 24, 12)
    input2 = torch.randn(7, 1024, 24, 12)
    model = SFC_G2()
    output = model(input1, input2)
    print(output.shape)