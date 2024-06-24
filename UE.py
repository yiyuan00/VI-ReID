import torchvision
import torch.nn as nn
import torch
from torch.nn import init
from torch.nn import functional as F
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class GEN(nn.Module):
    def __init__(self, in_feat_dim, out_img_dim, **kwargs):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.out_img_dim = out_img_dim

        self.conv0 = nn.Conv2d(self.in_feat_dim, 64, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2,bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2,bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2,bias=True)
        self.conv4 = nn.Conv2d(64, out_img_dim, kernel_size=5, stride=1, padding=2,bias=True)

        self.sa=SpatialAttention()


        self.up = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(64)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.relu(x)
        res1=self.sa(x)*x

        x = self.conv1(x)
        x = self.relu(x)
        x=x+res1
        x = self.up(x)
        res2=self.sa(x)*x

        x = self.conv2(x)
        x = self.relu(x)
        x=x+res2
        x = self.up(x)
        res3=self.sa(x)*x

        x = self.conv3(x)
        x = self.relu(x)
        x = x + res3
        x = self.up(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.up(x)

        #x = torch.tanh(x)
        return x

#可以去掉所有残差试试效果，在本人环境和配置下，能达到96.1的rank1.（regdb T2V）,sysu也可以调试

# input=torch.zeros([112,1024,24,12])
# model=GEN(1024,2)
# out=model(input)
# print(out.shape)