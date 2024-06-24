import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
import numpy as np
import math
import matplotlib.pyplot as plt
from PM import PM
from FSIM import FSIM
from SFC import SFC_G2
from UE import GEN
from torchvision import transforms
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
        

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


class att_resnet(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(att_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.SA = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        self.classifier = ClassBlock(2048, class_num) #初始化了 BNneck  注意这里的classifier和后面我们定义的 ”classifier类” 区分
        self.PM=PM(class_num)

        self.conv1 = nn.Conv2d(2048, 2, 1, 1,0)
        self.UE1=GEN(1024,2)
        self.UE2=GEN(1024,2)
        self.toPIL=transforms.ToPILImage()

        #self.bn=nn.BatchNorm2d(2048)

    def forward(self, x):
        f = self.base.layer4(x)
        #mask=self.conv1(f)
        f1,f2=torch.chunk(f,2,dim=1)

        mask=self.UE1(f1)
        maskc=self.UE2(f2)

        mask=F.softmax(mask,dim=1)  #预测背景和前景 ，然后softmax,表示概率分布（255-G_map，得到背景GT）
        mask=mask[:,0:1:,:]
        maskc = F.softmax(maskc, dim=1)  # 预测背景和前景 ，然后softmax,表示概率分布（255-G_map，得到背景GT）
        maskc = maskc[:, 0:1:, :]

        ms=torch.squeeze(maskc)
        ms=ms[0]
        pic=self.toPIL(ms)
        pic.save('msc.jpg')

        Wight_P=F.interpolate(mask,size=(24,12),mode='bilinear',align_corners=False)
        #f=Wight_P*f
#ssa
        #print(mask.shape)
        #x = torch.mul(x, self.sigmoid(torch.mean(f, dim=1, keepdim=True)))

        #f = torch.squeeze(self.base.avgpool(f)) #B,C,H,W  变为B，C,1,1 ->之后转变为 bs，c
        out,feat=self.PM(f)
        #out, feat = self.classifier(f)
        return x, out, mask,maskc
        
        
class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)

        x = self.base.layer2(x)

        x = self.base.layer3(x)
        return x


class ClassBlock(nn.Module):  #BNneck
    def __init__(self, input_dim, class_num, droprate=0.5, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block) #就是sequential把模块组合起来
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)#同上
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier #代表两种结构，addblock是从inputdim->num_bottleneck.  classifier是从num_bottleneck到 class_num(类别数)

    def forward(self, x):
        x = self.add_block(x)
        f = x
        x = self.classifier(x)
        return x, f  #第一个用于分类，第二个用于triplet loss


class classifier(nn.Module):   #PCB
    def __init__(self, num_part, class_num):
        super(classifier, self).__init__()
        input_dim = 1024
        self.part = num_part #默认12
        self.l2norm = Normalize(2)
        for i in range(num_part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num)) #self.classifier_i=ClassBlock(input_dim, class_num)//BNneck

    def forward(self, x, feat_all, out_all):
        start_point = len(feat_all)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            cls_part = getattr(self, name)
            #print(x.shape) [112, 1024, 12, 1]
            out_all[i + start_point], feat_all[i + start_point] = cls_part(torch.squeeze(x[:, :, i]))
            feat_all[i + start_point] = self.l2norm(feat_all[i + start_point])

        return feat_all, out_all


class embed_net(nn.Module):
    def __init__(self, class_num, part, arch='resnet50'):
        super(embed_net, self).__init__()

        self.part = part
        self.base_resnet = base_resnet(arch=arch)
        self.att_v = att_resnet(class_num)
        self.att_n = att_resnet(class_num)
        #self.att_v_g = att_resnet(class_num)
        #self.att_n_g = att_resnet(class_num)
        self.classifier = classifier(part, class_num)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1)) #这里pooling 到（12，1）

        #self.fsim=FSIM()
        self.sfc1=SFC_G2()
        #self.sfc2=SFC_G2()
        self.conv1=nn.Conv2d(1024,1024,3,1)
        self.conv2=nn.Conv2d(1024,1024,3,1)
        self.fw1=torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.fw2=torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.fw1.data.fill_(0.5)
        self.fw2.data.fill_(0.5)


    def forward(self, x1, x2=0, modal=0):
        if modal == 0:  #模式是训练
            # x1 = torch.cat((x1, x1), dim=0)  # net的输入是两张图片
            # x2=x1
            #x1, x2, x1_g, x2_g的形状     torch.Size([56, 3, 384, 192])

            #x1=self.fsim(x1) #传两个参进去
            #x2=self.fsim(x2)

            #x1=x1*(1-self.fw1)+x1_g*self.fw1
            #x2=x2*(1-self.fw2)+x2_g*self.fw2


            x = torch.cat((x1, x2), 0)  #从bs维度cat的，不需要调整维度
            x = self.base_resnet(x)
            x1, x2 = torch.chunk(x, 2, 0)#在bs维度做分离

            x1, out_v, mask_v,maskc_v = self.att_v(x1)# x1是门控后的特征， out_v是用于分类的，feat_v是用于算triplet loss的
            x2, out_n, mask_n,maskc_n = self.att_n(x2)

            #print(feat_v.shape) # 56, 206

            # x1, x2, x1_g, x2_g的形状     torch.Size([32, 1024, 24, 12])
            #x1=self.sfc1(x1,x2_g)
            #x2=self.sfc2(x2,x1_g)

            #x1=self.conv1(x1)
            #x2=self.conv2(x2)
            #x=self.sfc1(x1,x2)
            x = torch.cat((x1, x2), 0)

            feat_globe = torch.cat((mask_v, mask_n), 0)#没用上,现代表mask特征
            maskc_globe = torch.cat((maskc_v, maskc_n), 0)#没用上,现代表mask特征

            out_globe = torch.cat((out_v, out_n), 0)#这里还是保持，就只用这两个算ID loss
        elif modal == 1:#模式是推理
            #x1 = self.fsim(x1)
            x = self.base_resnet(x1)
            x, _, _,_ = self.att_v(x)
        elif modal == 2:#同上
            #x2 = self.fsim(x2)
            x = self.base_resnet(x2)
            x, _, _,_= self.att_n(x)

        x = self.avgpool(x) #这里的x接上，x = torch.cat((x1, x2), 0) 也就是两个走过specific weight后的特征。之后在pooling到（12，1）
        feat = {}
        out = {}
        feat, out = self.classifier(x, feat, out)# 走classifier类，
        if self.training:
            return feat, out, feat_globe, out_globe,maskc_globe
        else:
            for i in range(self.part):
                if i == 0:
                    featf = feat[i]
                else:
                    featf = torch.cat((featf, feat[i]), 1)
            return featf