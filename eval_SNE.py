from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
import scipy.io
import os
from SNE  import  plot_tsne
import sys
from model_row import embed_net
from utils import *
from random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='sysu_M_p8_n7_lr_0.03_weight_0_mAP_best.t', type=str,
                    help='resume from checkpoint')
# parser.add_argument('--resume', '-r', default='sysu_M_p8_n7_lr_0.03_weight_3.0_mAP_bset.t', type=str,
#                     help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='best_W/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=7, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--part_num', default=12, type=int, help='number of attention map')
parser.add_argument('--w_sas', default=0.5, type=float, help='weight of Cross-Center Loss')
parser.add_argument('--w_hc', default=2.0, type=float, help='weight of Cross-Center Loss')
parser.add_argument('--train_mode', default='AST', type=str, help='weight of Cross-Center Loss')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/ubuntu/Downloads/code/dataset/sysu/'
    n_class = 395
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '/home/tan/data/data/regdb/RegDB/'
    n_class = 395
    test_mode = [2, 1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')
net = embed_net(n_class, args.part_num, arch=args.arch)
net = torch.nn.DataParallel(net).to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method == 'id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

# def fliplr(img):
#     '''flip horizontal'''
#     inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
#     img_flip = img.index_select(3, inv_idx)
#     return img_flip




def extract_query_feat(train_loader):

    print('Extracting Query Feature...')
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

            temp, feat, output, feat_globe, out_globe = net(input1, input2)
            #fg = feat_globe.chunk(2, 0)
            #fg1 = fg[0]
            #fg2 = fg[1]
            labs = torch.cat((label1, label2), 0)

            labs= labs.cpu().numpy()

            print(labs)
            plot_tsne(temp.detach().cpu().numpy(), labs, "Set_wo_ca3", fileNameDir="test")
            #plot_tsne(fg2.detach().cpu().numpy(), lab2, "Set3", fileNameDir="test")
            break



if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)



    nquery = len(query_label)



    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              1)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(1)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    extract_query_feat(trainloader)  # query特征提取










