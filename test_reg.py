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
import matplotlib.pyplot as plt
import sys
from model import embed_net
from utils import *
from random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='regdb_M_p8_n7_lr_0.03_weight_2.5_trial_10_mAP_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='best_W/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=2, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
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
if dataset == 'regdb':
    data_path = '/home/ubuntu/Downloads/code/dataset/RegDB/'
    n_class = 206
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

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_gall_feat(gall_loader):  # rgb
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, args.part_num * 512))
    gall_feat_fc = np.zeros((ngall, args.part_num * 512))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            input2 = Variable(fliplr(input).cuda())
            feat_fc1 = net(input1, input1, input1, input1,test_mode[0])
            feat_fc2 = net(input2, input2, input1, input1,test_mode[0])
            feat_pool = feat_fc1 + feat_fc2
            feat_fc = feat_fc1 + feat_fc2
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, args.part_num * 512))
    query_feat_fc = np.zeros((nquery, args.part_num * 512))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            input2 = Variable(fliplr(input).cuda())
            print(input1.shape)
            feat_fc1 = net(input1, input1, input1, input1,test_mode[1])
            feat_fc2 = net(input2, input2, input1, input1,test_mode[1])
            feat_pool = feat_fc1 + feat_fc2
            feat_fc = feat_fc1 + feat_fc2
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc




if dataset == 'regdb':

    for trial in range(10):
        test_trial = trial + 1
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        net.eval()
        img=plt.imread('001.bmp')
        img_ten=transform_test(img)
        img_ten=torch.unsqueeze(img_ten,dim=0)
        img_ipt=torch.cat((img_ten,img_ten) ,dim=0)
        feat_fc1 = net(img_ipt, img_ipt, img_ipt, img_ipt, test_mode[0])  #test mode可以改
        print('finish')




