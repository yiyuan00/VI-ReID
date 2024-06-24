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
import cv2
from utils import *
from random_erasing import RandomErasing
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='regdb_M_p8_n7_lr_0.03_weight_2.5_trial_10_acc_best.t', type=str,
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
parser.add_argument('--w_hc', default=2.5, type=float, help='weight of Cross-Center Loss')
parser.add_argument('--train_mode', default='AST', type=str, help='weight of Cross-Center Loss')

parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument(
    '--image-path',
    type=str,
    default='005.bmp',
    help='Input image path')

parser.add_argument('--aug_smooth', action='store_true',
                    help='Apply test time augmentation to smooth the CAM')
parser.add_argument(
    '--eigen_smooth',
    action='store_true',
    help='Reduce noise by taking the first principle componenet'
         'of cam_weights*activations')

parser.add_argument(
    '--methods',
    type=str,
    default='gradcam++',
    help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')



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


def reshape_transform(tensor, height=24, width=12):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result





if dataset == 'regdb':

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.methods not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    print(net.module.base_resnet.base.layer2[-1].conv3)

    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
    print(net)

    net.eval()
    target_layers = [net.module.base_resnet.base.layer3[4]]



    cam = methods[args.methods](model=net,
                               target_layers=target_layers,
                                )





    # training set

    # img=plt.imread('001.bmp')
    # img_ten=transform_test(img)
    # img_ten=torch.unsqueeze(img_ten,dim=0)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (192, 384))
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    #img_ipt=torch.cat((input_tensor,input_tensor) ,dim=0)  #net的输入是两张图片
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.


    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam4.jpg', cam_image)

    # print(img_ipt.shape)
    # feat_fc1 = net(img_ipt, img_ipt, img_ipt, img_ipt, test_mode[0])  #test mode可以改，没改，还是训练数据流
    print('finish')




