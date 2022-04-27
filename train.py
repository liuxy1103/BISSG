# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm
import torch
import argparse
import numpy as np
from solver import Solver
from model.unet2d_residual import ResidualUNet2D
from model.Unet_EGNN import unet_egnn
from utils.utils import log_args
from data.dataset import CVPPP
from torch.utils.data import DataLoader
from utils.logger import Log
import os

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
parser.add_argument("-b", "--batch_size",type=int,default=1)
parser.add_argument("-g", "--gpu_nums",type=int,default=1)
parser.add_argument("-e", "--epochs",type=int,default=1000)
parser.add_argument("-r", "--lr",type=float,default=1e-3)
parser.add_argument("-p", "--lr_patience",type=int,default=30)
parser.add_argument("-n", "--network",type=str,default="unet_egnn(3,[16,32,64,128,256],3,args)")
parser.add_argument("-t", "--loss_type",type=str,default="BCE_loss")
parser.add_argument("-d", "--data_dir",type=str,default="/braindat/lab/liuxy/superpixel/A1")
parser.add_argument("-l", "--logs_dir",type=str,default="./log")
parser.add_argument("-c", "--ckps_dir",type=str,default="./ckp")
# parser.add_argument("-s", "--resample",type=tuple,default=(1, 0.25, 0.25),help="resample rate:(z,h,w)")
parser.add_argument("-w", "--weight_rate",type=list,default=[10,1])
parser.add_argument("-x", "--resume",type=bool,default=False)
parser.add_argument("-y", "--resume_path",type=str,default="./ckp/checkpoint-epoch530.pth")
# parser.add_argument("-z", "--tolerate_shape",type=tuple,default=(192, 384, 384))

#spixel

parser.add_argument('--train_img_height', '-t_imgH', default = 448,  type=int, help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default = 448, type=int, help='img width')
parser.add_argument('--input_img_height', '-v_imgH', default = 448,  type=int, help='img height')
parser.add_argument('--input_img_width', '-v_imgW', default = 448, type=int, help='img width')

#embedding
parser.add_argument("-a", "--alpha",type=int,default=1)
parser.add_argument("-be", "--beta",type=int,default=1)
parser.add_argument("-ga", "--gama",type=int,default=0.001)
#EGNN

#loss rate
parser.add_argument("-ls", "--loss_spixel",type=int,default=5) # for affinity
parser.add_argument("-le", "--loss_embedding",type=int,default=1)
parser.add_argument("-lb", "--loss_binary",type=int,default=100)
parser.add_argument("-lg", "--loss_gnn",type=int,default=10)
args = parser.parse_args()

SEED = 123

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
log = Log()
if __name__ == '__main__':
    #DDP
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')


    gpus = args.gpu_nums
    model = eval(args.network)
    #load pretrained model
    model_path = os.path.join(r'./checkpoint-epoch195.pth')
    net_dict = model.state_dict()
    pretrain = torch.load(model_path)
    pretrain_dict = {'unet.'+k: v for k, v in pretrain['state_dict'].items() if 'unet.'+k in net_dict.keys()}

    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    for k, v in model.named_parameters():
        # print(k)
        if 'unet.inconv' in k or 'unet.down' in k or '_spix' in k:
            v.requires_grad = False
    print('following parameters are training:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    criterion = args.loss_type
    metric = "dc_score"
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr


    trainset = CVPPP(dir=args.data_dir,mode="train",size=args.train_img_height)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    valset = CVPPP(dir=args.data_dir,mode="validation",size=args.input_img_height)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=False,sampler=val_sampler)

    logs_dir = args.logs_dir
    patience = args.lr_patience
    checkpoint_dir = args.ckps_dir
    # scale = args.resample
    weight = args.weight_rate
    resume = args.resume
    resume_path = args.resume_path
    # tolerate_shape = args.tolerate_shape

    #embedding
    alpha = args.alpha


    beta = args.beta
    gama = args.gama
    #spixel
    le = args.loss_embedding
    ls = args.loss_spixel
    lb = args.loss_binary

    log_args(args, log)

    solver = Solver(gpus=gpus,model=model,criterion=criterion,metric=metric,batch_size=batch_size,
                    epochs=epochs,lr=lr,trainset=trainset,valset=valset,train_loader=train_loader,
                    val_loader=val_loader,logs_dir=logs_dir,patience=patience,
                    checkpoint_dir=checkpoint_dir,weight=weight, resume=resume,resume_path=resume_path,
                    log=log,args = args)

    solver.train()
