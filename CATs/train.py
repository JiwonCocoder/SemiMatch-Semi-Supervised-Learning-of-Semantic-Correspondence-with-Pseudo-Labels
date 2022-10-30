r'''
    modified training script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
import pickle
import argparse
import os
import pickle
import random
import time
from os import path as osp
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader

from models.cats import CATs
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from data import download
import pdb

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default='auto',
                        help='automatically generate directory depending on 182 line')

    parser.add_argument('--time_stamp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')     

    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    #lr#
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[70, 80, 90]')
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--use_warmUp', type=boolean_string, nargs='?', const=True, default=True)

    parser.add_argument('--feature-size', type=int, default=16)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--hyperpixel', type=str, default='[2,17,21,22,25,26,28]')
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=True)
    
    #data_augmentation#
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--aug_mode', type=str, default='afftps', choices=['aff', 'tps', 'afftps'])
    #[geometric_default] affine_default: 0.15, TPS_Default: 0.4
    parser.add_argument('--aug_aff_scaling', type=float, default=0.25)
    parser.add_argument('--aug_tps_scaling', type=float, default=0.4)
    #[photomertric_default] trg_weak = 0 trg_strong 0.2
    parser.add_argument('--aug_photo_weak', type=float, default=0)
    parser.add_argument('--aug_photo_strong', type=float, default=0.2)

    #fixmatch#
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--semi_softmax_corr_temp', type=float, default=0.1)
    parser.add_argument('--semi_contrastive_temp', type=float, default=0.1)
    parser.add_argument('--loss_mode', type=str, default='contrastive', choices=['contrastive', 'EPE'])
    parser.add_argument('--inner_bbox_loss', action='store_true')
    parser.add_argument('--warmup_epoch', type=int, default=-1)
    parser.add_argument('--refined_corr_filtering', type=str, default='mutual', choices=['mutual', 'dual_softmax', 'soft_argmax'])
    parser.add_argument('--interpolation_mode', type=str, default='bilinear', choices=['bilinear', 'nearest', 'bicubic'])
    parser.add_argument('--use_fbcheck_mask', type=boolean_string, nargs='?', const=True, default=True)

    # transformation
    parser.add_argument('--interpolate_index', action='store_true')
    parser.add_argument('--use_self_loss', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--use_me_max_loss', type=boolean_string, nargs='?', const=True, default=True)


    #0.01, 0.05, 0.1, 1
    parser.add_argument('--alpha_1', type=float, default=0.1)
    parser.add_argument('--alpha_2', type=float, default=0.5)
    parser.add_argument('--semi_lambda', type=float, default=0.25)
    parser.add_argument('--self_lambda', type=float, default=0.5)

    #GPU
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')

    # Seed
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('Available devices', torch.cuda.device_count())
    print('Current cuda device', torch.cuda.current_device())
    torch.cuda.set_device(args.gpu_id)
    print('Changed cuda device', torch.cuda.current_device())
    device = torch.cuda.current_device()
    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size,
                                          args.aug_mode, args.aug_aff_scaling, args.aug_tps_scaling, args.aug_photo_weak, args.aug_photo_strong)
    val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val', args.augmentation, args.feature_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataloader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=True)
    val_dataloader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    # Model
    if args.freeze:
        print('Backbone frozen!')
    model = CATs(
        feature_size=args.feature_size, feature_proj_dim=args.feature_proj_dim,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        hyperpixel_ids=parse_list(args.hyperpixel), freeze=args.freeze, args=args)
    param_model = [param for name, param in model.named_parameters() if 'feature_extraction' not in name]
    param_backbone = [param for name, param in model.named_parameters() if 'feature_extraction' in name]

    # Optimizer
    optimizer = optim.AdamW([{'params': param_model, 'lr': args.lr}, {'params': param_backbone, 'lr': args.lr_backbone}], 
                weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = \
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, verbose=True)\
        if args.scheduler == 'cosine' else\
        lr_scheduler.MultiStepLR(optimizer, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True)

    if args.pretrained:
        #load args.pkl
        # load_model = pickle.load(open('./pretrained/args.pkl','rb'))
        # print(load_model)

        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                            filename=args.pretrained)

        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))
        cur_snapshot = args.name_exp
    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = 0
        start_epoch = 0

    if args.name_exp =='auto':
        cur_snapshot = f'{args.benchmark}_filtering_{args.refined_corr_filtering}_corr_{args.semi_softmax_corr_temp}_CT_{args.semi_contrastive_temp}_loss_{args.loss_mode}_aug_{args.aug_mode}_alpha_{args.alpha_1}_{args.alpha_2}_scale_{args.aug_tps_scaling}_interpolate_{args.interpolation_mode}_use_fb_{args.use_fbcheck_mask}_use_self_{args.use_self_loss}_{args.time_stamp}'
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))
    # with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
    #     pickle.dump(args, f)
    args.cur_snapshot = os.path.join(args.snapshots, cur_snapshot)
    with open(os.path.join(args.cur_snapshot, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, allow_unicode=True, default_flow_style=False)
    
    #Create save_path : snapshot/[defined_path]
    args.cur_snapshot = os.path.join(args.snapshots, cur_snapshot)
    save_path = args.cur_snapshot
    
    # create summary writer
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    # create img path
    args.save_img_path = os.path.join(save_path, 'img')
    if not osp.isdir(args.save_img_path):
        os.makedirs(args.save_img_path)

    model = model.to(device)

    train_started = time.time()
    result_file = '{}_results.txt'.format(args.time_stamp)

    for epoch in range(start_epoch, args.epochs):
        train_loss = optimize.train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                args,
                                save_path)
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)
        scheduler.step()

        val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                       val_dataloader,
                                                       device,
                                                       epoch=epoch)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('mean PCK is {}'.format(val_mean_pck))
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        test_writer.add_scalar('mean PCK', val_mean_pck, epoch)
        test_writer.add_scalar('val loss', val_loss_grid, epoch)

        train_loss_grid, train_mean_pck = optimize.validate_epoch(model,
                                                       train_dataloader,
                                                       device,
                                                       epoch=epoch)
        print(colored('==> ', 'blue') + 'Train average grid loss :',
              train_loss_grid)
        print('mean PCK is {}'.format(train_mean_pck))
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        test_writer.add_scalar('mean PCK', train_mean_pck, epoch)
        test_writer.add_scalar('Train loss', train_loss_grid, epoch)

        is_best = val_mean_pck > best_val
        best_val = max(val_mean_pck, best_val)
        with open(os.path.join(save_path, result_file),'a+') as file:
            file.write(f'{val_mean_pck, best_val}\n')
        if epoch != args.warmup_epoch:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_loss': best_val},
                            is_best, save_path, None)
            if epoch % 10 == 0 :
                save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_loss': best_val},
                            is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
        else:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_loss': best_val},
                            is_best, save_path, 'epoch_{}_warmUp.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
