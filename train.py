# from lib.ncnet import NCNet
import os
from os.path import exists, join, basename
from collections import OrderedDict
import random
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset
from lib.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from lib.vis import plot_NormMap2D_warped_Img
from lib.dataloader import DataLoader  # modified dataloader
from lib.model import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool
from lib.gen_transform import *
import argparse
from data import download
import argparse
import os
import sys
import pickle
import random
import time
from os import path as osp
import yaml
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
import pdb
from tqdm import tqdm
from lib.vis import plot_NormMap2D_warped_Img
from lib.utils import over_write_args_from_file
import math
from semimatch.semimatch import SemiMatch
from semimatch.evaluation import Evaluator

# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

print("ImMatchNet training script")

# Argument parsing
parser = argparse.ArgumentParser(description="Compute PF Pascal matches")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--image_size", type=int, default=400)
parser.add_argument(
    "--dataset_image_path",
    type=str,
    default="datasets/pf-pascal/",
    help="path to PF Pascal dataset",
)
parser.add_argument(
    "--dataset_csv_path",
    type=str,
    default="datasets/pf-pascal/image_pairs/",
    help="path to PF Pascal training csv",
)
parser.add_argument(
    "--num_epochs", type=int, default=20, help="number of training epochs"
)
parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument(
    "--ncons_kernel_sizes",
    nargs="+",
    type=int,
    default=[5, 5, 5],
    help="kernels sizes in neigh. cons.",
)
parser.add_argument(
    "--ncons_channels",
    nargs="+",
    type=int,
    default=[16, 16, 1],
    help="channels in neigh. cons",
)
parser.add_argument(
    "--result_model_fn",
    type=str,
    default="checkpoint_adam",
    help="trained model filename",
)
parser.add_argument(
    "--result-model-dir",
    type=str,
    default="trained_models",
    help="path to trained models folder",
)
parser.add_argument(
    "--fe_finetune_params", type=int, default=0, help="number of layers to finetune"
)

parser.add_argument('--benchmark', type=str, default='spair', choices=['pfpascal', 'spair'])
parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--train_fe', type=boolean_string, nargs='?', const=True, default=False)

# parser.add_argument('--aug_mode', type=str, default='afftps', choices=['aff', 'tps', 'afftps'])
# parser.add_argument('--aug_aff_scaling', type=float, default=0.25)
# parser.add_argument('--aug_tps_scaling', type=float, default=0.4)
# parser.add_argument('--aug_photo_strong', type=float, default=0.2)
# parser.add_argument('--aug_photo_source', type=float, default=0.2)
# parser.add_argument('--aug_photo_weak', type=float, default=0.2)

# parser.add_argument('--additional_weak', type=boolean_string, nargs='?', const=True, default=True)
parser.add_argument('--seed', type=int, default=2, help='Pseudo-RNG seed')
parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
parser.add_argument('--step', type=str, default='[150, 220]')
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=500,
                    help='number of training epochs')

#model_save 
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--snapshots', type=str, default='./snapshots_final')
parser.add_argument('--name_exp', type=str,
                    default='auto',
                    help='automatically generate directory depending on 182 line')
parser.add_argument('--time_stamp', type=str,
                    default=time.strftime('%Y_%m_%d_%H_%M'),
                    help='name of the experiment to save')   

#supplementary#
#data_augmentation#
parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
parser.add_argument('--aug_mode', type=str, default='afftps', choices=['aff', 'tps', 'afftps'])
#[geometric_default] affine_default: 0.15, TPS_Default: 0.4
parser.add_argument('--aug_aff_scaling', type=float, default=0.25)
parser.add_argument('--aug_tps_scaling', type=float, default=0.4)
#[photomertric_default] trg_weak = 0 trg_strong 0.2
parser.add_argument('--aug_photo_source', type=float, default=0.2)
parser.add_argument('--aug_photo_weak', type=float, default=0.2)
parser.add_argument('--aug_photo_strong', type=float, default=0.2)
parser.add_argument('--aug_mixup', type=float, default=0)
#KeyOut, KeyMix#
parser.add_argument('--keyout', type=float, default=0)
parser.add_argument('--keymix', type=float, default=0)
parser.add_argument('--strong_sup_loss', action='store_true')
parser.add_argument('--additional_weak', type=boolean_string, nargs='?', const=True, default=False)

#uncertainty uncertainty_lamda

parser.add_argument('--uncertainty_lamda', type=float, default=1.5)
parser.add_argument('--use_uncertainty', type=boolean_string, nargs='?', const=True, default=True)

# use class_aware_sup
parser.add_argument('--use_class_aware_sup', action='store_true')

#fixmatch#
parser.add_argument('--p_cutoff', type=float, default=0.50)
parser.add_argument('--semi_softmax_corr_temp', type=float, default=0.05)
parser.add_argument('--semi_contrastive_temp', type=float, default=0.1)
parser.add_argument('--loss_mode', type=str, default='contrastive', choices=['contrastive', 'EPE'])
parser.add_argument('--loss_sup', type=str, default='semi', choices=['ncnet', 'semi', 'sup'])

parser.add_argument('--inner_bbox_loss', action='store_true')
parser.add_argument('--warmup_epoch', type=int, default=20)
parser.add_argument('--refined_corr_filtering', type=str, default='soft_argmax', choices=['mutual', 'dual_softmax', 'soft_argmax'])
parser.add_argument('--interpolation_mode', type=str, default='nearest', choices=['bilinear', 'nearest', 'bicubic'])
parser.add_argument('--use_fbcheck_mask', type=boolean_string, nargs='?', const=True, default=True)
parser.add_argument('--contrastive_gt_mask', action='store_true')

# transformation
parser.add_argument('--interpolate_index', action='store_true')
parser.add_argument('--use_self_loss', type=boolean_string, nargs='?', const=True, default=False)
parser.add_argument('--use_me_max_loss', type=boolean_string, nargs='?', const=True, default=False)


#0.01, 0.05, 0.1, 1
parser.add_argument('--alpha_1', type=float, default=0.1)
parser.add_argument('--alpha_2', type=float, default=0.5)
parser.add_argument('--semi_lambda', type=float, default=0.5)
parser.add_argument('--self_lambda', type=float, default=0.5)

#GPU
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--amp', action='store_true')

parser.add_argument('--sparse_exp', action='store_true')
parser.add_argument('--dynamic_unsup', action='store_true')
parser.add_argument('--keyout_size', nargs=2, type=float, default=(0.03, 0.1))

parser.add_argument('--zero_sup', action='store_true')
parser.add_argument('--synthetic_box', action='store_true')

parser.add_argument('--c', type=str, default='')

args = parser.parse_args()
over_write_args_from_file(args, args.c)
print(args)

#device
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
torch.cuda.set_device(args.gpu_id)
print('Changed cuda device', torch.cuda.current_device())
device = torch.cuda.current_device()



#dataset
train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size,
                                        args.aug_mode, args.aug_aff_scaling, args.aug_tps_scaling, args.aug_photo_weak, args.aug_photo_strong, 
                                        aug_photo_source=args.aug_photo_source, additional_weak=args.additional_weak)
val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', args.augmentation, args.feature_size)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# Initialize Evaluator
Evaluator.initialize(args.benchmark, args.alpha)

train_dataloader = DataLoader(train_dataset,
    batch_size=args.batch_size,
    num_workers=args.n_threads,
    shuffle=True)
val_dataloader = DataLoader(val_dataset,
    batch_size=args.batch_size,
    num_workers=args.n_threads,
    shuffle=False)

# Create model
print("Creating CNN model...")
model = ImMatchNet(
    use_cuda=use_cuda,
    checkpoint=args.checkpoint,
    ncons_kernel_sizes=args.ncons_kernel_sizes,
    ncons_channels=args.ncons_channels,
    args = args,
    train_fe = args.train_fe
)

# Set which parts of the model to train
if args.fe_finetune_params > 0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i + 1)].parameters():
            p.requires_grad = True

print("Trainable parameters:")
for i, p in enumerate(filter(lambda p: p.requires_grad, model.parameters())):
    print(str(i + 1) + ": " + str(p.shape))

# Optimizer
print("using Adam optimizer")
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)

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
    # cur_snapshot = f'{args.benchmark}_filtering_{args.refined_corr_filtering}_corr_{args.semi_softmax_corr_temp}_CT_{args.semi_contrastive_temp}_loss_{args.loss_mode}_aug_{args.aug_mode}_alpha_{args.alpha_1}_{args.alpha_2}_scale_{args.aug_tps_scaling}_interpolate_{args.interpolation_mode}_use_fb_{args.use_fbcheck_mask}_use_self_{args.use_self_loss}_{args.time_stamp}'
    cur_snapshot = f'new_{args.benchmark}_{args.loss_sup}_{args.train_fe}_{args.lr}_{args.time_stamp}'
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

net = model.to(device)

train_started = time.time()
result_file = '{}_results.txt'.format(args.time_stamp)
semimatch = SemiMatch(net, device, args)
normalize = lambda x: torch.nn.functional.softmax(x, 1)
def cutout_aware(image, kpoint, mask_size_min=3, mask_size_max=15, p=0.2, cutout_inside=True, mask_color=(0, 0, 0),
                    cut_n=10, batch_size=2, bbox_trg=None, n_pts=None, cutout_size_min=0.03, cutout_size_max=0.1):
        # mask_size = torch.randint(low=mask_size_min, high=mask_size_max, size=(1,))

        # mask_size_half = mask_size // 2
        # offset = 1 if mask_size % 2 == 0 else 0

        def _cutout(image, x, y, mask_size, mask_size_half):
            # image = np.asarray(image).copy()

            if torch.rand(1) > p:
                return image

            _, h, w = image.shape

            # if cutout_inside:
            #     cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            #     cymin, cymax = mask_size_half, h + offset - mask_size_half
            # else:
            #     cxmin, cxmax = 0, w + offset
            #     cymin, cymax = 0, h + offset

            cx = x  # torch.randint(low=cxmin[0], high=cxmax[0], size=(1,))
            cy = y  # torch.randint(low=cymin[0], high=cymax[0], size=(1,))
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            image[:, ymin:ymax, xmin:xmax] = 0  # mask_color
            return image

        image_new = []
        for b in range(batch_size):
            bbox = bbox_trg[b]
            max_mask_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            buffer = image[b]
            kpoint_n = len(kpoint)
            # non_zero_index = torch.nonzero(kpoint[b][0])
            non_zero_index = torch.arange(n_pts[b])
            # cut_n_rand = torch.randint(1, cut_n, size=(1,))

            for n in range(len(non_zero_index)):
                if math.isclose(cutout_size_min, cutout_size_max):
                    mask_size = torch.ones(1,) * max_mask_size * cutout_size_min
                else:                   
                    mask_size = torch.randint(low=int(max_mask_size * cutout_size_min), 
                                            high=int(max_mask_size * cutout_size_max), size=(1,))
                # mask_size = int(max_mask_size * 0.1)
                mask_size_half = mask_size // 2
                buffer = _cutout(buffer, int(kpoint[b][0][n]), int(kpoint[b][1][n]),
                                mask_size, mask_size_half)
            image_new.append(buffer)
        image_new = torch.stack(image_new, dim=0)
        return image_new
def process_sup(net, trg_batch, src_batch):
    corr4d = net(trg_batch, src_batch, semi=False)
    bz = corr4d.size(0)
    fz = corr4d.size(2)
    nc_B_Avec = corr4d.view(bz, fz*fz, fz, fz)
    nc_A_Bvec = corr4d.view(bz, fz, fz, fz*fz).permute(0,3,1,2)
    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)        

    # compute matching scores
    scores_B, _ = torch.max(nc_B_Avec, dim=1)
    scores_A, _ = torch.max(nc_A_Bvec, dim=1)
    score_pos = torch.mean(scores_A + scores_B) / 2
    # negative
    src_batch = src_batch[np.roll(np.arange(bz), -1), :]  # roll
    corr4d = model(trg_batch, src_batch, semi = False)
    nc_B_Avec = corr4d.view(bz, fz*fz, fz, fz)
    nc_A_Bvec = corr4d.view(bz, fz, fz, fz*fz).permute(0,3,1,2)
    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)        
    
    # compute matching scores
    scores_B, _ = torch.max(nc_B_Avec, dim=1)
    scores_A, _ = torch.max(nc_A_Bvec, dim=1)
    score_neg = torch.mean(scores_A + scores_B) / 2

    # loss
    loss = score_neg - score_pos
    return loss 

def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,
                args,
                save_path,
                log_interval = 1,
                vis = True):
    net.train()
    running_total_loss = 0
    n_iter = epoch*len(train_loader)
    loss_file = '{}_loss_file.txt'.format(args.time_stamp)
    assert args.additional_weak == False
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))    
    for i, mini_batch in pbar:
        if vis == True:
            plot_NormMap2D_warped_Img(mini_batch['src_img'][0], mini_batch['trg_img_weak'][0], plot_name='test_{}'.format(i))    
        optimizer.zero_grad()

        # corr4d = model(mini_batch['trg_img_weak'].to(device),
        #                 mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
        if not args.strong_sup_loss:
            pdb.set_trace()
            #Sup & Unsup        
            pred_map_weak_gt, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_weak'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)    
            pred_map_weak = (pred_map_weak_gt, pred_map_weak_gt)
        # if args.keyout:
        #     mini_batch['trg_img_strong'] = cutout_aware(
        #         image= mini_batch['trg_img_strong'], 
        #         kpoint=mini_batch['trg_kps'], p=args.keyout, cut_n=10, 
        #         batch_size=mini_batch['trg_img'].size(0), bbox_trg=mini_batch['trg_bbox'],
        #         n_pts=mini_batch['n_pts'], cutout_size_min=args.keyout_size[0], cutout_size_max=args.keyout_size[1])
        mini_batch['trg_img_strong'] = semimatch.transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device), mode=args.aug_mode) 
        pred_map_strong, corr_strong, _, _ =\
                net(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device))


        _, loss_sup, loss_unsup, loss_self, diff_ratio =\
                semimatch(mini_batch=mini_batch, 
                        corr_weak=corr_weak, 
                        pred_map_weak=pred_map_weak,
                        corr_strong=corr_strong,
                        pred_map_strong=pred_map_strong,
                        occ_S_Tvec=occ_S_Tvec,
                        occ_T_Svec=occ_T_Svec,
                        epoch=epoch, n_iter=n_iter, it=i)

        if args.loss_sup == 'semi':
        # if args.loss_sup == 'ncnet':
        #     loss_sup = process_sup(net, 
        #             mini_batch['trg_additional_weak'].to(device),
        #             mini_batch['src_img'].to(device))
            # loss_unsup = loss_unsup *0.01
            if args.dynamic_unsup :
                if loss_unsup.int() == 0:
                    loss_unsup = torch.tensor(0.0, device=device)
                else:
                    loss_unsup = (torch.abs(loss_sup.detach()) / loss_unsup.detach()) * loss_unsup

            if args.zero_sup :
                loss_sup = torch.tensor(0.0, device=device)
            loss = loss_sup + loss_unsup
        elif args.loss_sup == 'sup':
            loss_unsup = torch.tensor(0.0, device=device)
            loss = loss_sup
            
        
        optimizer.step()
        loss_np = loss.data.cpu().numpy()
        print(loss_np)
        if i % log_interval == 0:
            print('train' + " Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        100.0 * i / len(train_loader),
                        loss_np
                ))
        running_total_loss += loss_np

        with open(os.path.join(args.cur_snapshot, loss_file), 'a+') as file:
            file.write(f'loss:{loss_sup.item(), loss_unsup.item()}|diff_ratio:{diff_ratio.item()}\n')

        # train_writer.add_scalar('train_loss_per_iter', loss.item(), n_iter)
        # train_writer.add_scalar('diff_point', diff_ratio, n_iter)
        n_iter += 1
        pbar.set_description(
            f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{loss.item():.3f}|SupLoss:{loss_sup.item():.3f}|UnsupLoss:{loss_unsup.item():.3f}|SelfsupLoss:{loss_self.item():.3f}|diff_ratio:{diff_ratio.item():.3f}')

    running_total_loss /= len(train_loader)
    return running_total_loss


train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)
best_train_loss = float("inf")
for epoch in range(start_epoch, args.epochs):
    train_loss[epoch-1] = train_epoch(net, 
                    optimizer, 
                    train_dataloader, 
                    device, 
                    epoch, 
                    train_writer,
                    args,
                    save_path)
    scheduler.step()
    # remember best loss
    is_best = train_loss[epoch - 1] < best_train_loss
    best_train_loss = min(train_loss[epoch - 1], best_train_loss)
    checkpoint_name = '{}_{}.pth'.format(args.benchmark, str(epoch))
    checkpoint_path = os.path.join(cur_snapshot, checkpoint_name)
    
    save_checkpoint(
        {
            "epoch": epoch,
            "args": args,
            "state_dict": model.state_dict(),
            "best_test_loss": best_train_loss,
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
        },
        is_best,
        checkpoint_path,
    )

print("Done!")


