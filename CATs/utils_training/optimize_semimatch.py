import pdb
import time
import sys
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F



sys.path.append('.')
# from data.dataset import CorrespondenceDataset, TpsGridGen
from semimatch.semimatch import SemiMatch
from semimatch.semimatch_loss import EPE
from semimatch.utils import flow2kps
from semimatch.evaluation import Evaluator



def train_epoch(semimatch:SemiMatch,
                net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,
                args,
                save_path):
    n_iter = epoch*len(train_loader)
    
    net.train()
    running_total_loss = 0

    loss_file = '{}_loss_file.txt'.format(args.time_stamp)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    assert args.strong_sup_loss == False
    assert args.additional_weak == True
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        #weak
        if args.strong_sup_loss and not args.additional_weak:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
        elif not args.strong_sup_loss and not args.additional_weak:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_weak'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
                
        elif not args.strong_sup_loss and args.additional_weak:
            pred_map_weak_gt, _, _, _ =\
                net(mini_batch['trg_img_weak'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)    # GT에 쓸거
            pred_map_weak_unsup, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_additional_weak'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)    # Unsup에 쓸거
            pred_map_weak = (pred_map_weak_gt, pred_map_weak_unsup)
            
        else:
            raise NotImplementedError
        
        #strong#
        if args.aug_mixup != 0 :
            ratio = random.random() * args.aug_mixup
            mini_batch['trg_img_strong'] = mini_batch['trg_img_strong'] * (1-ratio) + torch.flip(mini_batch['trg_img_strong'],[0]) * ratio
        
        if args.keymix:
            mini_batch['src_img'], mini_batch['trg_img_strong'] =\
                semimatch.keypoint_cutmix(image_s=mini_batch['src_img'], kpoint_s=mini_batch['src_kps'], 
                    image_t=mini_batch['trg_img_strong'], kpoint_t=mini_batch['trg_kps'], 
                    mask_size_min=5, mask_size_max=20, p=args.keymix, batch_size=mini_batch['trg_img'].size(0),
                    n_pts=mini_batch['n_pts'])
        if args.keyout:
            mini_batch['trg_img_strong'] = semimatch.cutout_aware(
                image= mini_batch['trg_img_strong'], 
                kpoint=mini_batch['trg_kps'], p=args.keyout, cut_n=10, 
                batch_size=mini_batch['trg_img'].size(0), bbox_trg=mini_batch['trg_bbox'],
                n_pts=mini_batch['n_pts'], cutout_size_min=args.keyout_size[0], cutout_size_max=args.keyout_size[1])
        
        mini_batch['trg_img_strong'] = semimatch.transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device), mode=args.aug_mode) 
        pred_map_strong, corr_strong, _, _ =\
                net(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device))
        
        loss_sup, loss_unsup, loss_self, diff_ratio =\
                semimatch(mini_batch=mini_batch, 
                        corr_weak=corr_weak, 
                        pred_map_weak=pred_map_weak,
                        corr_strong=corr_strong,
                        pred_map_strong=pred_map_strong,
                        occ_S_Tvec=occ_S_Tvec,
                        occ_T_Svec=occ_T_Svec,
                        epoch=epoch, n_iter=n_iter, it=i)
        Loss = loss_sup + loss_unsup
        if args.dynamic_unsup :
            if loss_unsup == 0:
                loss_unsup = torch.tensor(0.0, device=device)
            else:
                loss_unsup = (loss_sup.detach() / loss_unsup.detach()) * loss_unsup
            Loss = loss_sup + loss_unsup
        if args.zero_sup :
            loss_sup = torch.tensor(0.0, device=device)
            Loss = loss_sup + loss_unsup

        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item()

        with open(os.path.join(args.cur_snapshot, loss_file), 'a+') as file:
            file.write(f'loss:{loss_sup.item(), loss_unsup.item()}|diff_ratio:{diff_ratio.item()}\n')

        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        train_writer.add_scalar('diff_point', diff_ratio, n_iter)
        n_iter += 1
        pbar.set_description(
            f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_sup.item():.3f}|UnsupLoss:{loss_unsup.item():.3f}|SelfsupLoss:{loss_self.item():.3f}|diff_ratio:{diff_ratio.item():.3f}')

    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            pred_map, _, _, _ = net(mini_batch['trg_img'].to(device),
                                    mini_batch['src_img'].to(device))
            pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck


def validate_epoch_test(net,val_loader,device,epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        pck_cat_array = dict()
        mean_pck_cat_array = dict()
        for i, mini_batch in pbar:
            
            flow_gt = mini_batch['flow'].to(device)
            pred_map, _, _, _ = net(mini_batch['trg_img'].to(device),
                                    mini_batch['src_img'].to(device))
            pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result['pck']

            for idx, cat in enumerate(mini_batch['category']) :
                if cat not in pck_cat_array :
                    pck_cat_array[cat] = []
                pck_cat_array[cat].append(eval_result['pck'][idx])
                
            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

        for key in pck_cat_array :
            mean_pck_cat_array[key] = sum(pck_cat_array[key]) / len(pck_cat_array[key])



    return running_total_loss / len(val_loader), mean_pck, mean_pck_cat_array