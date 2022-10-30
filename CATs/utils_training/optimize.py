import pdb
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from data.dataset import TpsGridGen
from utils_training.utils import flow2kps
from utils_training.evaluation import Evaluator
from utils_training.vis import plot_from_norm, plot_keypoint, plot_keypoint_wo_line, gen_mask_kp2D_from_corr
import torch.nn as nn
import os

r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False, mask=None):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)

    # Sup loss
    if mask is None:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    
    # mask_2D unsup loss
    else:
        EPE_map = EPE_map[mask]

    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(mask)


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    criterion = nn.BCELoss()
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return criterion(logits, targets)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def consistency_loss(prob_w, prob_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    prob_w = prob_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        # pseudo_label = torch.softmax(logits_w, dim=-1)
        # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = prob_w.ge(p_cutoff).float()
        prob_s_1D= prob_s.view(-1)
        # zero_1D = torch.zeros_like(prob_s_1D, device=device, dtype=torch.float16)
        # prob_s = torch.cat((prob_s_1D, zero_1D), dim=1)
        # labels = torch.ones(int(prob_s_1D.size(0)), device=device, dtype=torch.float32)
        mask1D = prob_w.ge(p_cutoff).float().view(-1)

        if use_hard_labels:
            masked_loss = ce_loss(prob_s_1D, mask1D, use_hard_labels, reduction='none')
        # else:
        #     pseudo_label = torch.softmax(logits_w / T, dim=-1)
        #     masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss, mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')


def affine_transform(x, theta, interpolation_mode = 'bilinear'):
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid, mode = interpolation_mode)
    return x


def transform_by_grid(
                    src:torch.Tensor,
                    theta=None,
                    mode='aff',
                    interpolation_mode = 'bilinear',
                    padding_factor=1.0,
                    crop_factor=1.0,
                    use_mask=True):
    mode_list = []
    if mode == 'aff' or mode == 'tps':
        mode_list=[mode]
        theta_list = [theta]
    if mode == 'afftps':
        mode_list = ['aff', 'tps']
        theta_list = [theta[:,:6], theta[:,6:]]
    for i in range(len(mode_list)):
        theta = theta_list[i].float()
        sampling_grid = generate_grid(src.size(), theta, mode=mode_list[i])
        sampling_grid = sampling_grid.cuda()
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        src = F.grid_sample(src, sampling_grid.float(), align_corners=False, mode=interpolation_mode)
        mask = torch.autograd.Variable(torch.ones(src.size())).cuda()
        if use_mask:
            mask = F.grid_sample(mask, sampling_grid)
            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1
            src = mask*src
    return src


#Return transformed grid#
def generate_grid(img_size, theta=None, mode='aff'):
    out_h, out_w = img_size[2], img_size[3]
    gridGen = TpsGridGen(out_h, out_w)

    if mode == 'aff':
        return F.affine_grid(theta.view(-1,2,3), img_size)
    elif mode == 'tps':
        return gridGen(theta.view(-1,18,1,1))
    else:
        raise NotImplementedError


def convert_unNormFlow_to_unNormMap(flow):
    B, _, H, W = flow.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    map = flow + grid.cuda()
    return map
def sharpen(p, T=0.25):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p
def train_epoch(net,
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
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        flow_gt = mini_batch['flow'].to(device)
        #corr: softmax_with_temperatured_corr_T_Svec (3D)
        #weak
        if args.use_fbcheck_mask:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec= net(mini_batch['trg_img_weak'].to(device),
                                 mini_batch['src_img'].to(device), vis_fbcheck=True)
        else:
            pred_map_weak, corr_weak, _, _ = net(mini_batch['trg_img_weak'].to(device),
                                           mini_batch['src_img'].to(device))
        B, corrdim, W, H = corr_weak.size(0), args.feature_size * args.feature_size, args.feature_size, args.feature_size

        pred_flow_weak = net.unnormalise_and_convert_mapping_to_flow(pred_map_weak)
        #(check) interpolation mode
        transformed_pred_map_weak = transform_by_grid(pred_map_weak, theta = mini_batch[args.aug_mode].to(device), mode=args.aug_mode)
        transformed_pred_flow_weak = net.unnormalise_and_convert_mapping_to_flow(transformed_pred_map_weak)


        #not_transformed_weak
        # corr_weak = corr_weak / args.softmax_corr_temp

        if args.refined_corr_filtering == 'dual_softmax':
            corr_weak_prob = F.softmax(corr_weak, dim=1) * F.softmax(corr_weak, dim=2)
            corr_weak_prob = corr_weak_prob.view(-1, corrdim, W, H)
            score_weak, index_weak = torch.max(corr_weak_prob, dim=1)
            x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')            

        else:
            corr_weak = corr_weak.view(B, corrdim, H, W)
            corr_weak_prob = net.softmax_with_temperature(corr_weak, args.semi_softmax_corr_temp)

            if args.refined_corr_filtering == 'mutual':
                score_weak, index_weak = torch.max(corr_weak_prob, dim=1)
                x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')
            elif args.refined_corr_filtering == 'soft_argmax':
                # transformed_pred_map_weak = convert_unNormFlow_to_unNormMap(transformed_pred_flow_weak)
                # index_weak = transformed_pred_map_weak[:,1,:,:]*W + transformed_pred_map_weak[:,0,:,:]
                pred_map_weak = convert_unNormFlow_to_unNormMap(pred_flow_weak)
                index_weak = pred_map_weak[:,1,:,:]*W + pred_map_weak[:,0,:,:]
                index_weak = torch.round(index_weak)
                score_weak, _ = torch.max(corr_weak_prob, dim=1)
                x, y = pred_map_weak[:,0,:,:], pred_map_weak[:,1,:,:]

        score_mask = score_weak.ge(args.p_cutoff) #(B, 16, 16)

        # x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')
        bbox = torch.round(mini_batch['src_bbox'] / args.feature_size).cuda().long() # B x (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
        # for i in range(args.batch_size):
        #     bbox_mask[i][x1[i]:x2[i],y1[i]:y2[i]] = 1
        src_bbox_mask = (x >= x1.repeat_interleave(256).view(-1, W, W)) & \
                    (x <= x2.repeat_interleave(256).view(-1, W, W)) & \
                    (y >= y1.repeat_interleave(256).view(-1, W, W)) & \
                    (y <= y2.repeat_interleave(256).view(-1, W, W))

        bbox = torch.round(mini_batch['trg_bbox'] / args.feature_size).cuda().long() # B x (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W).repeat(B, 1, 1).cuda()
        yy = yy.view(1, H, W).repeat(B, 1, 1).cuda()

        # for i in range(args.batch_size):
        #     bbox_mask[i][x1[i]:x2[i],y1[i]:y2[i]] = 1
        trg_bbox_mask = (xx >= x1.repeat_interleave(256).view(-1, W, W)) & \
                    (xx <= x2.repeat_interleave(256).view(-1, W, W)) & \
                    (yy >= y1.repeat_interleave(256).view(-1, W, W)) & \
                    (yy <= y2.repeat_interleave(256).view(-1, W, W))
        #vis_variable(weak_not_T)
        mask2D_not_T = score_mask & src_bbox_mask & trg_bbox_mask
        if args.use_fbcheck_mask:
            mask2D_not_T = mask2D_not_T & occ_T_Svec.squeeze(1).bool()
            fb_mask = torch.round(transform_by_grid(occ_T_Svec.float().cuda(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                        interpolation_mode=args.interpolation_mode)).squeeze().bool()
        mask_tgt_kp2D_weak_not_T = (mask2D_not_T[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
        mask_src_1D = index_weak[0][mask2D_not_T[0]]
        mask_src_kp2D_weak_not_T = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

        #transformed_weak (prev)#
        # corr_weak = affine_transform(corr_weak, mini_batch['aff'].to(device))
        # bbox_mask = torch.round(transform_by_grid(bbox_mask.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
        #                                 interpolation_mode=args.interpolation_mode)).squeeze().bool()
        # score_mask = torch.round(transform_by_grid(score_mask.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
        #                                 interpolation_mode=args.interpolation_mode)).squeeze().bool()
        # index_weak = torch.round(transform_by_grid(index_weak.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
        #                                 interpolation_mode=args.interpolation_mode)).squeeze().long().clamp(0, 255)
        # mask_2D = score_mask & bbox_mask
        #
        # transformed_weak
        x_transformed, y_transformed = torch.round( transform_by_grid(x.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                              interpolation_mode=args.interpolation_mode)).squeeze().long().clamp(0, 255), \
                                       torch.round(transform_by_grid(y.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                            interpolation_mode=args.interpolation_mode)).squeeze().long().clamp(0, 255)
        index_weak = y_transformed * W + x_transformed

        # if args.use_fbcheck_mask:
        #     mask_2D = mask_2D & fb_mask
        mask_2D = torch.round(transform_by_grid(mask2D_not_T.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                        interpolation_mode=args.interpolation_mode)).squeeze().bool()
        #vis_variable(weak_T)
        mask_tgt_kp2D_weak = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
        mask_src_1D = index_weak[0][mask_2D[0]]
        mask_src_kp2D_weak = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

        #strong#
        mini_batch['trg_img_strong'] = transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device),  mode=args.aug_mode) 
        pred_map_strong, corr_strong, _, _ = net(mini_batch['trg_img_strong'].to(device),
                               mini_batch['src_img'].to(device))
        pred_flow_strong = net.unnormalise_and_convert_mapping_to_flow(pred_map_strong)

        corr_strong = corr_strong.view(B, corrdim, H, W)
        corr_strong_prob = net.softmax_with_temperature(corr_strong, args.semi_softmax_corr_temp)
        score_strong, index_strong = torch.max(corr_strong_prob, dim=1)
        #vis_variable(strong_T)
        mask_tgt_kp2D_strong = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
        mask_src_1D = index_strong[0][mask_2D[0]]
        mask_src_kp2D_strong = torch.cat(((mask_src_1D %  W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

        #plot_by_img_warping
        # plot_from_norm(mini_batch['src_img'][0].unsqueeze(0).to(device), mini_batch['trg_img_strong'][0].unsqueeze(0).to(device),
        #                mini_batch['src_bbox'][0],
        #                index_weak[0].unsqueeze(0), index_strong[0].unsqueeze(0), 16, i, use_mask=True)

        # plot_from_norm(mini_batch['src_img'][0].unsqueeze(0).to(device), mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),
        #                mini_batch['src_bbox'][0],
        #                index_weak[0].unsqueeze(0), index_strong[0].unsqueeze(0), 16, i, use_mask=True)

        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        #(corr_T_Svec: source coord, tgt_vec) corr.reshape (B, S, S, S*S)

        #semi_loss#
        if args.loss_mode == 'contrastive':
            B, Svec, T_h, T_w = corr_strong.size()
            masked_corr_strong = corr_strong.permute(0,2,3,1).reshape(B*T_h *T_w, Svec)
            masked_corr_strong = masked_corr_strong[mask_2D.view(-1)]

            masked_index_weak = index_weak[mask_2D].long()
            masked_num = masked_index_weak.size(0)
            if not masked_num == 0:
                mask_pixelCT = torch.zeros_like(masked_corr_strong).bool()
                mask_pixelCT[torch.arange(masked_num), masked_index_weak] = True
                positive = masked_corr_strong[mask_pixelCT].view(masked_num, -1)
                negative = masked_corr_strong[~mask_pixelCT].view(masked_num, -1)
                masked_pred = torch.cat([positive, negative], dim=1)

                masked_labels = torch.zeros(int(masked_num), device=device, dtype=torch.int64)
                masked_labels = masked_labels.detach()

                eps_temp = 1e-6
                masked_pred_with_temp = (masked_pred / args.semi_contrastive_temp) + eps_temp

                loss_unsup = criterion(masked_pred_with_temp, masked_labels) * args.semi_lambda
            else:
                loss_unsup = torch.tensor(0.0, device=device)

        elif args.loss_mode == 'EPE':
            #(type_1) affined_by_flow
            # pred_flow_weak = affine_transform(pred_flow_weak, mini_batch['aff'].to(device))
            pred_weak_transformed_map = torch.cat([x_transformed.unsqueeze(1), y_transformed.unsqueeze(1)], dim=1)
            pred_weak_transformed_flow = net.unnormalise_and_convert_mapping_to_flow(pred_weak_transformed_map)
            if not mask_2D.sum() ==0:
            #(type_1) affined_by_map
                loss_unsup = EPE(pred_flow_strong, pred_weak_transformed_flow.detach(), mask=(mask_2D)) * args.semi_lambda
            else:
                loss_unsup = torch.tensor(0.0, device=device)

        #sup_loss#
        loss_sup = EPE(pred_flow_weak, flow_gt)

        # self_loss#
        if args.use_self_loss:
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).cuda().float()
            grid[:,0,:,:] = 2 * grid[:,0,:,:].clone() / (W-1) -1
            grid[:, 1, :, :] = 2 * grid[:, 1, :, :].clone() / (H - 1) - 1
            self_map_gt = transform_by_grid(grid, mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                            interpolation_mode=args.interpolation_mode)

            self_flow_gt = net.unnormalise_and_convert_mapping_to_flow(self_map_gt)
            # net_input: trg, src
            pred_map_self, corr_self, _, _ = net(mini_batch['trg_img_strong'].to(device),
                                                 mini_batch['trg_img_weak'].to(device))
            pred_self_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map_self)
            loss_self = EPE(pred_self_flow, self_flow_gt.detach()) * args.self_lambda
            if i % 100 == 0:
                # vis#
                net.plot_NormMap2D_warped_Img(mini_batch['trg_img_weak'].to(device)[0].unsqueeze(0), mini_batch['trg_img_strong'].to(device)[0].unsqueeze(0),
                                            pred_map_self[0].unsqueeze(0), self_map_gt[0].unsqueeze(0),
                                            args.feature_size,
                                            plot_name="warped_self_{}".format(n_iter),
                                            self_img= True)

        else:
            loss_self = torch.tensor(0.0, device=device)
        #me_max loss#
        if args.use_me_max_loss:
            avg_prob = torch.mean(corr_weak_prob.view(B, corrdim, corrdim), dim=2)
            rloss = -1.0 * torch.sum(torch.log(avg_prob**(-avg_prob)), dim=1).mean()
            rloss= rloss
        else:
            rloss = torch.tensor(0.0, device=device)

        if epoch > args.warmup_epoch:
            Loss = loss_sup + loss_unsup + loss_self + rloss

        else:
            Loss = loss_sup
            loss_unsup = torch.tensor(0.0, device=device)
        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item()

        with open(os.path.join(args.cur_snapshot, loss_file), 'a+') as file:
            file.write(f'{loss_sup, loss_unsup}\n')

        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
            f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_sup:.3f}|UnsupLoss:{loss_unsup:.3f}|SelfsupLoss:{loss_self:.3f}|rloss:{rloss:.3f}')
        #vis section#
        #vis_sup
        if i % 100 == 0:
            if mask_2D[0].sum() == 0:
                pass
            else:

                image_ratio = 16
                #vis_sup_GT
                plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                                        mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                                        mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                                        mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],
                                        mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                                        plot_name = 'GT_{}'.format(n_iter),
                                        use_supervision ='sup', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

                plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                                        mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                                        mask_src_kp2D_weak_not_T * image_ratio,
                                        mask_tgt_kp2D_weak_not_T * image_ratio,
                                        mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                                        plot_name = 'weak_not_T_{}'.format(n_iter),
                                        use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

                #vis_unsup
                plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                                        # affine_transform(mini_batch['trg_img_weak'][0].unsqueeze(0),  mini_batch['aff'][0]).to(device)), 3),
                                        transform_by_grid(mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),  mini_batch[args.aug_mode][0].unsqueeze(0).to(device), mode=args.aug_mode)),3),
                                        mask_src_kp2D_weak * image_ratio,
                                        mask_tgt_kp2D_weak * image_ratio,
                                        mini_batch['src_bbox'][0], None,
                                        plot_name = 'weak_{}'.format(n_iter),
                                        use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

                plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                                        mini_batch['trg_img_strong'][0].unsqueeze(0).to(device)), 3),
                                        mask_src_kp2D_strong * image_ratio,
                                        mask_tgt_kp2D_strong * image_ratio,
                                        mini_batch['src_bbox'][0], None,
                                        plot_name = 'strong_{}'.format(n_iter),
                                        use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)
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