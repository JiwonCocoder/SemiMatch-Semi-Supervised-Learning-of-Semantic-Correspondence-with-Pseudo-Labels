import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def visualize(mini_batch, 
              mask_src_kp2D_weak_not_T,
              mask_tgt_kp2D_weak_not_T,
              mask_src_kp2D_weak,
              mask_tgt_kp2D_weak,
              mask_src_kp2D_strong,
              mask_tgt_kp2D_strong,
              weak_transformed,
              device, args, n_iter, diff_idx=None):
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
                            weak_transformed), 3),
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

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_strong'][0].unsqueeze(0).to(device)), 3),
                            mask_src_kp2D_strong * image_ratio,
                            mask_tgt_kp2D_strong * image_ratio,
                            mini_batch['src_bbox'][0], None,
                            plot_name = 'diff_{}'.format(n_iter),
                            use_supervision ='diff', benchmark=args.benchmark, cur_snapshot=args.save_img_path, diff_idx=diff_idx)

def plot_image(im, return_im=True):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if im.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    im = im.mul(std).add(mean) * 255.0
    im = im.data.squeeze(0).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()

def plot_keypoint(im_pair, src_kps, tgt_kps, src_bbox, trg_bbox,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None, diff_idx=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]),
                            src_bbox[2] - src_bbox[0],
                            src_bbox[3] - src_bbox[1],
                            linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect_src)
    if trg_bbox is not None:
        rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]),
                                trg_bbox[2] - trg_bbox[0],
                                trg_bbox[3] - trg_bbox[1],
                                linewidth=4, edgecolor='red', facecolor='none')

        plt.gca().add_artist(rect_trg)

    if use_supervision == 'sup':
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[0, i]) + 256
            yb = float(tgt_kps[1, i])
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'semi':
        # if diff_idx is None:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            # c = 'coral'
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'diff':
        # else:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            c = 'red' if i in diff_idx else 'lime'
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()

# Working
def plot_diff_keypoint(im_pair, src_kps, tgt_kps, src_bbox,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect = plt.Rectangle((src_bbox[0], src_bbox[1]),
                            src_bbox[2] - src_bbox[0],
                            src_bbox[3] - src_bbox[1],
                            linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect)
    if use_supervision == 'sup':
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[0, i]) + 256
            yb = float(tgt_kps[1, i])
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'semi':
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()