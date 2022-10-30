import pdb
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
def unNormMap1D_to_NormMap2D(idx_B_Avec, fs1, delta4d=None, k_size=1, do_softmax=False, scale='centered',
                             return_indices=False,
                             invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
    batch_size, sz = idx_B_Avec.shape
    w = sz // fs1
    h = w
    # fs2: width, fs1: height
    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        # XB, YB = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    elif scale == 'positive':
        XA, YA = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        # XB, YB = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    JA, IA = np.meshgrid(range(w), range(h))
    # JB, IB = np.meshgrid(range(w), range(h))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    # XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).contiguous().view(1, -1))), Variable(
        to_cuda(torch.LongTensor(IA).contiguous().view(1, -1)))
    # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    iA = IA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    jA = JA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    # iB = IB.expand_as(iA)
    # jB = JB.expand_as(jA)

    xA = XA[iA.contiguous().view(-1), jA.contiguous().view(-1)].contiguous().view(batch_size, -1)
    yA = YA[iA.contiguous().view(-1), jA.contiguous().view(-1)].contiguous().view(batch_size, -1)
    # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

    xA_WTA = xA.contiguous().view(batch_size, 1, h, w)
    yA_WTA = yA.contiguous().view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

    return Map2D_WTA


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
    mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow

def warp_from_NormMap2D(x, NormMap2D):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid

    vgrid = NormMap2D.permute(0, 2, 3, 1).contiguous()
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)  # N,C,H,W
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    #
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output*mask
    # return output


def gen_mask_kp2D_from_corr(corr, p_cutoff, temperature):
    GPU_NUM = torch.cuda.current_device()
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    corr = F.softmax(corr / temperature, dim=1)
    score, index = torch.max(corr, dim=1)
    B, f_h, f_w = index.size()

    mask_2D = score.ge(p_cutoff)  # (B, 16, 16)
    mask_tgt_kp2D = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)

    mask_src_1D = index[0][mask_2D[0]]
    mask_src_y = mask_src_1D // f_w
    mask_src_x = mask_src_1D % f_w
    mask_src_kp2D = torch.cat((mask_src_x.view(1, -1), mask_src_y.view(1, -1)), dim=0)
    return mask_src_kp2D, mask_tgt_kp2D

def gen_mask_kp2D_from_index1D(corr, p_cutoff, temperature):
    GPU_NUM = torch.cuda.current_device()
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')


    B, f_h, f_w = index.size()

    mask_tgt_kp2D = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)

    mask_src_1D = index[0][mask_2D[0]]
    mask_src_y = mask_src_1D // f_w
    mask_src_x = mask_src_1D % f_w
    mask_src_kp2D = torch.cat((mask_src_x.view(1, -1), mask_src_y.view(1, -1)), dim=0)
    return mask_src_kp2D, mask_tgt_kp2D

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

def plot_from_norm(src_img, tgt_img, src_bbox, index_weak, index_strong,
                   scale_factor, count, use_mask=False, plot_name=None):
    _, h, w = index_weak.size()
    index1D_weak= index_weak.view(1, -1)
    norm_map2D_weak = unNormMap1D_to_NormMap2D(index1D_weak, h)

    index1D_strong= index_strong.view(1, -1)
    norm_map2D_strong = unNormMap1D_to_NormMap2D(index1D_strong, h)

    norm_map2D_weak = F.interpolate(input=norm_map2D_weak, scale_factor=scale_factor, mode='bilinear',
                                      align_corners=True)
    norm_map2D_strong = F.interpolate(input=norm_map2D_strong, scale_factor=scale_factor, mode='bilinear',
                                      align_corners=True)

    warp_S_Tvec_by_weak = warp_from_NormMap2D(src_img, norm_map2D_weak)  # (B, 2, H, W)

    warp_S_Tvec_by_strong = warp_from_NormMap2D(src_img, norm_map2D_strong)

    # if use_mask:
    #     mask_strong = norm_map2D_strong.ge(-1) & norm_map2D_strong.le(1)
    #     mask_weak = norm_map2D_weak.ge(-1) & norm_map2D_weak.le(1)
    #     pdb.set_trace()

    # mask_weak = F.interpolate(input=mask_weak.type(torch.float),
    #                                     scale_factor=scale_factor,
    #                                     mode='bilinear',
    #                                     align_corners=True)
    # mask_strong = F.interpolate(input=mask_strong.type(torch.float),
    #                                     scale_factor=scale_factor,
    #                                     mode='bilinear',
    #                                     align_corners=True)
    # warp_S_Tvec_by_weak = warp_S_Tvec_by_weak * mask_weak
    # warp_S_Tvec_by_strong = warp_S_Tvec_by_strong * mask_strong

    src_img = plot_image(src_img)
    tgt_img = plot_image(tgt_img)
    warp_S_Tvec_by_weak = plot_image(warp_S_Tvec_by_weak)
    warp_S_Tvec_by_strong = plot_image(warp_S_Tvec_by_strong)

    rect = patches.Rectangle((src_bbox[0], src_bbox[1]),
                             src_bbox[2] - src_bbox[0],
                             src_bbox[3] - src_bbox[1],
                             linewidth=4, edgecolor='r', facecolor='none')
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_img)
    axis[0][0].add_patch(rect)
    axis[0][0].set_title("src_img_" + str(count))
    axis[0][1].imshow(tgt_img)
    axis[0][1].set_title("tgt_img_" + str(count))
    axis[1][0].imshow(warp_S_Tvec_by_weak)
    axis[1][0].set_title("warp_by_weak" + str(count))
    axis[1][1].imshow(warp_S_Tvec_by_strong)
    axis[1][1].set_title("warp_by_strong" + str(count))
    fig.savefig('{}/{}.png'.format('./weak_strong', plot_name),
                bbox_inches='tight')


def plot_keypoint_wo_line(im_src, im_tgt, src_kps, tgt_kps, save_dir, plot_name):
    im_src = plot_image(im_src, return_im=True)
    im_tgt = plot_image(im_tgt, return_im=True)

    fig, axis = plt.subplots(1, 2, figsize=(50, 50))
    axis[0].imshow(im_src)
    axis[1].imshow(im_tgt)
    count = 0
    for i in range(src_kps.size(1)):
        xa = float(src_kps[0, i])
        ya = float(src_kps[1, i])
        xb = float(tgt_kps[1, i])
        yb = float(tgt_kps[0, i])
        if (50 <= xa <= 200 and 50 <= ya <= 200) and (50 <= xb <= 200 and 50 <= yb <= 200):
            count +=1
            c = np.random.rand(3)
            circle_a = plt.Circle((xa, ya), 5, color=c)
            circle_b = plt.Circle((xb, yb), 5, color=c)
            axis[0].add_patch(circle_a)
            axis[1].add_patch(circle_b)
        if count > 10:
            continue
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    fig.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()

def plot_keypoint(im_pair, src_kps, tgt_kps, src_bbox, trg_bbox,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None):
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
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[0, i]) + 256
            yb = float(tgt_kps[1, i])
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