import os
from re import match
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import pdb
# from .CATs.dataset.dataset_utils import TpsGridGen
from .vis import unnormalise_and_convert_mapping_to_flow
from .semimatch_utils import visualize
from .semimatch_loss import EPE, ce_loss, consistency_loss
from .utils import flow2kps
from .evaluation import Evaluator
from .dataset_utils import TpsGridGen
sys.path.append('.')
# from lib.gen_transform import TpsGridGen

def unNormMap1D_to_NormMap2D(idx_B_Avec, fs1=16, delta4d=None, k_size=1, do_softmax=False, scale='centered',
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


class SemiMatch(nn.Module):
    def __init__(self, net, device, args):
        super().__init__()
        self.net = net
        self.device = device
        self.args = args
        self.count = 0
        self.criterion = nn.CrossEntropyLoss()

        # class-aware
        self.class_pcksum = torch.zeros(20)
        self.class_total = torch.zeros(20)
        
        self.sparse_exp = args.sparse_exp

        self.feature_size = 16
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

    def affine_transform(x, theta, interpolation_mode='bilinear'):
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, mode = interpolation_mode)
        return x



    def transform_by_grid(self,
                          src:torch.Tensor,
                          theta=None,
                          mode='aff',
                          interpolation_mode='bilinear',
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
            sampling_grid = self.generate_grid(src.size(), theta, mode=mode_list[i])
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
    def generate_grid(self, img_size, theta=None, mode='aff'):
        out_h, out_w = img_size[2], img_size[3]
        gridGen = TpsGridGen(out_h, out_w)

        if mode == 'aff':
            return F.affine_grid(theta.view(-1,2,3), img_size)
        elif mode == 'tps':
            return gridGen(theta.view(-1,18,1,1))
        else:
            raise NotImplementedError


    def convert_unNormFlow_to_unNormMap(self, flow):
        B, _, H, W = flow.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        map = flow + grid.cuda()
        return map

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        x_normal = np.linspace(-1,1,corr.size(2))
        x_normal = nn.Parameter(torch.tensor(x_normal, device='cuda', dtype=torch.float, requires_grad=False))
        y_normal = np.linspace(-1,1,corr.size(3))
        y_normal = nn.Parameter(torch.tensor(y_normal, device='cuda', dtype=torch.float, requires_grad=False))
        
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def warp_from_NormMap2D(self,x, NormMap2D):
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
        # return output*mask
        return output
    
    def plot_NormMap2D_warped_Img(self, src_img, tgt_img,
                                  norm_map2D_S_Tvec, norm_map2D_T_Svec,
                                  scale_factor,
                                  occ_S_Tvec=None, occ_T_Svec=None, plot_name=None, self_img=False):
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
            tgt_img = tgt_img.mul(std).add(mean)
            src_img = src_img.mul(std).add(mean)
        norm_map2D_S_Tvec = F.interpolate(input=norm_map2D_S_Tvec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)
        norm_map2D_T_Svec = F.interpolate(input=norm_map2D_T_Svec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)
        if self_img:
            masked_warp_S_Tvec = self.warp_from_NormMap2D(src_img, norm_map2D_S_Tvec)  # (B, 2, H, W)
        else:
            masked_warp_S_Tvec = self.warp_from_NormMap2D(tgt_img, norm_map2D_S_Tvec)  # (B, 2, H, W)

        masked_warp_T_Svec = self.warp_from_NormMap2D(src_img, norm_map2D_T_Svec)
        if occ_S_Tvec is not None and occ_T_Svec is not None:
            mask_img_S_Tvec = F.interpolate(input=occ_S_Tvec.type(torch.float),
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            align_corners=True)
            mask_img_T_Svec = F.interpolate(input=occ_T_Svec.type(torch.float),
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            align_corners=True)
            masked_warp_T_Svec = mask_img_T_Svec * masked_warp_T_Svec
            masked_warp_S_Tvec = mask_img_S_Tvec * masked_warp_S_Tvec
        tgt_img = tgt_img * 255.0
        src_img = src_img * 255.0
        tgt_img = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
        src_img = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

        masked_warp_T_Svec = masked_warp_T_Svec.data.squeeze(0).transpose(0, 1).transpose(1,
                                                                                          2).cpu().numpy()
        masked_warp_S_Tvec = masked_warp_S_Tvec.data.squeeze(0).transpose(0, 1).transpose(1,
                                                                                          2).cpu().numpy()

        fig, axis = plt.subplots(2, 2, figsize=(50, 50))
        axis[0][0].imshow(src_img)
        axis[0][0].set_title("src_img_" + str(self.count))
        axis[0][1].imshow(tgt_img)
        axis[0][1].set_title("tgt_img_" + str(self.count))
        axis[1][0].imshow(masked_warp_S_Tvec)
        axis[1][0].set_title("warp_T_to_S_" + str(self.count))
        axis[1][1].imshow(masked_warp_T_Svec)
        axis[1][1].set_title("warp_S_to_T_" + str(self.count))
        fig.savefig('{}/{}.png'.format(self.args.save_img_path, plot_name),
                    bbox_inches='tight')
        plt.close(fig)
        
    def generate_mask(self, flow, flow_bw, alpha_1, alpha_2):
        
        output_sum = flow + flow_bw
        output_sum = torch.sum(torch.pow(output_sum.permute(0, 2, 3, 1), 2), 3)
        output_scale_sum = torch.sum(torch.pow(flow.permute(0, 2, 3, 1), 2), 3) + torch.sum(
            torch.pow(flow_bw.permute(0, 2, 3, 1), 2), 3)
        occ_thresh = alpha_1 * output_scale_sum + alpha_2
        occ_bw = (output_sum > occ_thresh).float()
        mask_bw = 1. - occ_bw

        return mask_bw
    def calOcc(self, NormFlowMap2D_S_Tvec, NormMap2D_S_Tvec,
                     NormFlowMap2D_T_Svec, NormMap2D_T_Svec):
        Norm_flow2D_S_Tvec_bw = nn.functional.grid_sample(NormFlowMap2D_T_Svec, NormMap2D_S_Tvec.permute(0, 2, 3, 1))
        Norm_flow2D_T_Svec_bw = nn.functional.grid_sample(NormFlowMap2D_S_Tvec, NormMap2D_T_Svec.permute(0, 2, 3, 1))
        occ_S_Tvec = self.generate_mask(NormFlowMap2D_S_Tvec, Norm_flow2D_S_Tvec_bw,
                                   self.args.alpha_1, self.args.alpha_2)  # compute: feature_map-based
        occ_T_Svec = self.generate_mask(NormFlowMap2D_T_Svec, Norm_flow2D_T_Svec_bw,
                                   self.args.alpha_1, self.args.alpha_2)  # compute: feature_map-based
        occ_S_Tvec = occ_S_Tvec.unsqueeze(1)
        occ_T_Svec = occ_T_Svec.unsqueeze(1)

        return occ_S_Tvec, occ_T_Svec

    def unNormMap1D_to_NormMap2D_and_NormFlow2D(self,idx_B_Avec, h, w, delta4d=None, k_size=1, do_softmax=False, scale='centered',
                                 return_indices=False,
                                 invert_matching_direction=False):
        to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
        batch_size, sz = idx_B_Avec.shape

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

        xA = xA.contiguous().view(batch_size, 1, h, w)
        yA = yA.contiguous().view(batch_size, 1, h, w)
        Map2D= torch.cat((xA, yA), 1).float()
        grid = torch.cat((XA.unsqueeze(0), YA.unsqueeze(0)), dim =0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if Map2D.is_cuda:
            grid = grid.cuda()
        flow2D = Map2D - grid

        return Map2D, flow2D

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal.cuda()).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal.cuda()).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    

    def cutout_aware(self, image, kpoint, mask_size_min=3, mask_size_max=15, p=0.2, cutout_inside=True, mask_color=(0, 0, 0),
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
    
    def keypoint_cutmix(self, image_s, kpoint_s,
                        image_t, kpoint_t,
                        mask_size_min=5, mask_size_max=20, p=0.3,
                        batch_size=20, n_pts=None):
        def _cutmix(image1, x1, y1,
                    image2, x2, y2, mask_size, mask_size_half):
            # image = np.asarray(image).copy()

            image_new1 = torch.clone(image1)
            image_new2 = torch.clone(image2)

            if torch.rand(1) > p:
                return image1, image2

            _, h1, w1 = image1.shape
            _, h2, w2 = image2.shape

            cx1 = x1
            cy1 = y1
            xmin1 = cx1 - mask_size_half
            ymin1 = cy1 - mask_size_half
            xmax1 = xmin1 + mask_size
            ymax1 = ymin1 + mask_size
            xmin1 = max(0, xmin1)
            ymin1 = max(0, ymin1)
            xmax1 = min(w1, xmax1)
            ymax1 = min(h1, ymax1)

            cx2 = x2
            cy2 = y2
            xmin2 = cx2 - mask_size_half
            ymin2 = cy2 - mask_size_half
            xmax2 = xmin2 + mask_size
            ymax2 = ymin2 + mask_size
            xmin2 = max(0, xmin2)
            ymin2 = max(0, ymin2)
            xmax2 = min(w2, xmax2)
            ymax2 = min(h2, ymax2)

            if (ymax1 - ymin1) != (ymax2 - ymin2) or (xmax1 - xmin1) != (xmax2 - xmin2):
                return image1, image2

            image_new1[:, ymin1:ymax1, xmin1:xmax1] = image2[:, ymin2:ymax2, xmin2:xmax2]
            image_new2[:, ymin2:ymax2, xmin2:xmax2] = image1[:, ymin1:ymax1, xmin1:xmax1]
            return image_new1, image_new2

        image_new1 = []
        image_new2 = []
        for b in range(batch_size):

            buffer1 = image_s[b]
            buffer2 = image_t[b]

            # kpoint_n = len(kpoint_s)
            # non_zero_index_s = torch.nonzero(kpoint_s[b][0])
            # non_zero_index_t = torch.nonzero(kpoint_t[b][0])
            non_zero_index_s = torch.arange(n_pts[b])
            non_zero_index_t = torch.arange(n_pts[b])

            for n in range(len(non_zero_index_t)):
                mask_size = torch.randint(low=mask_size_min, high=mask_size_max, size=(1,))
                mask_size_half = mask_size // 2
                buffer1, buffer2 = _cutmix(buffer1, int(kpoint_s[b][0][n]), int(kpoint_s[b][1][n]),
                                        buffer2, int(kpoint_t[b][0][n]), int(kpoint_t[b][1][n]),
                                        mask_size, mask_size_half)
            image_new1.append(buffer1)
            image_new2.append(buffer2)
        image_new1 = torch.stack(image_new1, dim=0)
        image_new2 = torch.stack(image_new2, dim=0)
        return image_new1, image_new2
    
    def forward(self, 
                mini_batch:dict, 
                corr_weak=None, 
                pred_map_weak=None,
                corr_strong=None,
                pred_map_strong=None,
                occ_S_Tvec=None,
                occ_T_Svec=None,
                epoch=None, n_iter=None, it=None):


        pred_map_weak_gt, pred_map_weak = pred_map_weak
        pred_flow_weak_gt = unnormalise_and_convert_mapping_to_flow(pred_map_weak_gt)
        assert len(corr_weak.shape) == 3

        flow_gt = mini_batch['flow'].to(self.device)
        B, corrdim, H, W =\
            corr_weak.size(0), self.args.feature_size * self.args.feature_size,\
                            self.args.feature_size, self.args.feature_size
        self.args.feature_width, self.args.feature_height = self.args.feature_size, self.args.feature_size

        pred_flow_weak = unnormalise_and_convert_mapping_to_flow(pred_map_weak)


    
        # corr_weak_S_Tvec = corr_weak.transpose(-1,-2)
        # corr_weak_S_Tvec = corr_weak_S_Tvec.reshape(B, corrdim, H, W)
        # corr_weak_S_Tvec_prob = self.softmax_with_temperature(corr_weak_S_Tvec, self.args.semi_softmax_corr_temp)

        # corr_weak_S_Tvec_transformed = (self.transform_by_grid(corr_weak_S_Tvec, mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
        #                                 interpolation_mode=self.args.interpolation_mode))
        # corr_weak_S_Tvec_transformed_prob = self.softmax_with_temperature(corr_weak_S_Tvec_transformed, self.args.semi_softmax_corr_temp)

        
        corr_weak = corr_weak.view(B, corrdim, H, W)
        corr_weak_prob = self.softmax_with_temperature(corr_weak, self.args.semi_softmax_corr_temp)

        corr_weak_transformed = (self.transform_by_grid(corr_weak, mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode))
        corr_weak_transformed_prob = self.softmax_with_temperature(corr_weak_transformed, self.args.semi_softmax_corr_temp)        
        #Argmax from corr & corr_transformed
        if self.args.refined_corr_filtering == 'mutual':

            # score_weak_S_Tvec, index_weak_S_Tvec = torch.max(corr_weak_S_Tvec_prob, dim=1)
            # score_weak_S_Tvec_transformed, index_weak_S_Tvec_transformed = torch.max(corr_weak_S_Tvec_transformed_prob, dim=1)
            # x_S_Tvec, y_S_Tvec = (index_weak_S_Tvec % W), (index_weak_S_Tvec // W)
            score_weak_T_Svec, index_weak_T_Svec = torch.max(corr_weak_prob, dim=1)
            x_T_Svec, y_T_Svec = (index_weak_T_Svec % W), (index_weak_T_Svec // W)

            score_weak_T_Svec_transformed, index_weak_T_Svec_transformed = torch.max(corr_weak_transformed_prob, dim=1)
            x_T_Svec_transformed, y_T_Svec_transformed = (index_weak_T_Svec_transformed % W), (index_weak_T_Svec_transformed // W)

        elif self.args.refined_corr_filtering == 'soft_argmax':
            pred_map_weak = self.convert_unNormFlow_to_unNormMap(pred_flow_weak)
            index_weak_T_Svec = pred_map_weak[:,1,:,:].int()*W + pred_map_weak[:,0,:,:].int()
            # index_weak_T_Svec = torch.round(index_weak_T_Svec)
            score_weak_T_Svec, _ = torch.max(corr_weak_prob, dim=1)

            x_T_Svec, y_T_Svec = pred_map_weak[:,0,:,:].int(), pred_map_weak[:,1,:,:].int()
        # entropy
        uncertainty_transformed = (-(corr_weak_transformed_prob+1e-6) * torch.log(corr_weak_transformed_prob+1e-6)).sum(dim=1)
        index_weak = index_weak_T_Svec.detach().clone()
        index_weak_transformed = index_weak_T_Svec_transformed.detach().clone()

        #mask calcucation#
        #score_mask#
        score_mask = score_weak_T_Svec.ge(self.args.p_cutoff)
        score_mask_transformed = score_weak_T_Svec_transformed.ge(self.args.p_cutoff)

        #resize src bbox size aligned to map size
        bbox = mini_batch['src_bbox'].cuda()
        x1, y1, x2, y2 = torch.round(bbox[:,0] / W).long(),\
                         torch.round(bbox[:,1] / H).long(),\
                         torch.round(bbox[:,2] / W).long(),\
                         torch.round(bbox[:,3] / H).long()
        #src_mask#
        src_bbox_mask = (x_T_Svec >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (x_T_Svec <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y_T_Svec >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y_T_Svec <= y2.repeat_interleave(H*W).view(-1, H, W))

        src_bbox_mask_transformed = (x_T_Svec_transformed >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (x_T_Svec_transformed <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y_T_Svec_transformed >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y_T_Svec_transformed <= y2.repeat_interleave(H*W).view(-1, H, W)) 

                        # (x >= min_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                        # (x <= max_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                        # (y >= min_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                        # (y <= max_margin.repeat_interleave(H*W).view(-1, H, W)) 
        #trg_mask#
        bbox = mini_batch['trg_bbox'].cuda() # B x (x1, y1, x2, y2)
        x1, y1, x2, y2 = torch.round(bbox[:,0] / W).long(),\
                         torch.round(bbox[:,1] / H).long(),\
                         torch.round(bbox[:,2] / W).long(),\
                         torch.round(bbox[:,3] / H).long()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W).repeat(B, 1, 1).cuda()
        yy = yy.view(1, H, W).repeat(B, 1, 1).cuda()
        trg_bbox_mask = (xx >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (xx <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                        (yy >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (yy <= y2.repeat_interleave(H*W).view(-1, H, W))
        # to avoid overlapped training by sup and unsup
        #have to be revised (flow_gt (x), map_gt (o))




        mask2D = src_bbox_mask & score_mask & trg_bbox_mask & occ_T_Svec.squeeze(1).bool()
        # mask2D = src_bbox_mask & trg_bbox_mask 
        
        # assert self.args.contrastive_gt_mask == False
        if self.args.contrastive_gt_mask:
            mask2D = mask2D & ~(flow_gt[:,0,:,:].bool()) & ~(flow_gt[:,1,:,:].bool())
            
        # Transformation 'aff', 'tps', 'afftps'  
        # same as xy_transformed      
        # x_transformed, y_transformed = torch.round(self.transform_by_grid(x_S_Tvec.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
        #                     interpolation_mode=self.args.interpolation_mode)).squeeze().long().clamp(0, 15), \
        #                             torch.round(self.transform_by_grid(y_S_Tvec.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
        #                     interpolation_mode=self.args.interpolation_mode)).squeeze().long().clamp(0, 15)
        # index_weak = y_transformed * W + x_transformed


        # xy = torch.cat((x_S_Tvec.unsqueeze(1).float(),y_S_Tvec.unsqueeze(1).float()),1)
        # xy_transformed = self.transform_by_grid(xy, mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
        #                     interpolation_mode=self.args.interpolation_mode).long().clamp(0, 15)
        # index_weak = xy_transformed[:,1,:,:] * W + xy_transformed[:,0,:,:] 


        trg_bbox_mask_transformed = torch.round(self.transform_by_grid(trg_bbox_mask.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode)).squeeze().bool()
        occ_T_Svec_transformed = torch.round(self.transform_by_grid(occ_T_Svec.float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode)).squeeze().bool()

        mask2D_transformed = src_bbox_mask_transformed & score_mask_transformed & trg_bbox_mask_transformed & occ_T_Svec_transformed
        # mask2D_transformed = src_bbox_mask_transformed & trg_bbox_mask_transformed

        # uncertainty = self.transform_by_grid(uncertainty.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
        #                                 interpolation_mode=self.args.interpolation_mode).squeeze()
        

        #strong#
        corr_strong = corr_strong.view(B, corrdim, H, W)
        corr_strong_prob = self.softmax_with_temperature(corr_strong, self.args.semi_softmax_corr_temp)
        score_strong, index_strong = torch.max(corr_strong_prob, dim=1)
        
        #margin
        min_margin = torch.tensor([2]).cuda().long()                        
        max_margin = torch.tensor([14]).cuda().long()                                
        x_s, y_s = (index_strong % W), (index_strong // W)
        strong_margin_mask = (x_s >= min_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                (x_s <= max_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                (y_s >= min_margin.repeat_interleave(H*W).view(-1, H, W)) & \
                (y_s <= max_margin.repeat_interleave(H*W).view(-1, H, W))


        if it % 10000 == 1:
            if mask2D[0].sum() == 0:
                pass
            else:
                # Visualization Variable(weak_not_T)#
                mask_tgt_kp2D_weak = (mask2D[0] == True).nonzero(as_tuple=False).transpose(-1,-2)
                #type1 (by_softArgmax)
                # mask_tgt_kp2D_weak_not_T = pred_map_weak_S_Tvec[0].permute(1,2,0)[mask2D_not_T[0] == True]
                # mask_tgt_kp2D_weak_not_T = mask_tgt_kp2D_weak_not_T.transpose(-1,-2)
                # print("mask_src_kp2D_weak_not_T", mask_src_kp2D_weak_not_T)
                # print("mask_tgt_kp2D_weak_not_T", mask_tgt_kp2D_weak_not_T)
                #type2 (by_Argmax)
                mask_src_1D = index_weak[0][mask2D[0]]
                mask_src_kp2D_weak = torch.cat(((mask_src_1D // W).view(1,-1), (mask_src_1D % W).view(1,-1)), dim=0)
                # Visualization Variable(weak_T)#        
                # mask_src_kp2D_weak = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_tgt_kp2D_weak_transformed = (mask2D_transformed[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D_transformed = index_weak_transformed[0][mask2D_transformed[0]]
                mask_src_kp2D_weak_transformed = torch.cat(((mask_src_1D_transformed // W).view(1,-1), (mask_src_1D_transformed % W).view(1,-1)), dim=0)
                
                # Visualization Variable(strong_T)        
                mask_tgt_kp2D_strong = (mask2D_transformed[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D = index_strong[0][mask2D_transformed[0]]
                mask_src_kp2D_strong = torch.cat(((mask_src_1D //  W).view(1,-1), (mask_src_1D % W).view(1,-1)), dim=0)
                weak_transformed = self.transform_by_grid(
                                    mini_batch['trg_img_weak'][0].unsqueeze(0).to(self.device),
                                    mini_batch[self.args.aug_mode][0].unsqueeze(0).to(self.device), 
                                    mode=self.args.aug_mode)
                diff_point = 0.0
                diff_idx = []
                for idx, (weak_pt, strong_pt) in enumerate(zip(mask_src_kp2D_weak_transformed.permute(-1,-2), mask_src_kp2D_strong.permute(-1,-2))):  # weak_pt strong_pt (x,y)
                    dist = torch.sqrt((weak_pt[0] - strong_pt[0]).pow(2) + (weak_pt[1] - strong_pt[1]).pow(2))
                    if dist >= 1:
                        diff_point += 1.0
                        diff_idx.append(idx)
                diff_idx = torch.tensor(diff_idx)
                if diff_point != 0:
                    diff_point /= mask_tgt_kp2D_strong.size(1)
                visualize(mini_batch,
                          mask_src_kp2D_weak,
                          mask_tgt_kp2D_weak,
                          mask_src_kp2D_weak_transformed,
                          mask_tgt_kp2D_weak_transformed,
                          mask_src_kp2D_strong,
                          mask_tgt_kp2D_strong,
                          weak_transformed,
                          self.device, self.args, n_iter, diff_idx)




        sup_loss_weight = None
        B, Svec, T_h, T_w = corr_strong.size()
        semi_loss_weight = torch.ones(B*T_h*T_w)


        
        # Semi_Loss #
        if self.args.loss_mode == 'contrastive':
            #margin
            # mask2D_transformed = mask2D_transformed & strong_margin_mask
            B, Svec, T_h, T_w = corr_strong.size()
            masked_corr_strong = corr_strong.permute(0,2,3,1).reshape(B*T_h *T_w, Svec)
            masked_corr_strong = masked_corr_strong[mask2D_transformed.view(-1)]
            masked_uncertainty = uncertainty_transformed.view(-1)[mask2D_transformed.view(-1)]
            masked_semi_loss_weight = semi_loss_weight[mask2D_transformed.view(-1)]
            
            masked_index_weak_transformed = index_weak_transformed[mask2D_transformed].long()

            masked_num = masked_index_weak_transformed.size(0)

            if not masked_num == 0:
                mask_pixelCT = torch.zeros_like(masked_corr_strong).bool()
                #masked_pixelCT : (B*T_h*T_w, SveC)
                mask_pixelCT[torch.arange(masked_num), masked_index_weak_transformed] = True
                positive = masked_corr_strong[mask_pixelCT].view(masked_num, -1)
                negative = masked_corr_strong[~mask_pixelCT].view(masked_num, -1)
                masked_pred = torch.cat([positive, negative], dim=1)

                masked_labels = torch.zeros(int(masked_num), device=self.device, dtype=torch.int64)
                masked_labels = masked_labels.detach()

                eps_temp = 1e-6
                masked_pred_with_temp = (masked_pred / self.args.semi_contrastive_temp) + eps_temp
                # loss_unsup = self.criterion(masked_pred_with_temp, masked_labels) * self.args.semi_lambda
                if self.args.use_uncertainty :
                    self.criterion_uncertainty =  nn.CrossEntropyLoss(reduction = 'none')
                    # import pdb
                    # pdb.set_trace()
                    # unc = (-1)*torch.sigmoid((masked_uncertainty-torch.ones_like(masked_uncertainty))/0.3)+1
                    unc = 1 / torch.exp(self.args.uncertainty_lamda * masked_uncertainty)
                    loss_unsup = (self.criterion_uncertainty(masked_pred_with_temp, masked_labels)*unc.detach()* masked_semi_loss_weight.cuda() ).mean() * self.args.semi_lambda
                else :
                    loss_unsup = self.criterion(masked_pred_with_temp, masked_labels) * self.args.semi_lambda
                
                if loss_unsup.isnan() :
                    # import pdb
                    print("NAN UNSUP LOSS!!")
                    # pdb.set_trace()
                    loss_unsup = torch.tensor(0.0, device=self.device)

                # loss_unsup = self.args.semi_lambda * ce_loss_wUncertainty(masked_pred_with_temp, masked_labels, uncertainty=masked_uncertainty, lam=1.0)
            else:
                loss_unsup = torch.tensor(0.0, device=self.device)



        #self_loss#
        loss_self = torch.tensor(0.0, device=self.device)
        #me_max loss#
        rloss = torch.tensor(0.0, device=self.device)

        
        diff_ratio = (~(index_weak_transformed[mask2D_transformed] == index_strong[mask2D_transformed])).sum() / masked_num  # 
            
        
        if self.sparse_exp :
            # import pdb
            
            sparse_gt_kps = mini_batch['use']
            flow_gt *= sparse_gt_kps
            
        if self.args.additional_weak:
            loss_sup = EPE(pred_flow_weak_gt, flow_gt, weight=sup_loss_weight)
        else:
            loss_sup = EPE(pred_flow_weak, flow_gt, weight=sup_loss_weight)

        # if epoch > self.args.warmup_epoch:
        #     Loss = loss_sup + loss_cunsup + loss_self + diff_ratio

        # else:
        #     Loss = loss_sup
        #     loss_unsup = torch.tensor(0.0, device=self.device)
        
        return loss_sup, loss_unsup, loss_self, diff_ratio
        
