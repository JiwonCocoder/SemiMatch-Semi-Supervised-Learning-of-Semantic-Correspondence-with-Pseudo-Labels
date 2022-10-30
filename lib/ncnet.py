import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

# from .CATs.dataset.dataset_utils import TpsGridGen
from .vis import unnormalise_and_convert_mapping_to_flow
from .semimatch_utils import visualize
from .semimatch_loss import EPE, ce_loss, consistency_loss
from .utils import flow2kps
from .evaluation import Evaluator
sys.path.append('.')
from MMNet.data.dataset_utils import TpsGridGen

class NCNet(nn.Module):
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
    
    def cutout_aware(self, image, kpoint, mask_size_min=3, mask_size_max=15, p=0.2, cutout_inside=True, mask_color=(0, 0, 0),
                    cut_n=10, batch_size=2, bbox_trg=None, n_pts=None):
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
                mask_size = torch.randint(low=int(max_mask_size * 0.03), high=int(max_mask_size * 0.1), size=(1,))
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
                corr_weak:torch.Tensor=None, 
                pred_map_weak:torch.Tensor=None,
                corr_strong:torch.Tensor=None,
                pred_map_strong:torch.Tensor=None,
                pred_map_self:torch.Tensor=None,
                corr_self:torch.Tensor=None,
                occ_S_Tvec:torch.Tensor=None,
                occ_T_Svec:torch.Tensor=None,
                epoch=None, n_iter=None, it=None):
        
        if self.args.additional_weak:
            pred_map_weak_gt, pred_map_weak = pred_map_weak
            pred_flow_weak_gt = unnormalise_and_convert_mapping_to_flow(pred_map_weak_gt)
            
        if len(corr_weak.shape) == 3:   # CATs
            mode = 'CATs'
            flow_gt = mini_batch['flow'].to(self.device)
            B, corrdim, H, W =\
                corr_weak.size(0), self.args.feature_size * self.args.feature_size,\
                             self.args.feature_size, self.args.feature_size
            self.args.feature_width, self.args.feature_height = self.args.feature_size, self.args.feature_size
        elif len(corr_weak.shape) == 5: # MMNet
            mode = 'MMNet'
            B, corrdim, H, W =\
                corr_weak.size(0), corr_weak.size(1) * corr_weak.size(2), corr_weak.size(3), corr_weak.size(4)
        else:
            raise ValueError(f'Correlation size {corr_weak.shape} is not defined')

        pred_flow_weak = unnormalise_and_convert_mapping_to_flow(pred_map_weak)
        pred_flow_strong = unnormalise_and_convert_mapping_to_flow(pred_map_strong)

        if self.args.refined_corr_filtering == 'dual_softmax':
            corr_weak_prob = F.softmax(corr_weak, dim=1) * F.softmax(corr_weak, dim=2)
            corr_weak_prob = corr_weak_prob.view(-1, corrdim, W, H)
            score_weak, index_weak = torch.max(corr_weak_prob, dim=1)
            x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')  
        else:
            corr_weak = corr_weak.view(B, corrdim, H, W)
            corr_weak_prob = self.softmax_with_temperature(corr_weak, self.args.semi_softmax_corr_temp)

            if self.args.refined_corr_filtering == 'mutual':
                score_weak, index_weak = torch.max(corr_weak_prob, dim=1)
                x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')

            elif self.args.refined_corr_filtering == 'soft_argmax':
                pred_map_weak = self.convert_unNormFlow_to_unNormMap(pred_flow_weak)
                index_weak = pred_map_weak[:,1,:,:]*W + pred_map_weak[:,0,:,:]
                index_weak = torch.round(index_weak)
                score_weak, _ = torch.max(corr_weak_prob, dim=1)
                

                x, y = pred_map_weak[:,0,:,:], pred_map_weak[:,1,:,:]
        # entropy
        uncertainty = (-(corr_weak_prob+1e-6) * torch.log(corr_weak_prob+1e-6)).sum(dim=1)
        
        if self.args.use_uncertainty :
            score_mask = score_weak.ge(0.5)
        else :
            score_mask = score_weak.ge(self.args.p_cutoff)

        bbox = mini_batch['src_bbox'].cuda()
        x1, y1, x2, y2 = torch.round(bbox[:,0] / W).long(),\
                         torch.round(bbox[:,1] / H).long(),\
                         torch.round(bbox[:,2] / W).long(),\
                         torch.round(bbox[:,3] / H).long()
                         
        src_bbox_mask = (x >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (x <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                        (y <= y2.repeat_interleave(H*W).view(-1, H, W))

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

        mask2D_not_T = score_mask & src_bbox_mask & trg_bbox_mask
        if self.args.contrastive_gt_mask:
            mask2D_not_T = mask2D_not_T & ~(flow_gt[:,0,:,:].bool()) & ~(flow_gt[:,1,:,:].bool())

        if self.args.use_fbcheck_mask:
            assert occ_S_Tvec is not None
            mask2D_not_T = mask2D_not_T & occ_T_Svec.squeeze(1).bool()
            fb_mask = torch.round(self.transform_by_grid(occ_T_Svec.float().cuda(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode)).squeeze().bool()

        # Transformation 'aff', 'tps', 'afftps'
        x_transformed, y_transformed = torch.round(self.transform_by_grid(x.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                              interpolation_mode=self.args.interpolation_mode)).squeeze().long().clamp(0, 255), \
                                       torch.round(self.transform_by_grid(y.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                              interpolation_mode=self.args.interpolation_mode)).squeeze().long().clamp(0, 255)
        index_weak = y_transformed * W + x_transformed
        mask_2D = torch.round(self.transform_by_grid(mask2D_not_T.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode)).squeeze().bool()

        uncertainty = self.transform_by_grid(uncertainty.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                        interpolation_mode=self.args.interpolation_mode).squeeze()
        

        #strong#
        corr_strong = corr_strong.view(B, corrdim, H, W)
        corr_strong_prob = self.softmax_with_temperature(corr_strong, self.args.semi_softmax_corr_temp)
        score_strong, index_strong = torch.max(corr_strong_prob, dim=1)

        sup_loss_weight = None
        B, Svec, T_h, T_w = corr_strong.size()
        semi_loss_weight = torch.ones(B*T_h*T_w)

        if mode == 'CATs' :
            estimated_kps = flow2kps(mini_batch['trg_kps'].to(self.device),
                                     pred_flow_weak,
                                     mini_batch['n_pts'].to(self.device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            
            if self.args.use_class_aware_sup :

                pck = torch.tensor(eval_result['pck']) / 100
                cat = F.one_hot(mini_batch['category_id'], num_classes=self.class_total.shape[0])
                self.class_pcksum += cat.mul(pck.unsqueeze(1)).sum(dim=0)
                self.class_total += torch.bincount(mini_batch['category_id'], minlength=self.class_total.shape[0])
                class_total_ = self.class_total.clone()
                class_total_[class_total_==0] = 1
                avg_pck_perclass = self.class_pcksum / class_total_
                avg_pck = avg_pck_perclass[avg_pck_perclass!=0].mean()
                sup_loss_weight = (3-avg_pck_perclass*2)[mini_batch['category_id']].detach()
                if it % 2000 == 0 :
                    print("sup_loss_weight :")
                    print(3-2*avg_pck_perclass)

                # for semiloss
                semi_loss_weight = sup_loss_weight.repeat_interleave(T_h*T_w)
        
        # Semi_Loss #

        if self.args.loss_mode == 'contrastive':
            B, Svec, T_h, T_w = corr_strong.size()
            masked_corr_strong = corr_strong.permute(0,2,3,1).reshape(B*T_h *T_w, Svec)
            masked_corr_strong = masked_corr_strong[mask_2D.view(-1)]
            masked_uncertainty = uncertainty.view(-1)[mask_2D.view(-1)]
            masked_semi_loss_weight = semi_loss_weight[mask_2D.view(-1)]

            masked_index_weak = index_weak[mask_2D].long()
            masked_num = masked_index_weak.size(0)

            if not masked_num == 0:
                mask_pixelCT = torch.zeros_like(masked_corr_strong).bool()
                mask_pixelCT[torch.arange(masked_num), masked_index_weak] = True
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
                    import pdb
                    print("NAN UNSUP LOSS!!")
                    pdb.set_trace()
                    loss_unsup = torch.tensor(0.0, device=self.device)

                # loss_unsup = self.args.semi_lambda * ce_loss_wUncertainty(masked_pred_with_temp, masked_labels, uncertainty=masked_uncertainty, lam=1.0)
            else:
                loss_unsup = torch.tensor(0.0, device=self.device)

        elif self.args.loss_mode == 'EPE':
            pred_weak_transformed_map = torch.cat([x_transformed.unsqueeze(1), y_transformed.unsqueeze(1)], dim=1)
            pred_weak_transformed_flow = unnormalise_and_convert_mapping_to_flow(pred_weak_transformed_map)
            if not mask_2D.sum() ==0:
            #(type_1) affined_by_map
                loss_unsup = EPE(pred_flow_strong, pred_weak_transformed_flow.detach(), mask=(mask_2D)) * self.args.semi_lambda
            else:
                loss_unsup = torch.tensor(0.0, device=self.device)

        # self_loss#
        if self.args.use_self_loss:
            # semi_mask_2D = torch.round(self.transform_by_grid(trg_bbox_mask.unsqueeze(1).float(), mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
            #                             interpolation_mode=self.args.interpolation_mode)).squeeze().bool()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).cuda().float()
            grid[:,0,:,:] = 2 * grid[:,0,:,:].clone() / (W-1) -1
            grid[:, 1, :, :] = 2 * grid[:, 1, :, :].clone() / (H - 1) - 1


            semi_mask = torch.ones(B,2,H,W).to(self.device)
            semi_mask = self.transform_by_grid(semi_mask, mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                            interpolation_mode=self.args.interpolation_mode)
            semi_mask = torch.round(semi_mask[:,0,:,:]).bool()

            self_map_gt = self.transform_by_grid(grid, mini_batch[self.args.aug_mode].to(self.device), mode=self.args.aug_mode,
                                            interpolation_mode=self.args.interpolation_mode)

            self_flow_gt = unnormalise_and_convert_mapping_to_flow(self_map_gt)
            pred_self_flow = unnormalise_and_convert_mapping_to_flow(pred_map_self)
            loss_self = EPE(pred_self_flow, self_flow_gt.detach(),mask=semi_mask) * self.args.self_lambda
            if it % 100 == 1:
                # vis#
                self.plot_NormMap2D_warped_Img(mini_batch['trg_img_weak'].to(self.device)[0].unsqueeze(0), mini_batch['trg_img_strong'].to(self.device)[0].unsqueeze(0),
                                            pred_map_self[0].unsqueeze(0), self_map_gt[0].unsqueeze(0),
                                            W, H,
                                            plot_name="warped_self_{}".format(n_iter),
                                            self_img= True)
                self.count += 1

        else:
            loss_self = torch.tensor(0.0, device=self.device)
        #me_max loss#
        if self.args.use_me_max_loss:
            avg_prob = torch.mean(corr_weak_prob.view(B, corrdim, corrdim), dim=2)
            rloss = -1.0 * torch.sum(torch.log(avg_prob**(-avg_prob)), dim=1).mean()
            rloss= rloss
        else:
            rloss = torch.tensor(0.0, device=self.device)

        if it % 100 == 1:
            if mask_2D[0].sum() == 0:
                pass
            else:
                # Visualization Variable(weak_not_T)
                mask_tgt_kp2D_weak_not_T = (mask2D_not_T[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D = index_weak[0][mask2D_not_T[0]]
                mask_src_kp2D_weak_not_T = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

                # Visualization Variable(weak_T)        
                mask_tgt_kp2D_weak = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D = index_weak[0][mask_2D[0]]
                mask_src_kp2D_weak = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)
                
                # Visualization Variable(strong_T)        
                mask_tgt_kp2D_strong = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D = index_strong[0][mask_2D[0]]
                mask_src_kp2D_strong = torch.cat(((mask_src_1D %  W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)
                weak_transformed = self.transform_by_grid(
                                    mini_batch['trg_img_weak'][0].unsqueeze(0).to(self.device),
                                    mini_batch[self.args.aug_mode][0].unsqueeze(0).to(self.device), 
                                    mode=self.args.aug_mode)

                visualize(mini_batch,
                          mask_src_kp2D_weak_not_T,
                          mask_tgt_kp2D_weak_not_T,
                          mask_src_kp2D_weak,
                          mask_tgt_kp2D_weak,
                          mask_src_kp2D_strong,
                          mask_tgt_kp2D_strong,
                          weak_transformed,
                          self.device, self.args, n_iter)
        if mode == 'CATs':
            if self.sparse_exp :
                sparse_gt_kps = mini_batch['use']
                pred_flow_weak_gt = pred_flow_weak_gt[mini_batch['use'].bool()]
                flow_gt = flow_gt[mini_batch['use'].bool()]

                    

            if flow_gt.shape[0] == 0 :
                loss_sup = torch.tensor(0.0, device=self.device)
            else :
                if self.args.additional_weak:
                    loss_sup = EPE(pred_flow_weak_gt, flow_gt, weight=sup_loss_weight)
                else:
                    loss_sup = EPE(pred_flow_weak, flow_gt, weight=sup_loss_weight)

            if epoch > self.args.warmup_epoch:
                Loss = loss_sup + loss_unsup + loss_self + rloss

            else:
                Loss = loss_sup
                loss_unsup = torch.tensor(0.0, device=self.device)

            return Loss, loss_sup, loss_unsup, loss_self, rloss
        
        elif mode == 'MMNet':
            return loss_unsup, loss_self, rloss