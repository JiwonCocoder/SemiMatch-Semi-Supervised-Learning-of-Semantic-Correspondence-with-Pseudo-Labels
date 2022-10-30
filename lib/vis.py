import os
from os import path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import pdb
# from .CATs.dataset.dataset_utils import TpsGridGen
# from .vis import unnormalise_and_convert_mapping_to_flow
# from .semimatch_utils import visualize
# from .semimatch_loss import EPE, ce_loss, consistency_loss
# from .utils import flow2kps
# from .evaluation import Evaluator
sys.path.append('.')

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
    
def plot_NormMap2D_warped_Img(src_img, tgt_img,
                                #   norm_map2D_S_Tvec, norm_map2D_T_Svec,
                                #   scale_factor,
                                  occ_S_Tvec=None, occ_T_Svec=None, save_path = '../test', plot_name=None, self_img=False):
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        tgt_img = tgt_img.mul(std).add(mean)
        src_img = src_img.mul(std).add(mean)
        # norm_map2D_S_Tvec = F.interpolate(input=norm_map2D_S_Tvec, scale_factor=scale_factor, mode='bilinear',
        #                                   align_corners=True)
        # norm_map2D_T_Svec = F.interpolate(input=norm_map2D_T_Svec, scale_factor=scale_factor, mode='bilinear',
        #                                   align_corners=True)
        # if self_img:
        #     masked_warp_S_Tvec = self.warp_from_NormMap2D(src_img, norm_map2D_S_Tvec)  # (B, 2, H, W)
        # else:
        #     masked_warp_S_Tvec = self.warp_from_NormMap2D(tgt_img, norm_map2D_S_Tvec)  # (B, 2, H, W)

        # masked_warp_T_Svec = self.warp_from_NormMap2D(src_img, norm_map2D_T_Svec)
        # if occ_S_Tvec is not None and occ_T_Svec is not None:
        #     mask_img_S_Tvec = F.interpolate(input=occ_S_Tvec.type(torch.float),
        #                                     scale_factor=scale_factor,
        #                                     mode='bilinear',
        #                                     align_corners=True)
        #     mask_img_T_Svec = F.interpolate(input=occ_T_Svec.type(torch.float),
        #                                     scale_factor=scale_factor,
        #                                     mode='bilinear',
        #                                     align_corners=True)
        #     masked_warp_T_Svec = mask_img_T_Svec * masked_warp_T_Svec
        #     masked_warp_S_Tvec = mask_img_S_Tvec * masked_warp_S_Tvec
        tgt_img = tgt_img * 255.0
        src_img = src_img * 255.0
        tgt_img = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
        src_img = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

        # masked_warp_T_Svec = masked_warp_T_Svec.data.squeeze(0).transpose(0, 1).transpose(1,
        #                                                                                   2).cpu().numpy()
        # masked_warp_S_Tvec = masked_warp_S_Tvec.data.squeeze(0).transpose(0, 1).transpose(1,
        #                                                                                   2).cpu().numpy()

        fig, axis = plt.subplots(1, 2, figsize=(50, 50))
        axis[0].imshow(src_img)
        axis[0].set_title("src_img")
        axis[1].imshow(tgt_img)
        axis[1].set_title("tgt_img")
        # axis[1][0].imshow(masked_warp_S_Tvec)
        # axis[1][0].set_title("warp_T_to_S_" + str(self.count))
        # axis[1][1].imshow(masked_warp_T_Svec)
        # axis[1][1].set_title("warp_S_to_T_" + str(self.count))
        if not osp.isdir(save_path):
            os.makedirs(save_path)
        fig.savefig('{}/{}.png'.format(save_path, plot_name),
                    bbox_inches='tight')
        plt.close(fig)