import os
import sys
from operator import add
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models

from models.feature_backbones import resnet
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pdb
import cv2
r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape
        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        x = x.flatten(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, H, N, W).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, W)
        return x


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, embed_dim // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, img_size ** 2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, source, target):
        B = corr.shape[0]
        x = corr.clone()
        
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4)
        pos_embed = pos_embed.flatten(2, 3)

        x = torch.cat((x.transpose(-1, -2), target), dim=3) + pos_embed
        x = self.proj(self.blocks(x)).transpose(-1, -2) + corr  # swapping the axis for swapping self-attention.

        x = torch.cat((x, source), dim=3) + pos_embed
        x = self.proj(self.blocks(x)) + corr 

        return x.mean(1)


class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
    
    
    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super().__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        return correlation_tensor
class CATs(nn.Module):
    def __init__(self,
    feature_size=16,
    feature_proj_dim=128,
    depth=4,
    num_heads=6,
    mlp_ratio=4,
    hyperpixel_ids=[0,8,20,21,26,28,29,30],
    freeze=True,
    softmax_corr_temp=0.1,
    args=None):
        super().__init__()
        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        
        channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3

        self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.proj = nn.ModuleList([
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))
            
        self.l2norm = FeatureL2Norm()
        self.corr_jw = FeatureCorrelation(shape='4D', normalization=False)

        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
        self.count=0
        self.softmax_corr_temp = softmax_corr_temp
        self.args = args
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
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
    def plot_during_training_mask(self, save_path, count, src_img, tgt_img, scale_factor, flow_S_Tvec, flow_T_Svec,
                                  mask_S_Tvec, mask_T_Svec,
                             plot_name= None, h= 256, w=256):
        #interpolate
        flow_S_Tvec = F.interpolate(flow_S_Tvec, (h, w), mode='bilinear', align_corners=False)
        flow_T_Svec = F.interpolate(flow_T_Svec, (h, w), mode='bilinear', align_corners=False)
        flow_S_Tvec_x = scale_factor * flow_S_Tvec.detach().permute(0, 2, 3, 1)[0, :, :, 0]
        flow_S_Tvec_y = scale_factor * flow_S_Tvec.detach().permute(0, 2, 3, 1)[0, :, :, 1]
        flow_T_Svec_x = scale_factor * flow_T_Svec.detach().permute(0, 2, 3, 1)[0, :, :, 0]
        flow_T_Svec_y = scale_factor * flow_T_Svec.detach().permute(0, 2, 3, 1)[0, :, :, 1]
        assert flow_S_Tvec.shape == flow_T_Svec.shape

        mask_img_S_Tvec = F.interpolate(input=mask_S_Tvec.type(torch.float),
                                        scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=True)
        mask_img_T_Svec = F.interpolate(input=mask_T_Svec.type(torch.float),
                                        scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=True)


        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        tgt_img = tgt_img.mul(std).add(mean)
        src_img = src_img.mul(std).add(mean)
        tgt_img = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        src_img = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()


        #(content) src, (style) tgt
        remapped_tgt_img = self.remap_using_flow_fields(tgt_img,
                                                        flow_S_Tvec_x.cpu().numpy(),
                                                        flow_S_Tvec_y.cpu().numpy())
        remapped_src_img = self.remap_using_flow_fields(src_img,
                                                        flow_T_Svec_x.cpu().numpy(),
                                                        flow_T_Svec_y.cpu().numpy())

        #s
        remapped_src_img = mask_img_T_Svec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy() * remapped_src_img
        remapped_tgt_img = mask_img_S_Tvec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy() * remapped_tgt_img
        fig, axis = plt.subplots(2, 2, figsize=(50, 50))
        axis[0][0].imshow(src_img)
        axis[0][0].set_title("src_img_" + str(self.count))
        axis[0][1].imshow(tgt_img)
        axis[0][1].set_title("tgt_img" + str(self.count))
        axis[1][0].imshow(remapped_tgt_img)
        axis[1][0].set_title("remapped_tgt_img" + str(self.count))
        axis[1][1].imshow(remapped_src_img)
        axis[1][1].set_title("remapped_src_img" + str(self.count))
        fig.savefig('{}/{}_{}.png'.format(save_path, plot_name, count),
                    bbox_inches='tight')
        plt.close(fig)
    def plot_during_training(self, save_path, count, src_img, tgt_img, scale_factor, flow_S_Tvec, flow_T_Svec,
                             plot_name= None, h= 256, w=256):
        #interpolate
        flow_S_Tvec = F.interpolate(flow_S_Tvec, (h, w), mode='bilinear', align_corners=False)
        flow_T_Svec = F.interpolate(flow_T_Svec, (h, w), mode='bilinear', align_corners=False)
        flow_S_Tvec_x = scale_factor * flow_S_Tvec.detach().permute(0, 2, 3, 1)[0, :, :, 0]
        flow_S_Tvec_y = scale_factor * flow_S_Tvec.detach().permute(0, 2, 3, 1)[0, :, :, 1]
        flow_T_Svec_x = scale_factor * flow_T_Svec.detach().permute(0, 2, 3, 1)[0, :, :, 0]
        flow_T_Svec_y = scale_factor * flow_T_Svec.detach().permute(0, 2, 3, 1)[0, :, :, 1]
        assert flow_S_Tvec.shape == flow_T_Svec.shape
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        tgt_img = tgt_img.mul(std).add(mean)
        src_img = src_img.mul(std).add(mean)
        tgt_img = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        src_img = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        #(content) src, (style) tgt
        remapped_tgt_img = self.remap_using_flow_fields(src_img,
                                                        flow_S_Tvec_x.cpu().numpy(),
                                                        flow_S_Tvec_y.cpu().numpy())
        remapped_src_img = self.remap_using_flow_fields(src_img,
                                                        flow_T_Svec_x.cpu().numpy(),
                                                        flow_T_Svec_y.cpu().numpy())

        fig, axis = plt.subplots(2, 2, figsize=(50, 50))
        axis[0][0].imshow(src_img)
        axis[0][0].set_title("src_img_" + str(self.count))
        axis[0][1].imshow(tgt_img)
        axis[0][1].set_title("tgt_img" + str(self.count))
        axis[1][0].imshow(remapped_tgt_img)
        axis[1][0].set_title("remapped_tgt_img" + str(self.count))
        axis[1][1].imshow(remapped_src_img)
        axis[1][1].set_title("remapped_src_img" + str(self.count))
        fig.savefig('{}/{}_{}.png'.format(save_path, plot_name, count),
                    bbox_inches='tight')
        plt.close(fig)
    def remap_using_flow_fields(self, image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
        """
        opencv remap : carefull here mapx and mapy contains the index of the future position for each pixel
        not the displacement !
        map_x contains the index of the future horizontal position of each pixel [i,j] while map_y contains the index of the future y
        position of each pixel [i,j]
        All are numpy arrays
        :param image: image to remap, HxWxC
        :param disp_x: displacement on the horizontal direction to apply to each pixel. must be float32. HxW
        :param disp_y: isplacement in the vertical direction to apply to each pixel. must be float32. HxW
        :return:
        remapped image. HxWxC
        """
        h_scale, w_scale = image.shape[:2]

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        map_x = (X + disp_x).astype(np.float32)
        map_y = (Y + disp_y).astype(np.float32)
        remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

        return remapped_image
    def plot_test_map_mask_img(self, tgt_img, src_img,
                               # index_2D_S_Tvec, index_2D_T_Svec,
                               norm_map2D_S_Tvec, norm_map2D_T_Svec,
                               # occ_S_Tvec, occ_T_Svec,
                               scale_factor):  # A_Bvec #B_Avec
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
        # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
        tgt_img = tgt_img.mul(std).add(mean)
        src_img = src_img.mul(std).add(mean)

        # _, h, w = index_2D_S_Tvec.size()
        # index1D_S_Tvec = index_2D_S_Tvec.view(1, -1)
        # norm_map2D_S_Tvec = self.unNormMap1D_to_NormMap2D(index1D_S_Tvec, h)
        #
        # index1D_T_Svec = index_2D_T_Svec.view(1, -1)
        # norm_map2D_T_Svec = self.unNormMap1D_to_NormMap2D(index1D_T_Svec, h)

        norm_map2D_S_Tvec = F.interpolate(input=norm_map2D_S_Tvec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)
        norm_map2D_T_Svec = F.interpolate(input=norm_map2D_T_Svec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)

        masked_warp_S_Tvec = self.warp_from_NormMap2D(tgt_img, norm_map2D_S_Tvec)  # (B, 2, H, W)

        masked_warp_T_Svec = self.warp_from_NormMap2D(src_img, norm_map2D_T_Svec)
        # if self.use_mask:
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

        masked_warp_T_Svec = masked_warp_T_Svec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
        masked_warp_S_Tvec = masked_warp_S_Tvec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        fig, axis = plt.subplots(2, 2, figsize=(50, 50))
        axis[0][0].imshow(tgt_img)
        axis[0][0].set_title("tgt_img_" + str(self.count))
        axis[0][1].imshow(src_img)
        axis[0][1].set_title("src_img_" + str(self.count))
        axis[1][0].imshow(masked_warp_T_Svec)
        axis[1][0].set_title("warp_S_Tvec_" + str(self.count))
        axis[1][1].imshow(masked_warp_S_Tvec)
        axis[1][1].set_title("warp_T_Svec_" + str(self.count))
        # plt.show()
        # if self.use_mask:
        #     del mask_img_S_Tvec, mask_img_T_Svec
        # del tgt_img, src_img, index1D_S_Tvec, index1D_T_Svec, norm_map2D_S_Tvec, norm_map2D_T_Svec, masked_warp_T_Svec, masked_warp_S_Tvec
        torch.cuda.empty_cache()
        return fig
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

    def unnormalise_and_convert_mapping_to_flow(self, map):
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
    def forward(self, target, source, vis_fbcheck=False):
        B, _, H, W = target.size()

        src_feats = self.feature_extraction(source)
        tgt_feats = self.feature_extraction(target)

        corrs = []
        src_feats_proj = []
        tgt_feats_proj = []
        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            corr = self.corr(self.l2norm(src), self.l2norm(tgt))
            corrs.append(corr)
            src_feats_proj.append(self.proj[i](src.flatten(2).transpose(-1, -2)))
            tgt_feats_proj.append(self.proj[i](tgt.flatten(2).transpose(-1, -2)))

        src_feats = torch.stack(src_feats_proj, dim=1)
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)
        corr = torch.stack(corrs, dim=1)
        #to check corr_direction#
        #(finish) corr: B, (HxW(src)), (HxW(tgt))
        # corr_4d_S_T_jw = self.corr_jw(self.l2norm(src), self.l2norm(tgt))
        # B, ch, f1, f2, f3, f4 = corr_4d_S_T_jw.size()
        # corr_3d_T_Svec_jw = corr_4d_S_T_jw.view(B,
        #                                      self.feature_size * self.feature_size,
        #                                      self.feature_size, self.feature_size)
        # corr_3d_S_Tvec_jw = corr_4d_S_T_jw.view(B,
        #                                     self.feature_size, self.feature_size,
        #                                     self.feature_size * self.feature_size).permute(0, 3, 1, 2)
        # corr_3d_T_Svec = corr.squeeze(1).view(B, -1, 16, 16)
        # corr_3d_S_Tvec = corr.transpose(-1,-2).squeeze(1).view(B, -1, 16, 16)
        # scores_T_Svec, index_T_Svec = torch.max(corr_3d_T_Svec, dim=1)
        # NormMap2D_T_Svec = self.unNormMap1D_to_NormMap2D(index_T_Svec.view(B,-1), 16)
        # unNormFlowMap2D_T_Svec = self.unnormalise_and_convert_mapping_to_flow(NormMap2D_T_Svec)
        #
        # scores_S_Tvec, index_S_Tvec = torch.max(corr_3d_S_Tvec, dim=1)
        # NormMap2D_S_Tvec = self.unNormMap1D_to_NormMap2D(index_S_Tvec.view(B,-1), 16)
        # unNormFlowMap2D_S_Tvec = self.unnormalise_and_convert_mapping_to_flow(NormMap2D_S_Tvec)
        corr = self.mutual_nn_filter(corr)

        # gen_normalized2D_map_T_Svec
        refined_corr = self.decoder(corr, src_feats, tgt_feats)
        grid_x, grid_y = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size), beta=self.args.semi_softmax_corr_temp)
        map = torch.cat((grid_x, grid_y), dim=1)
     
        # if self.args.refined_corr_filtering == 'mutual':
        #     refined_corr = self.mutual_nn_filter(refined_corr.unsqueeze(1)).squeeze()
        if self.args.use_fbcheck_mask:
            refined_T_Svec = self.softmax_with_temperature(refined_corr.view(B, -1, self.feature_size, self.feature_size),
                                                           beta=self.args.semi_softmax_corr_temp, d=1)
            refined_S_Tvec = self.softmax_with_temperature(refined_corr.transpose(-1, -2).view(B, -1, self.feature_size, self.feature_size),
                                                           beta=self.args.semi_softmax_corr_temp, d=1)
            _, index_T_Svec = torch.max(refined_T_Svec, dim=1)
            _, index_S_Tvec = torch.max(refined_S_Tvec, dim=1)
            NormMap2D_T_Svec, _ = self.unNormMap1D_to_NormMap2D_and_NormFlow2D(index_T_Svec.view(B,-1), self.args.feature_size, self.args.feature_size)
            NormMap2D_S_Tvec, _ = self.unNormMap1D_to_NormMap2D_and_NormFlow2D(index_S_Tvec.view(B,-1), self.args.feature_size, self.args.feature_size)
            #
            # occ_S_Tvec, occ_T_Svec = self.calOcc(NormFlow2D_S_Tvec, NormMap2D_S_Tvec,
            #                                      NormFlow2D_T_Svec, NormMap2D_T_Svec)
            unNormFlowMap2D_S_Tvec = self.unnormalise_and_convert_mapping_to_flow(NormMap2D_S_Tvec)
            unNormFlowMap2D_T_Svec = self.unnormalise_and_convert_mapping_to_flow(NormMap2D_T_Svec)
            
            occ_S_Tvec, occ_T_Svec = self.calOcc(unNormFlowMap2D_S_Tvec, NormMap2D_S_Tvec,
                                                 unNormFlowMap2D_T_Svec, NormMap2D_T_Svec)


            #vis#
            if vis_fbcheck and self.count % 50000 == 0:
                self.plot_NormMap2D_warped_Img(source[0].unsqueeze(0), target[0].unsqueeze(0),
                                          NormMap2D_S_Tvec[0].unsqueeze(0), NormMap2D_T_Svec[0].unsqueeze(0),
                                          self.args.feature_size,
                                          occ_S_Tvec[0].unsqueeze(0), occ_T_Svec[0].unsqueeze(0),
                                          plot_name = "warped_{}".format(self.count))
            self.count+=1
            return map, refined_corr, occ_S_Tvec, occ_T_Svec

        return map, refined_corr, None, None
