from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
import pickle
from lib.mod import FeatureL2Norm
import matplotlib.pyplot as plt

from lib.torch_util import Softmax1D
from lib.conv4d import Conv4d
import pdb
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='resnet101', feature_extraction_model_file='', normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn=feature_extraction_cnn
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        # for resnet below
        resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
        if feature_extraction_cnn=='resnet101':
            self.model = models.resnet101(pretrained=True)            
            if last_layer=='':
                last_layer = 'layer3'                            
            resnet_module_list = [getattr(self.model,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index(last_layer)
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if feature_extraction_cnn=='resnet101fpn':
            if feature_extraction_model_file!='':
                resnet = models.resnet101(pretrained=True) 
                # swap stride (2,2) and (1,1) in first layers (PyTorch ResNet is slightly different to caffe2 ResNet)
                # this is required for compatibility with caffe2 models
                resnet.layer2[0].conv1.stride=(2,2)
                resnet.layer2[0].conv2.stride=(1,1)
                resnet.layer3[0].conv1.stride=(2,2)
                resnet.layer3[0].conv2.stride=(1,1)
                resnet.layer4[0].conv1.stride=(2,2)
                resnet.layer4[0].conv2.stride=(1,1)
            else:
                resnet = models.resnet101(pretrained=True) 
            resnet_module_list = [getattr(resnet,l) for l in resnet_feature_layers]
            conv_body = nn.Sequential(*resnet_module_list)
            self.model = fpn_body(conv_body,
                                  resnet_feature_layers,
                                  fpn_layers=['layer1','layer2','layer3'],
                                  normalize=normalization,
                                  hypercols=True)
            if feature_extraction_model_file!='':
                self.model.load_pretrained_weights(feature_extraction_model_file)

        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if train_fe==False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        # if self.normalization and not self.feature_extraction_cnn=='resnet101fpn':
            # features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):        
        if self.shape=='3D':
            b,c,h,w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        elif self.shape=='4D':
            b,c,hA,wA = feature_A.size()
            b,c,hB,wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,hA*wA).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,hB*wB) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,hA,wA,hB,wB).unsqueeze(1)
        
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        return correlation_tensor

class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in,out_channels=ch_out,kernel_size=k_size,bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x)+self.conv(x.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
            # because of the ReLU layers in between linear layers, 
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x

def MutualMatching(corr4d):
    # mutual matching
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

    corr4d_B=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

    # get max
    corr4d_B_max,_=torch.max(corr4d_B,dim=1,keepdim=True)
    corr4d_A_max,_=torch.max(corr4d_A,dim=3,keepdim=True)

    eps = 1e-5
    corr4d_B=corr4d_B/(corr4d_B_max+eps)
    corr4d_A=corr4d_A/(corr4d_A_max+eps)

    corr4d_B=corr4d_B.view(batch_size,1,fs1,fs2,fs3,fs4)
    corr4d_A=corr4d_A.view(batch_size,1,fs1,fs2,fs3,fs4)

    corr4d=corr4d*(corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
        
    return corr4d

def maxpool4d(corr4d_hres,k_size=4):
    slices=[]
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:,0,i::k_size,j::k_size,k::k_size,l::k_size].unsqueeze(0))
    slices=torch.cat(tuple(slices),dim=1)
    corr4d,max_idx=torch.max(slices,dim=1,keepdim=True)
    max_l=torch.fmod(max_idx,k_size)
    max_k=torch.fmod(max_idx.sub(max_l).div(k_size),k_size)
    max_j=torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size),k_size)
    max_i=max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d,max_i,max_j,max_k,max_l)

class ImMatchNet(nn.Module):
    def __init__(self, 
                 feature_extraction_cnn='resnet101', 
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 return_correlation=False,  
                 ncons_kernel_sizes=[3,3,3],
                 ncons_channels=[10,10,1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 relocalization_k_size=0,
                 half_precision=False,
                 checkpoint=None,
                 args = None
                 ):
        
        super(ImMatchNet, self).__init__()
        # Load checkpoint
        if checkpoint is not None and checkpoint is not '':
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
            # override relevant parameters
            print('Using checkpoint parameters: ')
            ncons_channels=checkpoint['args'].ncons_channels
            print('  ncons_channels: '+str(ncons_channels))
            ncons_kernel_sizes=checkpoint['args'].ncons_kernel_sizes
            print('  ncons_kernel_sizes: '+str(ncons_kernel_sizes))            
        self.args = args
        self.feature_size = args.feature_size
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))        
        self.count=0
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision
        self.l2norm = FeatureL2Norm()      
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='4D',normalization=False)

        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)

        # Load weights
        if checkpoint is not None and checkpoint is not '':
            print('Copying weights...')
            for name, param in self.FeatureExtraction.state_dict().items():
                if 'num_batches_tracked' not in name:
                    self.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
            for name, param in self.NeighConsensus.state_dict().items():
                self.NeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['NeighConsensus.' + name])
            print('Done!')
        
        self.FeatureExtraction.eval()

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data=p.data.half()
            for l in self.NeighConsensus.conv:
                if isinstance(l,Conv4d):
                    l.use_half=True
                    
    # used only for foward pass at eval and for training with strong supervision
    # def forward(self, 
    #             mini_batch:dict, 
    #             corr_weak:torch.Tensor=None, 
    #             pred_map_weak:torch.Tensor=None,
    #             corr_strong:torch.Tensor=None,
    #             pred_map_strong:torch.Tensor=None,
    #             pred_map_self:torch.Tensor=None,
    #             corr_self:torch.Tensor=None,
    #             occ_S_Tvec:torch.Tensor=None,
    #             occ_T_Svec:torch.Tensor=None,
    #             epoch=None, n_iter=None, it=None):
    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
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
    def forward(self, target, source, vis_fbcheck=True, semi=True): 
        # feature extraction
        device = torch.cuda.current_device()
        src_feats = self.FeatureExtraction(source.cuda())
        tgt_feats = self.FeatureExtraction(target.cuda())
        if self.half_precision:
            src_feats=src_feats.half()
            tgt_feats=tgt_feats.half()
        B, C, f_h, f_w = tgt_feats.size()
        # feature correlation
        corr4d = self.FeatureCorrelation(src_feats,tgt_feats)
        # corr2d_cats = self.corr(self.l2norm(src_feats), self.l2norm(tgt_feats))
        # corr3d_T_Svec = corr2d_cats.view(B, -1, 16, 16)
        # do 4d maxpooling for relocalization
        if self.relocalization_k_size>1:
            corr4d,max_i,max_j,max_k,max_l=maxpool4d(corr4d,k_size=self.relocalization_k_size)
        # run match processing model 
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)            
        corr4d = MutualMatching(corr4d)
        
        if semi:
            grid_x, grid_y = self.soft_argmax(corr4d.view(B, -1, f_h, f_w), beta=self.args.semi_softmax_corr_temp)                                                                           
            map = torch.cat((grid_x, grid_y), dim=1)

            if self.args.use_fbcheck_mask:
                nc_T_Svec = corr4d.view(B, f_h*f_h, f_h, f_h)
                nc_S_Tvec = corr4d.view(B, f_h, f_h, f_h*f_h).permute(0,3,1,2)                

                corr_T_Svec = self.softmax_with_temperature(nc_T_Svec,
                                                            beta=self.args.semi_softmax_corr_temp, d=1)        
                corr_S_Tvec = self.softmax_with_temperature(nc_S_Tvec,
                                                            beta=self.args.semi_softmax_corr_temp, d=1)

                _, index_T_Svec = torch.max(corr_T_Svec, dim=1)
                _, index_S_Tvec = torch.max(corr_S_Tvec, dim=1)
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
                if vis_fbcheck and self.count % 200 == 0:
                    self.plot_NormMap2D_warped_Img(source[0].unsqueeze(0), target[0].unsqueeze(0),
                                            NormMap2D_S_Tvec[0].unsqueeze(0), NormMap2D_T_Svec[0].unsqueeze(0),
                                            self.args.feature_size,
                                            occ_S_Tvec[0].unsqueeze(0), occ_T_Svec[0].unsqueeze(0),
                                            plot_name = "warped_{}".format(self.count))
                    
                self.count +=1
                
                return map, corr4d, occ_S_Tvec, occ_T_Svec 
                                                                                                                              
            else:
                return featureL2Norm(corr4d)
        
        else: 
            return corr4d
        # if self.relocalization_k_size>1:
        #     delta4d=(max_i,max_j,max_k,max_l)
        #     return (corr4d,delta4d)
        # else:
        #     if semi:
        #         return
        #     else:     
        #         return corr4d

