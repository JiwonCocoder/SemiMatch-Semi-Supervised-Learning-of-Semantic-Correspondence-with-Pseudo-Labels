r"""Superclass for semantic correspondence datasets"""
import os
import random
import pdb
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from .keypoint_to_flow import KeypointToFlow
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def resize(img, kps, size=(256, 256)):
    _, h, w = img.shape
    resized_img = torchvision.transforms.functional.resize(img, size)
    
    kps = kps.t()
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = kps[:, 0] * (size[1] / w)
    resized_kps[:, 1] = kps[:, 1] * (size[0] / h)
    
    return resized_img, resized_kps.t()

def random_crop(img, kps, bbox, size=(256, 256), p=0.5):
    if random.uniform(0, 1) > p:
        return resize(img, kps, size)
    _, h, w = img.shape
    kps = kps.t()
    left = random.randint(0, bbox[0])
    top = random.randint(0, bbox[1])
    height = random.randint(bbox[3], h) - top
    width = random.randint(bbox[2], w) - left
    resized_img = torchvision.transforms.functional.resized_crop(
        img, top, left, height, width, size=size)
    
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = (kps[:, 0] - left) * (size[1] / width)
    resized_kps[:, 1] = (kps[:, 1] - top) * (size[0] / height)
    resized_kps = torch.clamp(resized_kps, 0, size[0] - 1)
    
    return resized_img, resized_kps.t()

def compute_syn_theta(aug_mode, aug_aff_scaling, aug_tps_scaling):
    if aug_mode == 'aff' or aug_mode == 'afftps':
        rot_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 12;  # between -np.pi/12 and np.pi/12
        sh_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 6;  # between -np.pi/6 and np.pi/6
        lambda_1 = 1 + (2 * np.random.rand(1) - 1) * 0.25;  # between 0.75 and 1.25
        lambda_2 = 1 + (2 * np.random.rand(1) - 1) * 0.25;  # between 0.75 and 1.25
        tx = (2 * np.random.rand(1) - 1) * 0.25;  # between -0.25 and 0.25
        ty = (2 * np.random.rand(1) - 1) * 0.25;

        R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                         [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
        R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                            [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

        D = np.diag([lambda_1[0], lambda_2[0]])

        A = R_alpha @ R_sh.transpose() @ D @ R_sh
        theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
    if aug_mode == 'tps' or aug_mode == 'afftps':
        theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * aug_tps_scaling
        theta_tps = theta_tps.astype(np.float)
    if aug_mode == 'aff':
        return theta_aff
    elif aug_mode == 'tps':
        return theta_tps
    elif aug_mode == 'afftps':
        return np.concatenate((theta_aff, theta_tps))

def gen_affine():
    theta1 = np.zeros(9)
    theta1[0:6] = np.random.randn(6) * 0.15
    theta1 = theta1 + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    affine1 = np.reshape(theta1, (3, 3))
    affine_inverse1 = np.linalg.inv(affine1)
    affine1 = np.reshape(affine1, -1)[0:6]
    affine_inverse1 = np.reshape(affine_inverse1, -1)[0:6]
    affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
    affine_inverse1 = torch.from_numpy(affine_inverse1).type(torch.FloatTensor)
    return affine1, affine_inverse1

def gen_tps(random_t_tps=0.4):
    theta_tps = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
    theta_tps = theta_tps+(np.random.rand(18)-0.5)*2*random_t_tps
    return torch.from_numpy(theta_tps).type(torch.FloatTensor)
def gen_transform_list(param=0.2, strong_aug=False):
    if param != 0 and strong_aug == False:
        transform_list =A.Compose([
                    A.ToGray(p=param),
                    A.Posterize(p=param),
                    A.Equalize(p=param),
                    A.augmentations.transforms.Sharpen(p=param),
                    A.RandomBrightnessContrast(p=param),
                    A.Solarize(p=param),
                    A.ColorJitter(p=param),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2(),
                ])
    elif param != 0 and strong_aug == True:
        transform_list =A.Compose([
                    A.transforms.Blur(p=param),
                    A.ToGray(p=param),
                    A.Posterize(p=param),
                    A.Equalize(p=param),
                    A.RandomBrightnessContrast(p=param),
                    A.Solarize(p=param),
                    A.ColorJitter(p=param),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2(),
                ])

    else :
        transform_list =A.Compose([
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2(),
                ])
    return transform_list

class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair"""
    def __init__(self, benchmark, datapath, thres, device, split, 
                 augmentation, feature_size, aug_mode, aug_aff_scaling, 
                 aug_tps_scaling, aug_photo_weak, aug_photo_strong, aug_photo_source=0.2,
                 additional_weak=False):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech':  ('Caltech-101',
                         'test_pairs_caltech_with_category.csv',
                         '101_ObjectCategories',
                         '',
                         ''),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        if benchmark == 'caltech':
            self.max_pts = 400
        else:
            self.max_pts = 40
        self.split = split
        self.augmentation = augmentation
        self.device = device
        self.imside = 256
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        self.aug_photo_weak = aug_photo_weak
        self.aug_photo_strong = aug_photo_strong
        self.aug_photo_source = aug_photo_source
        self.additional_weak = additional_weak

        if split == 'trn' and augmentation:
            self.transfrom_src = gen_transform_list(self.aug_photo_source)
            self.transform_weak = gen_transform_list(self.aug_photo_weak)
            self.transform_strong = gen_transform_list(self.aug_photo_strong, strong_aug=True)
            self.transform_additional_weak = transforms.Compose([transforms.Resize((self.imside, self.imside)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.Resize((self.imside, self.imside)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []
        self.idx = []

        self.kps_to_flow = KeypointToFlow(receptive_field_size=35, jsz=256//feature_size, feat_size=feature_size, img_size=self.imside)

        self.aug_mode = aug_mode
        self.aug_aff_scaling = aug_aff_scaling
        self.aug_tps_scaling = aug_tps_scaling

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Constructs and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        # Image as tensor
        if self.split == 'trn' and self.augmentation:
            batch['src_img'] = self.transfrom_src(image=np.array(src_pil))['image']
            batch['trg_img_weak'] = self.transform_weak(image=np.array(trg_pil))['image']
            batch['trg_img_strong'] = self.transform_strong(image=np.array(trg_pil))['image']
            batch[self.aug_mode] = compute_syn_theta(self.aug_mode, self.aug_aff_scaling, self.aug_tps_scaling)
        else:
            batch['src_img'] = self.transform(src_pil)
            batch['trg_img'] = self.transform(trg_pil)

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size)
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)
        
        if self.additional_weak:
            batch['trg_additional_weak'] = self.transform_additional_weak(trg_pil)

        return batch

    def get_image(self, imnames, idx):
        r"""Reads PIL image from path"""
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def get_pckthres(self, batch, imsize):
        r"""Computes PCK threshold"""
        if self.thres == 'bbox':
            bbox = batch['src_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['src_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image with size of (240,240)"""
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1
        if self.split == 'trn' and self.augmentation:
            x_crds = pts_list[idx][0]
            y_crds = pts_list[idx][1]
        else:
            x_crds = pts_list[idx][0] * (self.imside / org_imsize[0])
            y_crds = pts_list[idx][1] * (self.imside / org_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts


def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t()
    _, nearest_idx = dist.min(dim=1)

    return nearest_idx

"""
https://github.com/PruneTruong/GLU-Net
"""
class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        # return torch.cat((points_X_prime, points_Y_prime), 3)
        return torch.cat((points_X_prime, points_Y_prime), 3).cuda()
