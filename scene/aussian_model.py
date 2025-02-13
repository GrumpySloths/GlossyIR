#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from arguments import GroupParams
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from scene.embedding import Embedding
from torch_scatter import scatter_max
from functools import reduce

class GaussianModel:
    def setup_functions(self) -> None:
        def build_covariance_from_scaling_rotation(
            scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor
        ) -> torch.Tensor:
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.material_activation = torch.sigmoid
        #pbr属性添加
        self.normal_activation=torch.tanh #将法线贴图范围限制到[-1,1]

        self.rotation_activation = torch.nn.functional.normalize

    # def __init__(self, sh_degree: int) -> None:
    #     self.active_sh_degree = 0
    #     self.max_sh_degree = sh_degree
    #     self._xyz = torch.empty(0)
    #     self._features_dc = torch.empty(0)
    #     self._features_rest = torch.empty(0)
    #     self._scaling = torch.empty(0)
    #     self._rotation = torch.empty(0)
    #     self._opacity = torch.empty(0)
    #     self._normal = torch.empty(0)
    #     self._albedo = torch.empty(0)
    #     self._roughness = torch.empty(0)
    #     self._metallic = torch.empty(0)
    #     self.max_radii2D = torch.empty(0)
    #     self.xyz_gradient_accum = torch.empty(0)
    #     self.denom = torch.empty(0)
    #     self.optimizer = None
    #     self.percent_dense = 0
    #     self.spatial_lr_scale = 0
    #     self.setup_functions()
    def __init__(self, 
                 feat_dim: int=32, 
                 pbr_feature_dim: int=32, #pbr feature
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 0,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_pbr_dist : bool = False, #这里存一个疑问，需要添加pbr_dist吗？即这个距离设计真的很重要吗
                 ):

        self.active_sh_degree = 0
        self.max_sh_degree = 3 
        self.feat_dim = feat_dim #32
        #添加pbr特征，用于后续解耦normal，albedo，roughness，metallic
        self.pbr_feature_dim=pbr_feature_dim #32

        self.n_offsets = n_offsets #10
        self.voxel_size = voxel_size #0.001
        self.update_depth = update_depth  #3
        self.update_init_factor = update_init_factor #16
        self.update_hierachy_factor = update_hierachy_factor #4
        self.use_feat_bank = use_feat_bank  #false

        self.appearance_dim = appearance_dim #0
        self.embedding_appearance = None
        self.ratio = ratio   #1 
        self.add_opacity_dist = add_opacity_dist #false
        self.add_cov_dist = add_cov_dist #false
        self.add_color_dist = add_color_dist #false
        self.add_pbr_dist=add_pbr_dist #false

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)  #scaling维度为6,前3个表示offset的缩放，后3个表示scale的缩放,用于进一步调整每个neural gaussian
        self._rotation = torch.empty(0) #一个疑问就是这里的rotation是否真的是冗余的呢？要回头看看做一下消融实验了
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        #pbr 属性添加
        self._normal = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim), #3+1表示anchor和camera之间的视角信息和相对距离
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),  #这里的7表示的 四元数q 和scalar s,4+3
        ).cuda()
        
        #pbr属性添加
        self.pbr_dist_dim = 1 if self.add_pbr_dist else 0
        self.mlp_pbr = nn.Sequential(
            nn.Linear(pbr_feature_dim+3+self.pbr_dist_dim, pbr_feature_dim),
            nn.ReLU(True),
            nn.Linear(pbr_feature_dim, 8*self.n_offsets), #这里的8表示 3+3+1+1 [normal,albedo,roughness,metallic]
        ).cuda()
        
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(  #这里的appearance_dim表示的可能是可以扩展的视角依赖的球谐函数信息
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        #添加pbr属性
        self.mlp_pbr.eval()
        
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        #添加pbr属性
        self.mlp_pbr.train()
        
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
    
    def capture_anchor(self):
        return (
            self._anchor,
            self._offset,
            self._scaling,
            self._rotation,
            #pbr属性添加
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
            #pbr属性添加结束
            self._opacity,
            self.max_radii2D,
            self.anchor_demon,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore_anchor(self, model_args, training_args):
        ( 
        self._anchor, 
        self._offset,
        self._scaling, 
        self._rotation, 
        #pbr属性添加
        self._normal,
        self._albedo,
        self._roughness,
        self._metallic,
        #pbr属性添加结束
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup_anchor(training_args)
        self.anchor_demon = denom
        self.optimizer.load_state_dict(opt_dict)
           
    def capture(
        self,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.optim.Optimizer,
        float,
    ]:
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(
        self,
        model_args: Tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict,
            float,
        ],
        training_args: Optional[GroupParams] = None,
    ) -> None:
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
            
    @property
    def get_scaling(self) -> torch.Tensor:
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_normal(self) -> torch.Tensor:
        return F.normalize(self._normal, p=2, dim=-1)

    @property
    def get_albedo(self) -> torch.Tensor:
        return self.material_activation(self._albedo)

    @property
    def get_roughness(self) -> torch.Tensor:
        return self.material_activation(self._roughness)

    @property
    def get_metallic(self) -> torch.Tensor:
        return self.material_activation(self._metallic)

    #anchor property属性添加和扩展=========================================
    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_pbr_mlp(self):
        return self.mlp_pbr

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    #anchor property属性添加和扩展=========================================

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) #这里的fused_color是rgb基色，相当于f_dc，即degree为0对应的结果
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        # normal = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0
        albedo = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        roughness = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        metallic = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        #高斯的优化使用nn.Parameter进行管理，方便后续计算图的构建以及使用优化器optimizer进行参数更新，使用nn.Parameter
        #进行tensor注册的好处是其作为叶子节点会自动的将tensor的requires_grad属性设置为True，这样在反向传播的时候，这个
        # tensor的梯度就会被计算,而一般的叶子节点或创建的tensor其默认的requires_grad属性是False，反向传播时不会计算梯度，
        #进而也就不会被更新
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )  #这里通过transpose将features维度从(P,3,SH_coeffs)转化为(P,SH_coeffs,3)主要是便于后续的计算和统一，即任意tensor最后一个维度都代表rgb channel
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size  #这里相当于将在同一个voxel内的点合并成一个点
        
        return data

    def create_from_pcd_anchor(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio] #self.ratio 控制的是初始化点云的采样比例

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')  #0.001 for voxel_size
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)  #(N,3)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda() #(N,10,3)
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda() #(N,32)
        anchors_pbr_feat=torch.zeros((fused_point_cloud.shape[0], self.pbr_feature_dim)).float().cuda() #(N,32)
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6) #为什么这里的scales的维度是6呢?
                                    #回答:前3个表示offset的缩放，后3个表示scele的缩放,用于进一步调整每个neural gaussian
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        #pbr 属性添加 ==========================================
        normal = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0
        albedo = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        roughness = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        metallic = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        #pbr 属性添加结束 ==========================================
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._anchor_pbr_feat = nn.Parameter(anchors_pbr_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False)) #这里像rotation,opacity以及normal这些属性，全都是通过一个mlp从
        self._opacity = nn.Parameter(opacities.requires_grad_(False))  #feature中提取出来的，故梯度更新只用更新对应的mlp就可以了
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        #pbr 属性添加 ==========================================
        self._normal = nn.Parameter(normal.requires_grad_(False))
        self._albedo = nn.Parameter(albedo.requires_grad_(False))
        self._roughness = nn.Parameter(roughness.requires_grad_(False))
        self._metallic = nn.Parameter(metallic.requires_grad_(False))
        #pbr 属性添加结束 ==========================================
        
    def training_setup_anchor(self, training_args):
        self.percent_dense = training_args.percent_dense #0.01

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_pbr_feat], 'lr': training_args.pbr_feature_lr, "name": "anchor_pbr_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                #pbr 属性添加
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
                #pbr属性添加
                {'params': self.mlp_pbr.parameters(), 'lr': training_args.mlp_pbr_lr_init, "name": "mlp_pbr"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_pbr_feat], 'lr': training_args.pbr_feature_lr, "name": "anchor_pbr_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                #pbr 属性添加
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
                #pbr属性添加
                {'params': self.mlp_pbr.parameters(), 'lr': training_args.mlp_pbr_lr_init, "name": "mlp_pbr"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_pbr_feat], 'lr': training_args.pbr_feature_lr, "name": "anchor_pbr_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                #pbr 属性添加
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                #pbr属性添加
                {'params': self.mlp_pbr.parameters(), 'lr': training_args.mlp_pbr_lr_init, "name": "mlp_pbr"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        #pbr属性添加
        self.mlp_pbr_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_pbr_lr_init,
                                                        lr_final=training_args.mlp_pbr_lr_final,
                                                        lr_delay_mult=training_args.mlp_pbr_lr_delay_mult,
                                                        max_steps=training_args.mlp_pbr_lr_max_steps)
        
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def training_setup(self, training_args: GroupParams) -> None:
        self.percent_dense = training_args.percent_dense  #0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._normal], "lr": training_args.opacity_lr, "name": "normal"},
            {"params": [self._albedo], "lr": training_args.opacity_lr, "name": "albedo"},
            {"params": [self._roughness], "lr": training_args.opacity_lr, "name": "roughness"},
            {"params": [self._metallic], "lr": training_args.opacity_lr, "name": "metallic"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration: int) -> float:
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_learning_rate_anchor(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            #pbr 属性添加
            if param_group["name"] == "mlp_pbr":
                lr=self.mlp_pbr_scheduler_args(iteration)
                param_group['lr']=lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
                
    def construct_list_of_attributes(self) -> List[str]:
        l = ["x", "y", "z"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self._normal.shape[1]):
            l.append(f"normal_{i}")
        for i in range(self._albedo.shape[1]):
            l.append(f"albedo_{i}")
        l.append("roughness")
        l.append("metallic")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l

    def construct_list_of_attributes_anchor(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        #pbr 属性添加
        for i in range(self._anchor_pbr_feat.shape[1]):
            l.append('f_anchor_pbr_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        #pbr 属性添加
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('roughness')
        l.append('metallic')
        return l
    
    def save_ply(self, path: str) -> None:
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                f_dc,
                f_rest,
                opacities,
                normal,
                albedo,
                roughness,
                metallic,
                scale,
                rotation,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_ply_anchor(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        # normals = np.zeros_like(anchor) #这里anchor是可以存储法线的，后续的修改可以考虑让anchor存储法线做变种
        normals=self._normal.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        anchor_pbr_feat=self._anchor_pbr_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_anchor()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, anchor_pbr_feat,
                                                opacities, scale, rotation,albedo,roughness,metallic), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        #法线加载
        normals= np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1).astype(np.float32)
        

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        #加载roughness和metallic属性
        roughness= np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis].astype(np.float32)
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        #anchor_pbr_feats
        anchor_pbr_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_pbr_feat")]
        anchor_pbr_feat_names= sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_pbr_feats = np.zeros((anchor.shape[0], len(anchor_pbr_feat_names)))
        for idx, attr_name in enumerate(anchor_pbr_feat_names):
            anchor_pbr_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            
        #albedo
        albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo")]
        albedo_names = sorted(albedo_names, key = lambda x: int(x.split('_')[-1]))
        albedos = np.zeros((anchor.shape[0], len(albedo_names)))
        for idx, attr_name in enumerate(albedo_names):
            albedos[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_pbr_feat = nn.Parameter(torch.tensor(anchor_pbr_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        #pbr属性添加
        self._albedo=  nn.Parameter(torch.tensor(albedos, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness=  nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic=  nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))

    def load_mlp_sparse_gaussian(self,path):
        #mlp模型参数加载
        self.mlp_opacity.load_state_dict(torch.load(os.path.join(path, 'opacity_mlp.pt')))
        self.mlp_cov.load_state_dict(torch.load(os.path.join(path, 'cov_mlp.pt')))
        self.mlp_pbr.load_state_dict(torch.load(os.path.join(path, 'pbr_mlp.pt')))
        self.mlp_color.load_state_dict(torch.load(os.path.join(path, 'color_mlp.pt')))
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(torch.load(os.path.join(path, 'feature_bank_mlp.pt')))
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(torch.load(os.path.join(path, 'embedding_appearance.pt'))) 

    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            self.mlp_pbr = torch.jit.load(os.path.join(path, 'pbr_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            self.mlp_pbr.load_state_dict(checkpoint['pbr_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
        
    def reset_opacity(self) -> None:
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        normal = np.stack(
            (
                np.asarray(plydata.elements[0]["normal_0"]),
                np.asarray(plydata.elements[0]["normal_1"]),
                np.asarray(plydata.elements[0]["normal_2"]),
            ),
            axis=1,
        )
        albedo = np.stack(
            (
                np.asarray(plydata.elements[0]["albedo_0"]),
                np.asarray(plydata.elements[0]["albedo_1"]),
                np.asarray(plydata.elements[0]["albedo_2"]),
            ),
            axis=1,
        )
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._normal = nn.Parameter(
            torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._albedo = nn.Parameter(
            torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._roughness = nn.Parameter(
            torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._metallic = nn.Parameter(
            torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> Dict:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        #这里anchor_visible_mask 的shape和temp_opacity是相对应的， 
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask #这里使用offset_selection_mask是因为除了anchor的visiblity之外，
                                                         #附加在每个anchor上的guassian也会因为neural opaticy是否大于0而判断是否有效
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter  #这里指的是并非所有的neural gaussian都是有效的，部分附着在可见anchor上的neural gaussian
                                                  #并不一定在高斯render的cull frustum内
        # 既然这样的话那最终有效的mask_filter还是要依靠最原初的gaussian render函数进行判定，那是不是有方法可以简化这一部分的设计呢?
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm  #这里累积的是梯度偏移量，但比较奇怪的点在于为什么是针对屏幕空间点这样做呢?
        self.offset_denom[combined_mask] += 1   #这里存储的是对应gausiian更新gradient的总次数，便于后续计算平均梯度变化值

    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  #这里的update_depth 对应的是论文的多分辨率voxel的anchor point更新
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold 这里和3d gs是类似的，即梯度变化显著的区域进行anchor更新
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask) #这里的candidate_mask筛选的本质上是在一个更新
            #interval之内(本论文设置为100)大多数情况下都是在视锥之内且有效的高斯点，且这样的高斯点梯度变化要显著，同时为了防止
            #anchor point增长过快，要进一步进行了一个随机筛选
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor  #update_init_factor 16 update_hierachy_factor 4
            #这里的grid_coords构建的是已经存在的anchor点的稀疏网格坐标系,用于标记已经存在的anchor point，anchor point对应着一个voxel,
            #anchor point在该voxel的中心点，这里构建grid_coords是为了后续去重，避免重复添加anchor point
            grid_coords = torch.round(self.get_anchor / cur_size).int() #这里实际上是通过简单的数学来实现一个稀疏网格坐标系
            #这里的selected_xyz以及selected_grid_coords指的是实际的真实neural 高斯点数据，通过筛选的高斯点数据得到筛选高斯点所对应的
            #稀疏网格坐标系,很值得参考，避免了重复建整个空间的坐标系问题
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]  #这里的代码实现trick是值得参考的
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
            #这里remove_duplicates应该指的是当前挑选的网格点，取反代表有效的实际要添加在该分辨率层级上的anchor point
            #candidate_anchor是通过筛选后要新添加的anchor point
            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size #这里重新
                        #乘以cur_size是为了恢复anchor point的真实xyz坐标，之前最开始是除以了cur_size进行了一个voxel去重操作
            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0
                #pbr 属性添加
                new_normal = torch.zeros([candidate_anchor.shape[0], 3], device=candidate_anchor.device).float()
                new_normal[:, 2] = 1.0
                new_albedo = torch.ones([candidate_anchor.shape[0], 3], device=candidate_anchor.device).float()
                new_roughness = torch.ones([candidate_anchor.shape[0], 1], device=candidate_anchor.device).float()
                new_metallic =torch.ones([candidate_anchor.shape[0], 1], device=candidate_anchor.device).float()

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                #这里new_feat进行的下列复杂操作实际上是为了保证新添加的anchor point的特征和最近邻的anchor point其特征是相一致的，
                # 这样可以保证优化的稳定性，且局部特征是一致和相似的,哪个scatter_max函数是为了保证其最终维度和candidate_anchor维度是相一致的
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                #pbr feat 添加
                new_pbr_feat=self._anchor_pbr_feat.unsqueeze(dim=1).repeat([1,self.n_offsets,1]).view([-1,self.pbr_feat_dim])[candidate_mask]
                new_pbr_feat=scatter_max(new_pbr_feat,inverse_indices.unsqueeze(1).expand(-1,new_pbr_feat.size(1)),dim=0)[0][remove_duplicates]
                
                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_pbr_feat":new_pbr_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "normal":new_normal,
                    "albedo":new_albedo,
                    "roughness":new_roughness,
                    "metallic":new_metallic
                }
                
                #这里像3d 高斯属性的扩展和优化一般是通过每隔一定阶段删除原有的高斯点属性，新添加新的高斯点属性重新更新计算图再优化新的高斯点属性逐步迭代优化实现的吗?
                #回答: 现在来看应该是这样子的
                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer_anchor(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_pbr_feat=optimizable_tensors["anchor_pbr_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._normal=optimizable_tensors["normal"]
                self._albedo=optimizable_tensors["albedo"]
                self._roughness=optimizable_tensors["roughness"]
                self._metallic=optimizable_tensors["metallic"]
                     
    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1] 计算平均梯度
        grads[grads.isnan()] = 0.0  #处理可能的数值不稳定性
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors,这里根据opacity裁剪掉平庸的anchor point
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors
    
    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._anchor_pbr_feat=optimizable_tensors["anchor_pbr_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal=optimizable_tensors["normal"]
        self._albedo=optimizable_tensors["albedo"]
        self._roughness=optimizable_tensors["roughness"]
        self._metallic=optimizable_tensors["metallic"]
        
    def cat_tensors_to_optimizer_anchor(self, tensors_dict):
        #看代码解析本质上真的就是重新构建计算图，将新的tensor添加到optimizer中
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
            
    def _prune_optimizer(self, mask: torch.Tensor) -> Dict:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask: torch.Tensor) -> None:
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict: Dict) -> Dict:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_normal: torch.Tensor,
        new_albedo: torch.Tensor,
        new_roughness: torch.Tensor,
        new_metallic: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
    ) -> None:
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "normal": new_normal,
            "albedo": new_albedo,
            "roughness": new_roughness,
            "metallic": new_metallic,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
    ) -> None:
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_normal,
            new_albedo,
            new_roughness,
            new_metallic,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
    ) -> None:
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_normal,
            new_albedo,
            new_roughness,
            new_metallic,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: int,
    ) -> None:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
    ) -> None:
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def save_mlp_checkpoints_myversion(self, path, mode='split'):  # split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            torch.save(self.mlp_opacity.state_dict(), os.path.join(path, 'opacity_mlp.pt'))
            torch.save(self.mlp_cov.state_dict(), os.path.join(path, 'cov_mlp.pt'))
            torch.save(self.mlp_color.state_dict(), os.path.join(path, 'color_mlp.pt'))
            torch.save(self.mlp_pbr.state_dict(), os.path.join(path, 'pbr_mlp.pt'))
            if self.use_feat_bank:
                torch.save(self.mlp_feature_bank.state_dict(), os.path.join(path, 'feature_bank_mlp.pt'))
            if self.appearance_dim:
                torch.save(self.embedding_appearance.state_dict(), os.path.join(path, 'embedding_appearance.pt'))
        elif mode == 'unite':
            state_dict = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'pbr_mlp': self.mlp_pbr.state_dict(),
            }
            if self.use_feat_bank:
                state_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                state_dict['appearance'] = self.embedding_appearance.state_dict()
            torch.save(state_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError
    
    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()
            
            self.mlp_pbr.eval()
            pbr_mlp = torch.jit.trace(self.mlp_pbr, (torch.rand(1, self.feat_dim+3+self.pbr_dist_dim).cuda()))
            pbr_mlp.save(os.path.join(path, 'pbr_mlp.pt'))
            self.mlp_pbr.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'pbr_mlp': self.mlp_pbr.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'pbr_mlp': self.mlp_pbr.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'pbr_mlp': self.mlp_pbr.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError