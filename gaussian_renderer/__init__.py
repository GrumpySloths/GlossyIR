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
import math
from typing import Dict, Optional

from einops import repeat
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from arguments import GroupParams
from scene.cameras import Camera
from scene.aussian_model import GaussianModel
from utils.sh_utils import eval_sh


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    feat_pbr=pc._anchor_pbr_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    #添加pbr local_view
    cat_local_view_pbr=torch.cat([feat_pbr,ob_view,ob_dist],dim=1) # [N, c+3+1]
    cat_local_view_pbr_wodist=torch.cat([feat_pbr,ob_view],dim=1)  # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist) #[N,30]
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]  [N*10,3]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    #pbr属性添加 get offset's pbr attribute
    if pc.add_pbr_dist:
        pbr=pc.get_pbr_mlp(cat_local_view_pbr)
    else:
        pbr=pc.get_pbr_mlp(cat_local_view_pbr_wodist)
    pbr=pbr.reshape([anchor.shape[0]*pc.n_offsets,8])
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot,pbr, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, pbr,offsets = masked.split([6, 3, 3, 7, 8, 3], dim=-1)
    
    # post-process cov  这里scaling的处理是依据anchor的scaling来扩展附属在anchor上neural_gaussian的scaling
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    #pbr属性获取
    normal=pc.normal_activation(pbr[:,:3])
    albedo=pc.material_activation(pbr[:,3:6])
    roughness=pc.material_activation(pbr[:,6])
    metallic=pc.material_activation(pbr[:,7])  #比较疑惑的一个点是metallic解耦出来的值是有效的
    #修改shape以适应自定义cuda的前向和后向传播过程 
    roughness=roughness.reshape([-1,1])
    metallic=metallic.reshape([-1,1])
     
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  #这里可能是一个trick吧，同时利用了anchor的scaling和offsets来计算neural_gaussian的中心
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, normal,albedo,roughness,metallic,neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot,normal,albedo,roughness,metallic,
    

def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    visible_mask=None,
    retain_grad=False,
    override_color: Optional[torch.Tensor] = None,
    inference: bool = False,
    pad_normal: bool = False,
    derive_normal: bool = False,
) -> Dict:
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training
    #这里有一个疑问就是在pbr训练流程管线上哪些部分该是training模式，哪些部分该是eval模式呢?
    # print("is_training:",is_training)
        
    if is_training:
        xyz, color, opacity, scaling, rot,normal,albedo,roughness,metallic, neural_opacity, mask = (
                            generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        )
    else:
        xyz, color, opacity, scaling, rot,normal,albedo,roughness,metallic =(
                            generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        ) 

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    #这里使用retain_grad()是因为对于高斯来说非叶子节点的梯度是不会被保存的，所以这里需要手动保存，其梯度才会被记录下来，但不管
    #是否使用retain_grad()整个计算图都会被保存下来，即反向传播过程中该被计算的梯度都会被计算的
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        inference=inference,
        argmax_depth=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # means2D = screenspace_points
    # opacity = pc.get_opacity
    # normal = pc.get_normal
    # albedo = pc.get_albedo
    # roughness = pc.get_roughness
    # metallic = pc.get_metallic
    assert albedo.shape[0] == roughness.shape[0] and albedo.shape[0] == metallic.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    #         dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) #这里是将rgb颜色范围限制在0到1之间
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        rendered_image,
        radii,
        opacity_map,
        depth_map,
        normal_map_from_depth,
        normal_map,  #shape :[3,800,800]
        albedo_map,
        roughness_map,
        metallic_map,
    ) = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        opacities=opacity,
        normal=normal,
        shs=None,
        colors_precomp=color,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
        derive_normal=derive_normal,  #derive_normal 为true
    )

    # print("xyz.shape:",xyz.shape)
    # print("screenspace_points.shape:",screenspace_points.shape)
    # print("opacity.shape:",opacity.shape)
    # print("normal.shape:",normal.shape)
    # print("color.shape:",color.shape)
    # print("albedo.shape:",albedo.shape)
    # print("roughness.shape:",roughness.shape)
    # print("metallic.shape:",metallic.shape)
    # print("scaling.shape:",scaling.shape)
    # print("rot.shape:",rot.shape)
    
    normal_mask = (normal_map != 0).all(0, keepdim=True)  #shape [1,800,800],这里应该是去筛选有效的法线
    normal_from_depth_mask = (normal_map_from_depth != 0).all(0)  #shape [800,800]
    if pad_normal:
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map < 0.004,
            torch.zeros_like(opacity_map),
            opacity_map,
        )
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map > 1.0 - 0.004,
            torch.ones_like(opacity_map),
            opacity_map,
        )
        normal_bg = torch.tensor([0.0, 0.0, 1.0], device=normal_map.device)
        normal_map = normal_map * opacity_map + (1.0 - opacity_map) * normal_bg[:, None, None]
        mask_from_depth = (normal_map_from_depth == 0.0).all(0, keepdim=True).float()
        normal_map_from_depth = normal_map_from_depth * (1.0 - mask_from_depth) + mask_from_depth * normal_bg[:, None, None]
        
    #下面的代码是将normal_map 中的每个法线normal进行归一化处理，使其模长为1
    normal_map_from_depth = torch.where(
        torch.norm(normal_map_from_depth, dim=0, keepdim=True) > 0,
        F.normalize(normal_map_from_depth, dim=0, p=2),
        normal_map_from_depth,
    )
    normal_map = torch.where(
        torch.norm(normal_map, dim=0, keepdim=True) > 0,
        F.normalize(normal_map, dim=0, p=2),
        normal_map,
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "opacity_map": opacity_map,
            "depth_map": depth_map,
            "normal_map_from_depth": normal_map_from_depth,
            "normal_from_depth_mask": normal_from_depth_mask,
            "normal_map": normal_map,
            "normal_mask": normal_mask,
            "albedo_map": albedo_map,
            "roughness_map": roughness_map,
            "metallic_map": metallic_map,
            "neural_opacity": neural_opacity,
            "selection_mask": mask,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "opacity_map": opacity_map,
            "depth_map": depth_map,
            "normal_map_from_depth": normal_map_from_depth,
            "normal_from_depth_mask": normal_from_depth_mask,
            "normal_map": normal_map,
            "normal_mask": normal_mask,
            "albedo_map": albedo_map,
            "roughness_map": roughness_map,
            "metallic_map": metallic_map,
        }
