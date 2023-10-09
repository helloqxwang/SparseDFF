import sys
import os
import torch
import numpy as np
import open3d as o3d
from prune.tools import *
from prune.prune_3D import find_match_3D, find_match_3D_quotient, vote_3D
from camera import pipeline
from typing import List
from camera.camera_tools import vis_color_pc
from scipy.spatial.transform import Rotation
from refinement.model import LinearProbe, LinearProbe_Thick, LinearProbe_Juicy, LinearProbe_PerScene, LinearProbe_PerSceneThick, LinearProbe_Glayer

def get_points_features_from_mesh(key=0, mesh_path=None, device='cuda', scale=4, img_num=40, img_size=1024, method='3D_match'):
    """get the points and the features of the mesh

    Args:
        key (int, optional): which keys of the anime used. Defaults to 0.
        mesh_path (_type_, optional): bunny default. Defaults to None.
        device (str, optional): device. Defaults to 'cuda'.
        scale (int, optional): scale of the feature generation. Defaults to 4.
        img_num (int, optional): num of image used. Defaults to 40.
        img_size (int, optional): size of the original images. Defaults to 1024.

    Returns:
        match_points: torch.tensor (num, 3)
        match_features: torch.tensor dino_features (num, 768)
    """
    if mesh_path is None:
        mesh_path = '/home/user/wangqx/stanford/bunnyQ_Attack1_{}.obj'.format(key)
    imgs, depths_ori, c2w, K, camera_params = mesh2rgbd([mesh_path], device, num=img_num, img_size=img_size)
    print('Finish loading images')
    features = get_features(imgs, scale, key, img_num, img_size=img_size)
    n, h, w, c= features.shape
    depths = torch.from_numpy(np.array([cv2.resize(depth, (h , w), interpolation=cv2.INTER_NEAREST) for depth in depths_ori.cpu().numpy()])).to(device)
    if method == '2D_match':
        points = depth2pt(depths, camera_params, torch.from_numpy(c2w).to(device).to(torch.float32), device=device)
        match_points, match_features = get_matched_pt_ft(features, depths, points, device=device)
        return match_points, match_features
    elif method == '3D_match':
        points, batch_sign, filter_depth = depth2pt(depths, camera_params, torch.from_numpy(c2w).to(device).to(torch.float32), xyz_images=False, device=device)
        threshold = 0.1
        points_select, index_select= find_match_3D(points, batch_sign, img_num, dis_threshold=threshold)
        features_select = features.reshape(-1, c)[filter_depth][index_select]
        pt_vis(points_select.cpu().numpy(), size=threshold)
        return points_select, features_select
    else:
        raise NotImplementedError


def get_points_features_from_real(path=None, extrinsics_path:str=None, save=True,
                                  key=0, name='bear', device='cuda', scale=6,
                                   method='binearest-match', dis_threshold=0.1, 
                                   quotient_threshold=0.8, verbose=False, model_path=None,
                                   p0 = 'pyhsics', p1= 'pyhsics'):
    if key == 0:
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=save, scale=scale, name = name, prune_method=p0, key=0)
    elif key == 1:
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=save, scale=scale, name = name, prune_method=p1, key=1)
    points_ref, _ = prune_box(raw_points, x=[-0.42, 0.48], y=[-0.56, 0.56], z=[-0.135, 0.8])

    img_num = points.shape[0]
    if method == 'quotient_match':
        points_select, index_select= find_match_3D_quotient(points, batch_sign, img_num, dis_threshold=dis_threshold, quotien_threshold=quotient_threshold)
    elif method == 'binearest_match':
        points_select, index_select= find_match_3D(points, batch_sign, img_num, dis_threshold=dis_threshold)
    elif method == 'vote_3D':
        points_select, index_select = vote_3D(points, batch_sign, img_num, dis_threshold=dis_threshold, selected_num=int(points.shape[0] * 0.8))
    elif method == 'haha':
        # points_select, index_select = points_pruned[batch_sign_pruned == 2], batch_sign_pruned == 2
        points_select, index_select = points, torch.arange(points.shape[0]).to(device)
    else:
        raise NotImplementedError

    features_select = features.reshape(-1, features.shape[-1])[index_select]
    batch_sign_select = batch_sign[index_select.to(batch_sign.device)]
    colors_select = colors[index_select.cpu().numpy()]
    if model_path is not None:
        model = LinearProbe_Glayer(768, 768 * 4, 768, g_size=64, ref=True).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        features_select = model(features_select).detach()

    if verbose:
        print('features_select: ', features_select.shape)
        print(f'The whole number of points of object{key}: {points_select.shape[0]}')

    return points_select, features_select, colors_select, points.cpu().numpy(), colors, points_ref
    
