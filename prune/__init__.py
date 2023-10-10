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

def get_points_features_from_real(path=None, extrinsics_path:str=None, save=True,
                                  key=0, name='bear', device='cuda', scale=6,
                                   method='binearest-match', dis_threshold=0.1, 
                                   quotient_threshold=0.8, verbose=False, model_path=None,
                                   p0 = 'pyhsics', p1= 'pyhsics'):
    if key == 0:
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=save, scale=scale, name = name, prune_method=p0, key=0, verbose=verbose)
    elif key == 1:
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=save, scale=scale, name = name, prune_method=p1, key=1, verbose=verbose)
    points_ref, _ = prune_box(raw_points, x=[-0.42, 0.48], y=[-0.56, 0.56], z=[-0.135, 0.8])

    img_num = points.shape[0]
    if method == 'quotient_match':
        points_select, index_select= find_match_3D_quotient(points, batch_sign, img_num, dis_threshold=dis_threshold, quotien_threshold=quotient_threshold)
    elif method == 'binearest_match':
        points_select, index_select= find_match_3D(points, batch_sign, img_num, dis_threshold=dis_threshold)
    elif method == 'vote_3D':
        points_select, index_select = vote_3D(points, batch_sign, img_num, dis_threshold=dis_threshold, selected_num=int(points.shape[0] * 0.8))
    elif method == 'raw':
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
    
    np.save(f'./data/points_{key}.npy', points_select.cpu().numpy())
    np.save(f'./data/colors_{key}.npy', colors_select)
    np.save(f'./data/features_{key}.npy', features_select.cpu().numpy())

    if verbose:
        print('features_select: ', features_select.shape)
        print(f'The whole number of points of object{key}: {points_select.shape[0]}')

    return points_select, features_select, colors_select, points.cpu().numpy(), colors, points_ref
    
