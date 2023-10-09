import sys
import os
import numpy as np
import torch
from typing import Tuple, List
from camera import undistort, pipeline
from camera.camera_tools import load_cddi, get_extrinsics_from_json, vis_color_pc
import argparse

def match_ij(points0:torch.Tensor, points1:torch.Tensor, dis_thre=0.01):
    """_summary_

    Args:
        points0 (torch.Tensor): (num0, 3)
        points1 (torch.Tensor): (num1, 3)
        dis_thre (float, optional): _description_. Defaults to 0.025.
    """
    dis = torch.cdist(points0.unsqueeze(0).to(torch.float32), points1.unsqueeze(0).to(torch.float32), p=2).squeeze(0)
    map = dis < dis_thre
    index = torch.nonzero(map)
    return index.cpu().numpy()

def load_data(data_path, extrinsics_path, data_ls=None, auto_detect=False, scale=3, key=0, device='cuda', mode='linear_probe')->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """load Color, Depth, Distortion, Intrinsics from data_path
    Every thing stores in numpy.ndarray


    Args:
        data_path (str): the path of a folder for a CLASS of captures
        data_ls (str, optional): the name of a folder for ONE capture. Defaults to None.
        load_from_path (bool, optional): Auto-Detect and Load all the folders under data_path. Defaults to False.
    Return
        colors, depths, intrinsics (np.ndarray) (batch_size, camera_num, h, w, 3) for colors

    """

    assert data_ls is not None or auto_detect is True, 'data_ls and load_from_path cannot be Negtive at the same time'
    assert data_ls is None or auto_detect is False, 'data_ls and load_from_path cannot be Positive at the same time'
    path_ls = []
    if auto_detect:
        for dir in os.listdir(data_path):
            path = os.path.join(data_path, dir)
            path_ls.append(path)
    else:
        for stem in data_ls:
            path = os.path.join(data_path, stem)
            # only reserve valid data
            if os.path.exists(os.path.join(path, '000262413912')):
                path_ls.append(path)
    colors_ls, features_ls, points_ls, sign_ls = [], [], [], []
    extrinsics, _ = get_extrinsics_from_json(extrinsics_path)
    print('Finish loading data')
    for ii, path in enumerate(path_ls):
        name = os.path.split(path)[-1]
        name = name + 'read'

        
        
        
        
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=False, scale=scale, name = name, prune_method='sam')
        vis_color_pc(points.cpu().numpy(), colors.reshape(-1,3), 0.1, save=True)
        # exit()
        # np.save(f'./features_pruned_{ii}.npy', features.cpu().numpy())
        # np.save(f'./points_pruned_{ii}.npy', points.cpu().numpy())
        if mode == 'mha':
            np.save(f'./mha_data{key}/points_pruned_{ii}.npy', points.cpu().numpy())
            np.save(f'./mha_data{key}/features_pruned_{ii}.npy', features.cpu().numpy())
            continue
        colors_ls.append(colors)
        features_ls.append(features)
        points_ls.append(points)
        sign_ls.append(batch_sign)
        print('Saving_path:', os.path.abspath(f'./data{key}'))
        for i in range(4):
            point_i = points[batch_sign == i + 1]
            feature_i = features[batch_sign == i + 1]
            batch_sign_i = batch_sign[batch_sign == i + 1]
            np.save(f'./data{key}/points_{ii}_{i}.npy', point_i.cpu().numpy())
            np.save(f'./data{key}/features_{ii}_{i}.npy', feature_i.cpu().numpy())
            np.save(f'./data{key}/scene_sign_{ii}_{i}.npy', batch_sign_i.cpu().numpy())
            for j in range(4):
                if i >= j :
                    continue
                print(i, j)
                point_j = points[batch_sign == j + 1]
                feature_j = features[batch_sign == j + 1]
                tt:np.ndarray = match_ij(point_i, point_j, dis_thre=0.01)
                print(tt.shape)
                np.save(f'./data{key}/match_{ii}_{i}_{j}.npy', tt)

                
                

        # vis_color_pc(points.cpu().numpy(), colors_ref_pruned.reshape(-1,3), 0.1)
    
    for i in range(len(colors_ls)):
        print('colors shape:', colors_ls[i].shape)
        print('features shape:', features_ls[i].shape)
        print('points shape:', points_ls[i].shape)
        print('sign shape:', sign_ls[i].shape)


    return colors_ls, features_ls, points_ls, sign_ls

def from_to(data_path, f, t):
    all_data = os.listdir(data_path)
    all_data.sort()
    f_idx = all_data.index(f)
    t_idx = all_data.index(t)
    data_ls = all_data[f_idx:t_idx+1]
    return data_ls



    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='linear_probe')
    argparser.add_argument('--key', type=int, default=0)
    argparser.add_argument('--scale', type=bool, default=14)
    args = argparser.parse_args()
    data_path = '../camera/data'
    extrinsics_path = '../camera/workspace/calibration.json'
    # data_ls = from_to(data_path, '20230911_142418', '20230911_142957')
    data_ls = ['20230929_150633']
    print(data_ls)
    load_data(data_path, extrinsics_path, data_ls=data_ls, auto_detect=False, scale=args.scale, key=args.key, device='cuda', mode = args.mode)
    # data_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54'
    # extrinsics_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54/extrinsics.json'
    # load_data(data_path, extrinsics_path, data_ls=['2021-01-27-16-21-54-0'], auto_detect=False, scale=3, key=0, device='cuda')
    # data_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54'
    # extrinsics_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54/extrinsics.json'
    # load_data(data_path, extrinsics_path, data_ls=['2021-01-27-16-21-54-0'], auto_detect=False, scale=3, key=0, device='cuda')
    # data_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54'
    # extrinsics_path = '/home/user/wangqx/stanford/kinect/workspace/2021-01-27-16-21-54/extrinsics.json'