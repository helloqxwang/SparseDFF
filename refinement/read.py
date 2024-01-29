import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from typing import Tuple, List
from camera import pipeline
# from camera.camera_tools import load_cddi, get_extrinsics_from_json, vis_color_pc
import argparse
import open3d as o3d
import json
import yaml
from scipy.spatial.transform import Rotation

def vis_color_pc(points:np.ndarray, colors:np.ndarray, size:int=0.1, save=False, rot=None):
    """view the point cloud with colors

    Args:
        points (np.ndarray): (n, 3)
        colors (np.ndarray): (n, 3)
    """
    if size == 0:
        size = 0.1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    if rot is not None:
        axis.rotate(rot, center=[0, 0, 0])
    if save:
        o3d.io.write_point_cloud('./vis_pcd.ply', pcd)
    else:
        o3d.visualization.draw_geometries([pcd, axis])

def read_tranformation(data_path:str='../camera/transform.yaml'):
    print('Reading transformation from ', data_path)
    with open(data_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    position = np.array([data['pose']['position']['x'], data['pose']['position']['y'], data['pose']['position']['z']])
    quaternion = np.array([data['pose']['orientation']['x'], data['pose']['orientation']['y'], data['pose']['orientation']['z'], data['pose']['orientation']['w']])
    table2cam = np.eye(4)
    table2cam[:3, 3] = position
    table2cam[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    cam2base_str:str = data['cam2base']
    cam2base_str = cam2base_str.replace('[', '').replace(']', '').replace('\n', '')
    cam2base = np.array([float(i) for i in cam2base_str.split()]).reshape(4, 4)
    return table2cam, cam2base

def get_extrinsics_from_json(path:str):
    """return the extrinsics of the four cameras 
        world2cam0, world2cam1, world2cam2, world2cam3

    Args:
        path (str): where calibration json file is

    Returns:
        np.ndarray: extrinsics of the four cameras [world_cam] shape: (4, 4, 4)
    """

    with open(path, 'r') as f:
        data = json.load(f)
    cam1 = data['camera_poses']["cam1_to_cam0"]
    cam2 = data['camera_poses']["cam2_to_cam0"]
    cam3 = data['camera_poses']["cam3_to_cam0"]

    table2cam, cam2base = read_tranformation()
    ### we seen the table frame as the world frame
    world_c0 = table2cam
    world_c0[:3, 3] = world_c0[:3, 3] * 1000
    world2base = cam2base @ world_c0

    c0_c1 = np.eye(4)
    c0_c1[:3, :3] = np.array(cam1['R']).reshape(3, 3)
    c0_c1[:3, 3] = np.array(cam1['T']) * 1000
    world_c1 = c0_c1 @ world_c0
    c0_c2 = np.eye(4)
    c0_c2[:3, :3] = np.array(cam2['R']).reshape(3, 3)
    c0_c2[:3, 3] = np.array(cam2['T']) * 1000
    world_c2 = c0_c2 @ world_c0
    c0_c3 = np.eye(4)
    c0_c3[:3, :3] = np.array(cam3['R']).reshape(3, 3)
    c0_c3[:3, 3] = np.array(cam3['T']) * 1000
    world_c3 = c0_c3 @ world_c0
    extrinsics = np.stack([world_c0, world_c1, world_c2, world_c3], axis=0)

    return extrinsics, world2base

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
        
        points, features, colors, batch_sign, raw_points= pipeline(path, extrinsics_path, save=False, scale=scale, name = name, prune_method='sam', samckp_path='../thirdparty_module/sam_vit_h_4b8939.pth')
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
        saving_path = os.path.abspath(f'./data{key}')
        print('Saving_path:', os.path.abspath(saving_path))
        os.makedirs(saving_path, exist_ok=True)
        for i in range(4):
            point_i = points[batch_sign == i + 1]
            feature_i = features[batch_sign == i + 1]
            batch_sign_i = batch_sign[batch_sign == i + 1]
            np.save(os.path.join(saving_path, f'points_{ii}_{i}.npy'), point_i.cpu().numpy())
            np.save(os.path.join(saving_path, f'features_{ii}_{i}.npy'), feature_i.cpu().numpy())
            np.save(os.path.join(saving_path, f'scene_sign_{ii}_{i}.npy'), batch_sign_i.cpu().numpy())
            for j in range(4):
                if i >= j :
                    continue
                print(i, j)
                point_j = points[batch_sign == j + 1]
                feature_j = features[batch_sign == j + 1]
                tt:np.ndarray = match_ij(point_i, point_j, dis_thre=0.01)
                print(tt.shape)
                np.save(os.path.join(saving_path, f'match_{ii}_{i}_{j}.npy'), tt)

                
                

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
    argparser.add_argument('--dir_path', type=str, default='../example_data/img')
    argparser.add_argument('--extrinsics_path', type=str, default='../camera/workspace/calibration.json')
    argparser.add_argument('--img_data_path', type=str, default='20231010_monkey_original')
    args = argparser.parse_args()
    data_ls = [args.img_data_path]
    load_data(args.dir_path, args.extrinsics_path, data_ls=data_ls, auto_detect=False, scale=args.scale, key=args.key, device='cuda', mode = args.mode)