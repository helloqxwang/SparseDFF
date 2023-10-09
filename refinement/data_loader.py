import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
from prune.tools import depth2pt_K_numpy, downsample, get_features, prune_sphere
from camera import undistort
from camera.camera_tools import load_cddi, get_extrinsics_from_json, vis_color_pc
from typing import Tuple
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import re

class MHA_Dataset(Dataset):

    @staticmethod
    def load_data(data_path, data_ls=None, auto_detect=False)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                path_ls.append(path)
        colors_ls, depths_ls, intrinsics_ls = [], [], []
        for path in path_ls:
            colors_distort, depths_distort, distortions, intrinsics = load_cddi(path)
            colors, depths = undistort(colors_distort, depths_distort, intrinsics, distortions)
            colors_ls.append(colors)
            depths_ls.append(depths)
            intrinsics_ls.append(intrinsics)

        colors= np.stack(colors_ls, axis=0)
        depths = np.stack(depths_ls, axis=0)
        intrinsics = np.stack(intrinsics_ls, axis=0)
        return colors, depths, intrinsics

    @staticmethod
    def line_dist(points:torch.tensor, line:torch.tensor)->torch.tensor:
        """compute the distance of the points to the line

        Args:
            point (torch.tensor): (num, 3)
            line (torch.tensor): (4, ) [a, b, c, d]

        Returns:
            dist (toch.tensor); (num, )
        """
        dist = torch.matmul(points, line[:-1].reshape(3, 1)) + line[-1]
        return dist.squeeze()      

    def __init__(self, root_dir, extrinsic_dir, scale=4, 
                 pose_ls=None, pose_auto_detect=True, 
                 key=0, name='test', raw:bool=True):
        """Load data from the root_dir.
        The return data is a tuple of (points, features) 
        (batch_size, cam_num, h, w, 3)
        (batch_size, cam_num, h, w, features_dim)

        Attention that, because of the optimize Goal -- ouput-feature-consistency in the same batch
        the batch_size is distributed in the first dimension,
        So we donnot need the auto-batch at all

        Args:
            root_dir (_type_): _description_
            extrinsic_dir (_type_): _description_
            scale (int, optional): _description_. Defaults to 4.
            pose_ls (_type_, optional): _description_. Defaults to None.
            pose_auto_detect (bool, optional): _description_. Defaults to True.
            key (int, optional): _description_. Defaults to 0.
            name (str, optional): _description_. Defaults to 'test'.
            raw (bool, optional): _description_. Defaults to True.
        """
        self.root_dir = root_dir
        self.extrinsic = get_extrinsics_from_json(extrinsic_dir)
        self.pose_path_ls = []
        if pose_auto_detect:
            for dir in os.listdir(root_dir):
                pose_path = os.path.join(root_dir, dir)
                self.pose_path_ls.append(pose_path)
        else:
            for stem in pose_ls:
                pose_path = os.path.join(root_dir, stem)
                self.pose_path_ls.append(pose_path)
        self.key = key
        self.name = name
        self.scale = scale
        self.raw = raw

    def __len__(self):
        return len(self.pose_path_ls)

    def __getitem__(self, idx):
        """get the points and the features 
        Attention that the batch_size already be distributed in the first dimension
        if raw is True, return points, features with (batch_size, cam_num, 3, h, w) and (batch_size, cam_num, 3, h, w)

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_path = self.pose_path_ls[idx]
        colors_full, depths_full, intrinsics_full = MHA_Dataset.load_data(data_path, auto_detect=True)
        
        if self.raw:
            points = depth2pt_K_numpy(depths_full.reshape(-1, depths[-2], depths[-1]), 
                                  intrinsics_full.reshape(-1, depths[-2], depths[-1]), 
                                  np.linalg.inv(self.extrinsic.repeat(depths.shape[0], axis=0)), 
                                  xyz_images=True).reshape(colors.shape)
            batch_size, cam_num, h, w, _ = points.shape
            h = h // self.scale
            w = w // self.scale
            # convert depths and points to torch.Tensor but remain colors being np.ndarray
            colors, depths, points = colors_full, torch.from_numpy(depths), torch.from_numpy(points)

            points = points.permute(0, 1, 4, 2, 3).reshape(batch_size, cam_num, -1, h, w)

            resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)

            # colors = resize(colors)
            points = resize(points) # (batch_size, cam_num, 3, h, w)
            depths = resize(depths) # (batch_size, cam_num, h, w)
            features = get_features(colors.reshpae(-1, colors.shape[-3], colors.shape[-2], colors.shape[-1]), self.scale, self.key, batch_size*cam_num, img_size=h, name=self.name)
            return points, features
        else:
            ### actually I can just return numpy here.
            # This just a dataset, so there is no need to convert everything to torch.Tensor
            depths_full[depths_full < 0] = 0
            batch_size, cam_num, h, w, _ = colors_full.shape
            h = h // self.scale
            w = w // self.scale

            points_ls, features_ls = [], []
            max_length = 0
            for i in range(colors.shape[0]):
                points_undistort = depth2pt_K_numpy(depths_full[i], intrinsics_full[i], np.linalg.inv(self.extrinsic), xyz_images=True)
                colors = colors_full[i]
                depths = depths_full[i]
                
                points_dd = np.array([cv2.resize(points_undistort[idx], (h, w), interpolation=cv2.INTER_NEAREST) for idx in range(points_undistort.shape[0])])
                depths_dd = np.array([cv2.resize(depths[idx], (h, w), interpolation=cv2.INTER_NEAREST) for idx in range(depths.shape[0])])
                zero_filter = (depths_dd != 0).reshape(-1)
                # batch_sign = batch_sign[zero_filter]
                points_dd = points_dd.reshape(-1, 3)[zero_filter]
                # colors_dd = colors_dd.reshape(-1, 3)[zero_filter]
                # we donnot need the batch_sign actually

                colors = colors.astype('float32')
                points_dd = points_dd.astype('float32')
                colors = torch.from_numpy(colors)
                points = torch.from_numpy(points_dd)
                zero_filter = torch.from_numpy(zero_filter)
                points = points / 1000.
                img_num = colors.shape[0]
                img_size = colors.shape[2]
                print('Finish loading images')
                features = get_features(colors, self.scale, i, img_num, img_size=img_size, name=self.name)
                features = features.reshape(-1, features.shape[-1])[zero_filter]
                print('Finish loading features')

                center = torch.tensor([-0.0849, -0.0590,  - 0.0056])
                radius = 0.6
                points_pruned, features_pruned = self.norm_points_features(points, features, center, radius)

                points_ls.append(points_pruned)
                features_ls.append(features_pruned)
                max_length = max(max_length, points_pruned.shape[0])
                print('the length of points_pruned:', points_pruned.shape[0])
            # points_ls [(num_0, 3), (num_1, 3), ...]
            for i in range(len(points_ls)):
                points_ls[i] = torch.cat([points_ls[i], torch.zeros(max_length - points_ls[i].shape[0], 3)], dim=0)
                features_ls[i] = torch.cat([features_ls[i], torch.zeros(max_length - features_ls[i].shape[0], features_ls[i].shape[-1])], dim=0)
            points = torch.stack(points_ls, dim=0)
            features = torch.stack(features_ls, dim=0)
            # (batch_size, points_max_num, 3), (batch_size, points_max_num, features_dim)
            return points, features
    
    def norm_points_features(self, points:torch.Tensor, features:torch.Tensor, center:torch.Tensor, radius:float=0.6)->Tuple[torch.Tensor, torch.Tensor]:
        # normalize the points
        points -= torch.mean(points, axis=0, keepdims=True)
        
        points_pruned, index_pruned = prune_sphere(points, center, radius)
        # vis_color_pc(points_pruned.reshape(-1,3).cpu().numpy(), colors_ref.reshape(-1,3), dis_threshold)
        
        ### prune the plane off
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_pruned.cpu().numpy())
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=2000)
        plane = np.array(plane_model) / np.linalg.norm(plane_model[:3])
        plane = torch.from_numpy(plane).to(torch.float32)
        dist_plane = MHA_Dataset.line_dist(points_pruned, plane)
        distance = 0.015
        if torch.count_nonzero(dist_plane < - distance) > torch.count_nonzero(dist_plane >  distance):
            ### make sure the norm_vec of the plane point to the bear side
            plane = - plane
            index_prune_plane = dist_plane < - distance
        else:
            index_prune_plane = dist_plane >  distance

        points_pruned = points_pruned[index_prune_plane]

        ### rotate and normalize the plane
        norm_vec = plane[:-1].cpu().numpy()
        angle = np.arccos(norm_vec.dot([0, 0, 1]) / np.linalg.norm(norm_vec))
        axis = np.cross(norm_vec, [0, 0, 1])
        r = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
        rotate_mat = torch.from_numpy(r.as_matrix()).to(torch.float32)
        points_pruned = torch.matmul(rotate_mat, points_pruned.T).T
        points_pruned -= points_pruned.mean(axis=0, keepdims=True)
        features_pruned = features[index_pruned][index_prune_plane]
        return points_pruned, features_pruned
            
def normalize(tensor:torch.Tensor)->torch.Tensor:
    return (tensor - tensor.mean(axis=0, keepdims=True)) / tensor.std(axis=0, keepdims=True)    

class MatchPairDataset(torch.utils.data.Dataset):
  def __init__(self, key, norm=False):
    self.key = key
    self.path = f'./data{key}'
    path_ls = os.listdir(self.path)
    self.files = [file for file in path_ls if 'match' in file]
    self.norm = norm



  def reset_seed(self, seed=0):
    print(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    """_summary_

    Args:
        idx (_type_): _description_

    Returns:
        (points0, points1, features0, features1, match_index)
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        (n, 3), (m, 3), (n, 768), (m, 768), (p, 2)
    """
    file = self.files[idx]
    parttern = r"(\d+)_(\d+)_(\d+).npy$"
    mm = re.search(parttern, file)
    b_idx, i, j = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
    match_index = np.load(os.path.join(self.path, file))
    points0 = np.load(os.path.join(self.path, f'points_{b_idx}_{i}.npy'))
    # print(os.path.join(self.path, f'points_{b_idx}_{j}.npy'))
    points1 = np.load(os.path.join(self.path, f'points_{b_idx}_{j}.npy'))
    features0 = np.load(os.path.join(self.path, f'features_{b_idx}_{i}.npy'))
    features1 = np.load(os.path.join(self.path, f'features_{b_idx}_{j}.npy'))
    scene_sign0 = np.load(os.path.join(self.path, f'scene_sign_{b_idx}_{i}.npy'))
    scene_sign1 = np.load(os.path.join(self.path, f'scene_sign_{b_idx}_{j}.npy'))
    if self.norm:
        points0 = normalize(points0)
        points1 = normalize(points1)
        features0 = normalize(features0)
        features1 = normalize(features1)

    return points0, points1, features0, features1, match_index, (scene_sign0,scene_sign1)

class MHADataset(torch.utils.data.Dataset):
  def __init__(self, key, norm=False):
    self.key = key
    self.path = f'./mha_data{key}'
    path_ls = os.listdir(self.path)
    self.files = [file for file in path_ls if 'points' in file]
    self.norm = norm



  def reset_seed(self, seed=0):
    print(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    """_summary_

    Args:
        idx (_type_): _description_

    Returns:
        (points0, points1, features0, features1, match_index)
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        (n, 3), (m, 3), (n, 768), (m, 768), (p, 2)
    """
    file = self.files[idx]
    # parttern = r"_(\d+).npy$"
    # mm = re.search(parttern, file)
    # b_idx = int(mm.group(1))
    points = np.load(os.path.join(self.path, file))
    features = np.load(os.path.join(self.path, file.replace('points', 'features')))

    if self.norm:
        points = normalize(points)
        features = normalize(features)

    return points, features

def default_collate_pair_fn(list_data):
    points_ls0, points_ls1, features_ls0, features_ls1, match_index_ls = list(zip(*list_data))

    points0 = np.stack(points_ls0, axis=0)
    points1 = np.stack(points_ls1, axis=0)
    features0 = np.stack(features_ls0, axis=0)
    features1 = np.stack(features_ls1, axis=0)
    match_index = np.stack(match_index_ls, axis=0)
    return {
        'sinput0_C': points0,
        'sinput0_F': features0,
        'sinput1_C': points1,
        'sinput1_F': features1,
        'correspondences': match_index,
    }

def collate_fn_lonely(data):
    points0, points1, features0, features1, match_index ,(i,j)= data
    return {
        'input0_P': torch.from_numpy(points0),
        'input0_F': torch.from_numpy(features0),
        'input1_P': torch.from_numpy(points1),
        'input1_F': torch.from_numpy(features1),
        'correspondences': torch.from_numpy(match_index),
        'index': (i,j)
    }

def collate_fn_MHA(data):
    points_ls = []
    features_ls = []
    for item in data:
        points, features = item
        #### we use cm 
        points_ls.append(torch.from_numpy(points) * 100.)
        features_ls.append(torch.from_numpy(features))
    return points_ls , features_ls
    

def get_loader(key,  norm=False, shuffle=True):
    dset = MatchPairDataset(key=key, norm=norm)
    collate_pair_fn = collate_fn_lonely
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=None,
        shuffle=shuffle,
        collate_fn=collate_pair_fn,)
    return loader

def get_loader_MHA(key,  norm=False, shuffle=True, batch_size=2):
    dset = MHADataset(key=key, norm=norm)
    collate_pair_fn = collate_fn_MHA
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pair_fn,
        drop_last=True)
    return loader

if __name__ == '__main__':

    dset = MatchPairDataset(key=0)
    # collate_pair_fn = default_collate_pair_fn
    # batch_size = 4

    # loader = torch.utils.data.DataLoader(
    #     dset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate_pair_fn,
    #     drop_last=True)

    ## we can just use the unbatched version
    loader = torch.utils.data.DataLoader(dset, batch_size=None, shuffle=True)