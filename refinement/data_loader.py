import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import re

           
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