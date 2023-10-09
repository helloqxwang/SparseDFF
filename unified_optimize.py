import torch
import numpy as np
import time
import random
import argparse
from scipy.spatial.transform import Rotation
from prune import get_points_features_from_mesh, get_points_features_from_real
from optimize.alignment import Hand_AlignmentCheck, Gripper_AlignmentCheck
import cv2
import open3d as o3d
import skimage
import os
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
  
)
from omegaconf import DictConfig, OmegaConf, open_dict

class home_made_feature_interpolator:

    def __init__(self, points:np.ndarray, features:np.ndarray, device = None) -> None:
        """Initialize the interpolator

        Args:
            points (np.ndarray): (n, 3)
            features (np.ndarray): (n, dim)
            device (_type_, optional): the device used for torch. Defaults to None(auto-detect).
        """
        if device:
            self.dev = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.dev = torch.device('cuda:0')
            else:
                self.dev = torch.device('cpu')
        self.sigma = 0.01
        self.points = torch.from_numpy(points).to(torch.float32).to(self.dev)
        print('points_mean:', self.points.mean(dim=0))
        # self.points = self.points - self.points.mean(dim=0)
        self.features = torch.from_numpy(features).to(torch.float32).to(self.dev)
    
    def get_points(self)->np.ndarray:
        return self.points.cpu().numpy()
    
    def predict(self, query_points:torch.Tensor)->torch.Tensor:
        """
            Get the features of the query points
        params:
            query_points (torch.Tensor): coordinates of the query points(batch_size, num_query_points, dim)
        
        return:
            interpolated_features (torch.Tensor): interpolated features of the query points (batch_size, num_query_points, dim)
        """
        # from pdb import set_trace; set_trace()
        dim = self.points.shape[1]
        b, n, _ = query_points.shape
        query_points = query_points.reshape(-1, dim)
        num_points = self.points.shape[0]
        num_query_points = query_points.shape[0]
        # show_pc(self.points.cpu().detach().numpy(), query_points.cpu().detach().numpy())
        
        # 扩展points和query_points，使其shape变为(num_query_points, num_points, dim)
        # points_exp = self.points.unsqueeze(0).repeat((num_query_points, 1, 1))
        # query_points_exp = query_points.unsqueeze(1).repeat((1, num_points, 1))
        points_exp = self.points[None, :, :]
        query_points_exp = query_points[:, None, :]
        
        # 计算query_points和points之间的欧氏距离，shape为(num_query_points, num_points)
        # dist min max 0.2 0.6
        dists = torch.norm((points_exp - query_points_exp), dim=-1)
        if dists.isnan().any():
            # from pdb import set_trace;set_trace()
            raise ValueError('nan in dists')

        # from torch.distributions.normal import Normal
        # gau = Normal(0, self.sigma)
        # weights = gau.log_prob(dists)
        weights = 1 / (dists + 1e-10)**2
        if weights.isnan().any():
            raise ValueError('nan in weights')
        
        # 对每个query_point，计算其特征的加权平均，shape为(num_query_points, num_features)
        # print(weights.dtype, self.features.dtype)
        if self.features.isnan().any():
            raise ValueError('nan in self.features')
        interpolated_features = torch.mm(weights, self.features) / torch.sum(weights, dim=1, keepdim=True)
        if interpolated_features.isnan().any():
            raise ValueError('nan in interpolated_features')
        
        return interpolated_features.reshape(b, n, -1)
class Dino_Processor:
    def __init__(self, conf, name, mode) -> None:
        self.conf = conf
        seed = conf.seed
        self.name = name
        self.mode = mode
        np.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)
        if conf.device:
            self.device = torch.device(conf.device)
        else:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        points1, features1, self.color_ref1, self.points_vis1, self.color_vis1, _ = get_points_features_from_real(path=conf.data1,
                                                            extrinsics_path=conf.extrinsics_path, key=0,
                                                            dis_threshold=conf.dis_threshold, quotient_threshold=conf.quotient_threshold, 
                                                            method=conf.method,verbose=conf.verbose, model_path=conf.model_path,
                                                            scale=conf.scale, name=self.name, p0=conf.img_preprocess[0])
        points2, features2, self.color_ref2, self.points_vis2, self.color_vis2, self.points_ref2 = get_points_features_from_real(path=conf.data2, 
                                                               extrinsics_path=conf.extrinsics_path, key=1, 
                                                               dis_threshold=conf.dis_threshold, quotient_threshold=conf.quotient_threshold, 
                                                               method=conf.method, verbose=conf.verbose, model_path=conf.model_path,
                                                               scale=conf.scale, name=self.name, p1=conf.img_preprocess[1])

        self.points1, self.features1 = points1.cpu().numpy(), features1.cpu().numpy()
        self.points2, self.features2 = points2.cpu().numpy(), features2.cpu().numpy()

        if conf.verbose:
            print('points1: ', self.points1.shape)
            print('points2: ', self.points2.shape)
            print('features1: ', self.features1.shape)
            print('features2: ', self.features2.shape)
        self.interpolator1 = home_made_feature_interpolator(self.points1, self.features1)
        self.interpolator2 = home_made_feature_interpolator(self.points2, self.features2)
    
    def process(self):
        
        if self.mode == 'hand':
            alignment = Hand_AlignmentCheck(self.interpolator1, self.interpolator2, self.points1, self.points2,
                                                self.color_ref1, self.color_ref2,
                                                self.points_vis1, self.points_vis2,
                                                self.color_vis1, self.color_vis2,
                                                self.points_ref2,
                                                trimesh_viz=self.conf.visualize, opt_iterations=self.conf.alignment.opt_iterations, 
                                                opt_nums=self.conf.hand_model.pt_nums, tip_aug=self.conf.hand_model.tip_aug,
                                                name=os.path.split(self.conf.data1)[-1])
        elif self.mode == 'gripper':
            alignment = Gripper_AlignmentCheck(self.interpolator1, self.interpolator2, self.points1, self.points2,
                                                self.color_ref1, self.color_ref2,
                                                self.points_vis1, self.points_vis2,
                                                self.color_vis1, self.color_vis2,
                                                self.points_ref2,
                                                trimesh_viz=self.conf.visualize, opt_iterations=self.conf.alignment.opt_iterations, 
                                                opt_nums=self.conf.hand_model.pt_nums, tip_aug=self.conf.hand_model.tip_aug,
                                                name=os.path.split(self.conf.data2)[-1])
        else:
            raise NotImplementedError
        alignment.sample_pts()

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.yaml')
    argparser.add_argument('--name', type=str, default='m0')
    args = argparser.parse_args()
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)
    dino_processor = Dino_Processor(conf, args.name, conf.mode)
    dino_processor.process()
    end_time = time.time()
    print('Whole Time: ', end_time - start_time)