import torch
import numpy as np
import time
import random
import argparse
from prune import get_points_features_from_real
from optimize.alignment import Hand_AlignmentCheck, Gripper_AlignmentCheck
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from pyvirtualdisplay import Display
import pyglet

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
        points_exp = self.points[None, :, :]
        query_points_exp = query_points[:, None, :]
        
        dists = torch.norm((points_exp - query_points_exp), dim=-1)
        if dists.isnan().any():
            raise ValueError('nan in dists')

        weights = 1 / (dists + 1e-10)**2
        if weights.isnan().any():
            raise ValueError('nan in weights')
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
    start_time = time.time()
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./visualize', exist_ok=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.yaml')
    argparser.add_argument('--name', type=str, default='test')
    args = argparser.parse_args()
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)
    if conf.visualize == False:
        display = Display(visible=0, size=(1024, 768))
        display.start()
        pyglet.options['shadow_window'] = False
        pyglet.options['display'] = display.display
    dino_processor = Dino_Processor(conf, args.name, conf.mode)
    dino_processor.process()
    end_time = time.time()
    print('Whole Time: ', end_time - start_time)