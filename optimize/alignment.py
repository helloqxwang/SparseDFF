import os
import torch
import numpy as np
import trimesh
import copy
import plotly.graph_objects as go

import open3d as o3d
from optimize.hand_model import HandModelMJCF
from optimize.gripper_model import GripperModel
from scipy.spatial.transform import Rotation as R

from matplotlib import cm
from camera.camera_tools import vis_color_pc
from camera.camera_tools import read_hand_arm
from optimize.hand_model import robust_compute_rotation_matrix_from_ortho6d

def trimesh_show(np_pcd_list, mesh_list, color_add_list=None, color_list=None, rand_color=False, show=True, name=None):
    colormap = cm.get_cmap('brg', len(np_pcd_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(np_pcd_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(np_pcd_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
        else:
            color_list = colors
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))
        if color_add_list is not None:
            if i == 0:
                tpcd.colors = color_add_list[0]
            elif i == 2:
                tpcd.colors = color_add_list[1]

        tpcd_list.append(tpcd)

    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    scene.add_geometry(mesh_list)
    
    if show:
        scene.show()
    if name:
        img = scene.save_image((480, 480))
        with open('./data/result.png', 'wb') as f:
            f.write(img)
    return scene

class Hand_AlignmentCheck:
    def __init__(self, interpolator1, interpolator2, pcd1, pcd2, 
                 color_ref1:np.ndarray=None, color_ref2:np.ndarray=None, 
                 points_vis1:np.ndarray=None, points_vis2:np.ndarray=None,
                 colors_vis1:np.ndarray=None, colors_vis2:np.ndarray=None,
                 points_ref:torch.Tensor=None, skip_inverse:bool = False,
                 opt_iterations=1500, opt_nums=500, 
                 trimesh_viz=False, hand_file = "./mjcf/shadow_hand_vis.xml", 
                 tip_aug=None, name=None):
        ### load the model and set the params
        self.interpolator1, self.interpolator2 = interpolator1, interpolator2
        self.opt_iterations = opt_iterations
        self.viz = trimesh_viz
        self.name = name
        self.skip_inverse = skip_inverse
        self.perturb_scale = 0.001
        self.perturb_decay = 0.5
        self.n_opt_pts = opt_nums
        self.points_vis1 = points_vis1
        self.points_vis2 = points_vis2
        self.color_vis1 = colors_vis1
        self.color_vis2 = colors_vis2
        self.points_ref:torch.Tensor = points_ref
        self.pcd1 = pcd1
        self.pcd2 = pcd2
        self.color_ref1, self.color_ref2 = color_ref1, color_ref2
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        ### load the hand_model
        self.hand = HandModelMJCF(hand_file, "mjcf/meshes", n_surface_points=self.n_opt_pts, device=self.dev, tip_aug=tip_aug, ref_points=torch.from_numpy(points_vis2))

        self.loss_fn = torch.nn.L1Loss()

    ###### can sample some pt from the reference frame and then return the best corresponding points in the test frame
    def sample_pts(self, hand_gt_pose: torch.Tensor = None):
        hand_gt_pose = read_hand_arm(name='monkey0')       
        hand_gt_pose[:, 2] += 0.12
        hand_gt_pose[:, 1] -= 0.10
        hand_gt_pose[:, 0] -= 0.20
        hand_gt_pose[:, 3:9] = np.array([0, -1, 0, 0, 0, 1])[None, :]   
        hand_gt_pose = torch.from_numpy(hand_gt_pose).float().to(self.dev)
        self.hand.set_parameters(hand_gt_pose, retarget=False, robust=True)
        vquery_mesh = self.hand.get_trimesh_data(0)
        hand_gt:np.ndarray = self.hand.get_surface_points()[0].detach().cpu().numpy()
        self.hand.save_pose('./data/des_ori.npy', hand_gt_pose, False, False)
        # trimesh_show([self.pcd1 ], [vquery_mesh], show=self.viz, name=self.name, color_add_list=[self.color_ref1,])
        reference_query_pts = hand_gt
        # exit()

        reference_model_input = {}
        ref_query_pts = torch.from_numpy(reference_query_pts).float().to(self.dev)
        ### the pc of the reference shape
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        # get the descriptors for these reference query points
        reference_act_hat = self.interpolator1.predict(reference_model_input['coords']).detach()

        best_loss = np.inf
        best_idx = 0
        M = 10

        motion = (torch.rand(M, 31)*0.03).float().to(self.dev)        
        motion[:, 2] = float(self.pcd2[:, 2].max()) + (torch.rand(M)*0.1 + 0.2)[None, :].float().to(self.dev)
        motion[:, 0:2] = (torch.rand(M, 2)*0.2).float().to(self.dev) 
        motion[:, 3:9] = torch.from_numpy(np.array([0,-1,0,0,0,1])[None, :].repeat(M, axis=0)).to(self.dev)
        
        ori_rotm = torch.from_numpy(np.array([0., 0 ,-1,-1,0,0,0,1,0]).reshape((3,3))).to(self.dev).to(torch.float32)
        motion.requires_grad_()
        opt = torch.optim.Adam([motion], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.opt_iterations/ 50, eta_min=1e-4)  

        loss_values = []

        # run optimization
        pcd_traj_list = {}
        execution_traj_list = {}
        for i in range(M):
            pcd_traj_list[i] = []
            execution_traj_list[i] = []
        for i in range(self.opt_iterations):
            self.hand.set_parameters(motion)
            X_new_ori = self.hand.get_surface_points()
            # vis_color_pc(X_new_ori[0].detach().cpu().numpy(), None, 0.1 , save=False, rot=ori_rotm.cpu().numpy())
            # exit()
            X_new = X_new_ori + torch.rand_like(X_new_ori) * self.perturb_scale 
            self.perturb_scale *= self.perturb_decay

            ######################### stuff for visualizing the reconstruction ##################33

            motion_save = self.hand.save_pose(path=None, hand_pose=motion)
            for jj in range(M):
                X_np = X_new[jj].detach().cpu().numpy()
                centroid = np.mean(X_np, axis=0)
                pcd_traj_list[jj].append(centroid)
                if len(execution_traj_list[jj]) == 0 or np.linalg.norm((execution_traj_list[jj][-1][:3] - motion[jj].detach().cpu().numpy()[:3])) > 0.01:
                    execution_traj_list[jj].append(motion_save[jj])
                    
                    
            ###############################################################################

            act_hat = self.interpolator2.predict(X_new)
            t_size = reference_act_hat.size()


            losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]
            losses = torch.stack(losses)
            
            
            # distances = self.hand.cal_distance(self.points_ref.expand(M, -1, -1))
            # distances[distances <= 0] = 0
            # E_pen = distances.sum(-1)
            # E_spen = self.hand.self_penetration()
            # E_joint = self.hand.get_E_joints()
            # losses += E_pen * 1e-1 + E_spen * 1e-2 + E_joint * 1e-2

            rot_ms = robust_compute_rotation_matrix_from_ortho6d(motion[:, 3:9])
            rot_ms = rot_ms.reshape((M, 3, 3)).to(torch.float32)
            
            x_axis_ori_hand = ori_rotm[:, 0][None, ...].repeat(M, 1)
            y_axis_ori_hand = ori_rotm[:, 1][None, ...].repeat(M, 1)
            z_axis_ori_hand = ori_rotm[:, 2][None, ...].repeat(M, 1)

            z_axis_object = rot_ms[:, :, 2]
            z_axis_y = z_axis_object - torch.sum(z_axis_object * ori_rotm[:, 0][None, ...].repeat(M, 1), dim=-1, keepdim=True) * ori_rotm[:, 0][None, ...].repeat(M, 1)
            z_axis_y = z_axis_y / (torch.norm(z_axis_y, dim=1, keepdim=True) + 1e-8)
        
            roll = torch.arccos(torch.clamp(torch.sum(z_axis_y * z_axis_ori_hand, dim=-1), -1 + 1e-4, 1-1e-4))
            sign_roll = torch.sign(torch.sum(z_axis_y * y_axis_ori_hand, dim=-1))
            roll = roll * sign_roll

            roll[roll < np.pi / 6 * 0.8] = 0
            roll = torch.abs(roll)
            roll[roll >= np.pi / 6 * 0.8]  -= np.pi / 6 * 0.8
            

            z_axis_x = z_axis_object - torch.sum(z_axis_object * ori_rotm[:, 1][None, ...].repeat(M, 1), dim=-1, keepdim=True) * ori_rotm[:, 1][None, ...].repeat(M, 1)
            z_axis_x = z_axis_x / (torch.norm(z_axis_x, dim=1, keepdim=True) + 1e-8)
            pitch = torch.arccos(torch.clamp(torch.sum(z_axis_x * z_axis_ori_hand, dim=-1), -1 + 1e-4, 1-1e-4))
            sign_pitch = torch.sign(torch.sum(z_axis_x * x_axis_ori_hand, dim=-1))
            pitch = pitch * sign_pitch
            pitch[pitch.abs() < torch.pi / 6] = 0
            pitch = torch.abs(pitch)
            pitch[pitch >= torch.pi / 6] -= torch.pi / 6     

            if roll.any():
                losses += roll * 1e-1
            if pitch.any():
                losses += pitch * 1e-1

            loss = torch.mean(losses)
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                print(f'i: {i}, losses: {loss_str}')

            loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            if i % 50 == 0:
                scheduler.step()
            
        if self.skip_inverse:
            rot_sixd = self.hand.save_pose(path=None, hand_pose=motion)[:, 3:9]
            rot_m = np.eye(3)[None, ...].repeat(M, axis=0)
            rot_m[:, 0] = rot_sixd[:, 0:3]
            rot_m[:, 1] = rot_sixd[:, 3:6]
            rot_m[:, 2] = np.cross(rot_m[:, 0], rot_m[:, 1])
            rot_zxy = R.from_matrix(rot_m).as_euler('zxy', degrees=False)
            rot_ini = np.array([ 1.57079633, -1.57079633,  0.        ])[None, :]
            rot_index = np.abs(rot_zxy - rot_ini) < 0.6
            min_loss = torch.min(losses[rot_index]).item()
            best_idx = torch.nonzero(losses == min_loss).squeeze().item()
        else:
            best_idx = torch.argmin(losses).item()

        best_loss = losses[best_idx]
        print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        best_X = X_new[best_idx].detach().cpu().numpy()

        offset = np.array([0.7, 0, 0])
        vpcd1 = copy.deepcopy(self.pcd1)
        vquery1 = copy.deepcopy(reference_query_pts)
        self.hand.set_parameters(motion)
        X_mesh = self.hand.get_trimesh_data(best_idx)

        vpcd1 += offset
        vquery1 += offset
        vquery_mesh.apply_translation(offset)
        self.hand.save_pose('./data/des_final.npy', motion[best_idx][None, ...])

        best_execution_traj = np.stack(execution_traj_list[best_idx], axis=0)
        np.save('./data/execution_traj.npy', best_execution_traj)
        np.save('./data/pcd_traj.npy', pcd_traj_list[best_idx])
        np.save('./data/best_X.npy', best_X)
        vquery_mesh.export('./data/vquery_mesh.stl', file_type='stl')
        X_mesh.export('./data/X_mesh.stl', file_type='stl')

        if self.color_ref1 is not None and self.color_ref2 is not None:
            trimesh_show([vpcd1, vquery1 , self.pcd2, best_X, pcd_traj_list[best_idx]], [vquery_mesh, X_mesh], show=self.viz, name=self.name, color_add_list=[self.color_ref1, self.color_ref2])
        else:
            trimesh_show([vpcd1, vquery1 , self.pcd2, best_X, pcd_traj_list[best_idx]], [vquery_mesh, X_mesh], show=self.viz, name=self.name)


class Gripper_AlignmentCheck:
    def __init__(self, interpolator1, interpolator2, pcd1, pcd2, 
                 color_ref1:np.ndarray=None, color_ref2:np.ndarray=None, 
                 points_vis1:np.ndarray=None, points_vis2:np.ndarray=None,
                 colors_vis1:np.ndarray=None, colors_vis2:np.ndarray=None,
                 points_ref:torch.Tensor=None, skip_inverse:bool = False,
                 opt_iterations=1500, opt_nums=500, 
                 trimesh_viz=False, hand_file = "mjcf/shadow_hand_wrist_free.xml", 
                 tip_aug=None, name=None):
        ### load the model and set the params
        self.interpolator1, self.interpolator2 = interpolator1, interpolator2
        self.opt_iterations = opt_iterations
        self.viz = trimesh_viz
        self.name = name
        self.skip_inverse = skip_inverse
        self.perturb_scale = 0.001
        self.perturb_decay = 0.5
        self.n_opt_pts = opt_nums
        self.points_vis1 = points_vis1
        self.points_vis2 = points_vis2
        self.color_vis1 = colors_vis1
        self.color_vis2 = colors_vis2
        self.points_ref:torch.Tensor = points_ref
        ### let the pcd be center on the origin
        self.pcd1 = pcd1
        self.pcd2 = pcd2
        self.color_ref1, self.color_ref2 = color_ref1, color_ref2
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        self.gripper = GripperModel(stl_path='/home/user/wangqx/stanford/2F85_Opened_20190924.stl', n_surface_points=self.n_opt_pts, device=self.dev)

        self.loss_fn = torch.nn.L1Loss()

    ###### can sample some pt from the reference frame and then return the best corresponding points in the test frame
    def sample_pts(self, hand_gt_pose: torch.Tensor = None):


        gripper_gt_pose = torch.zeros((1, 9)).float().to(self.dev)
        gripper_gt_pose[:, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0]).float().to(self.dev)[None, :]
        self.gripper.set_parameters(gripper_gt_pose)
        vquery_mesh = self.gripper.get_trimesh_data(0)
        gripper_gt:np.ndarray = self.gripper.get_surface_points()[0].detach().cpu().numpy()
        self.gripper.save_pose('./data/des_gripper_ori.npy', gripper_gt_pose)
        # trimesh_show([self.pcd1 ], [vquery_mesh], show=self.viz, name=self.name, color_add_list=[self.color_ref1,])
        reference_query_pts = gripper_gt
        # exit()

        reference_model_input = {}
        ref_query_pts = torch.from_numpy(reference_query_pts).float().to(self.dev)
        ### the pc of the reference shape
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        # get the descriptors for these reference query points
        reference_act_hat = self.interpolator1.predict(reference_model_input['coords']).detach()


        best_loss = np.inf
        best_idx = 0
        M = 10 

        motion = (torch.rand(M, 9)*0.3).float().to(self.dev)        

        motion[:, 2] = float(self.pcd2[:, 2].max()) + (torch.rand(M)*0.1)[None, :].float().to(self.dev)
        motion[:, 0:2] = (torch.rand(M, 2)*0.4).float().to(self.dev) - 0.2
        motion.requires_grad_()
        opt = torch.optim.Adam([motion], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.opt_iterations/ 50, eta_min=1e-4)  

        loss_values = []

        # run optimization
        pcd_traj_list = {}
        execution_traj_list = {}
        for i in range(M):
            pcd_traj_list[i] = []
            execution_traj_list[i] = []
        for i in range(self.opt_iterations):
            self.gripper.set_parameters(motion)
            X_new_ori = self.gripper.get_surface_points()
            # vis_color_pc(X_new_ori[0].detach().cpu().numpy(), None, 0.1 , save=False, rot=ori_rotm.cpu().numpy())
            # exit()
            X_new = X_new_ori + torch.rand_like(X_new_ori) * self.perturb_scale 
            self.perturb_scale *= self.perturb_decay

            ######################### stuff for visualizing the reconstruction ##################33

            motion_save = self.gripper.save_pose(path=None, gripper_pose=motion)
            for jj in range(M):
                X_np = X_new[jj].detach().cpu().numpy()
                centroid = np.mean(X_np, axis=0)
                pcd_traj_list[jj].append(centroid)
                if len(execution_traj_list[jj]) == 0 or np.linalg.norm((execution_traj_list[jj][-1][:3] - motion[jj].detach().cpu().numpy()[:3])) > 0.01:
                    execution_traj_list[jj].append(motion_save[jj])
            ###############################################################################

            act_hat = self.interpolator2.predict(X_new)
            t_size = reference_act_hat.size()


            losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]
            losses = torch.stack(losses)
            loss = torch.mean(losses)
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                print(f'i: {i}, losses: {loss_str}')

            loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            if i % 50 == 0:
                scheduler.step()
        
        best_idx = torch.argmin(losses).item()
        best_loss = losses[best_idx]
        print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        best_X = X_new[best_idx].detach().cpu().numpy()

        offset = np.array([1.7, 0, 0])
        vpcd1 = copy.deepcopy(self.pcd1)
        vquery1 = copy.deepcopy(reference_query_pts)
        self.gripper.set_parameters(motion)
        X_mesh = self.gripper.get_trimesh_data(best_idx)

        vpcd1 += offset
        vquery1 += offset
        vquery_mesh.apply_translation(offset)
        motion_best = motion[best_idx].detach().cpu().numpy()
        self.gripper.save_pose('./data/des_final.npy', motion[best_idx][None, ...])

        best_execution_traj = np.stack(execution_traj_list[best_idx], axis=0)
        print(best_execution_traj.shape)
        np.save('./data/execution_traj.npy', best_execution_traj)

        if self.color_ref1 is not None and self.color_ref2 is not None:
            trimesh_show([vpcd1, vquery1 , self.pcd2, best_X, pcd_traj_list[best_idx], best_execution_traj[-40:, :3]], [vquery_mesh, X_mesh], show=self.viz, name=self.name, color_add_list=[self.color_ref1, self.color_ref2])
        else:
            trimesh_show([vpcd1, vquery1 , self.pcd2, best_X, pcd_traj_list[best_idx]], [vquery_mesh, X_mesh], show=self.viz, name=self.name)