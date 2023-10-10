import os
import torch
import pytorch_kinematics as pk
import trimesh
import pytorch3d.structures
import pytorch3d.ops
import torch.nn.functional as F
import numpy as np
# from torchsdf import index_vertices_by_faces, compute_sdf
import json
from scipy.spatial.transform import Rotation 
from prune import pt_vis

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3

    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = middle
    y = orthmid
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_matrix_ori(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack((a1, a2, a3), dim=-2)


class GripperModel:
    def __init__(self, stl_path='/home/user/wangqx/stanford/2F85_Opened_20190924.stl', n_surface_points=1000, device=None):
        mesh = trimesh.load_mesh(stl_path)
        self.mesh:trimesh.Trimesh = mesh.apply_scale(0.001)
        self.n_surface_points = n_surface_points
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        mesh = pytorch3d.structures.Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)], faces=[torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)])
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.n_surface_points)
        surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.n_surface_points)[0][0]
        surface_points.to(dtype=float, device=self.device)
        self.surface_points = surface_points
        # the first 3 are the x,y,z of the center of the gripper
        # the next 6 are the 6D rotation representation
        self.global_translation = None
        self.global_rotation = None

    
    def set_parameters(self, gripper_pose:torch.Tensor):
        """
        Batch sytle pose setting 
        Args:
            gripper_pose (torch.Tensor): (n, 9) (x, y, z, 6D rotation representation)
        """
        self.global_translation = gripper_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(gripper_pose[:, 3:9])
        # print('self.global_rotation', self.global_rotation.shape)
    
    def get_surface_points(self):
        """
        Get surface points
        
        Returns
        -------
        points: (B, `n_surface_points`, 3)
            surface points
        """
        batch_size = self.global_translation.shape[0]
        points = self.surface_points.expand(batch_size, self.n_surface_points, 3)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_trimesh_data(self, i):
        v = self.mesh.vertices
        f = self.mesh.faces
        v_ = v @ self.global_rotation[i].transpose(0, 1).detach().cpu().numpy() + self.global_translation[i].detach().cpu().numpy()
        mesh = trimesh.Trimesh(v_, f)
        return mesh
    
    def save_pose(self, path=None, gripper_pose:torch.Tensor=None, idx=0):
        """
        Save gripper pose to a file or return the pose as np.ndarray
        when directly save the poses, assume the gripper_pose is a batch of poses and we only save one of them. (idx)
        
        Parameters
        ----------
        path: str
            if not None,
                path to save file
            else:
                return the pose as np.ndarray
        gripper_pose: 
            (B, 3+3) torch.Tensor (not detach / on cuda device is also ok)
        """
        if isinstance(gripper_pose, torch.Tensor):
            gripper_pose = gripper_pose.detach().cpu().numpy()
            
        translation = gripper_pose[:, 0:3]
        rotation = robust_compute_rotation_matrix_from_ortho6d(torch.from_numpy(gripper_pose[:, 3:9]))
        rot_vec = Rotation.from_matrix(rotation.cpu().numpy()).as_rotvec()
        # from pdb import set_trace; set_trace()
        pose:np.ndarray = np.concatenate([translation, rot_vec], axis=-1)
        if path is None:
            return pose
        else:
            if pose.ndim == 2:
                pose = pose[idx]
            assert pose.ndim == 1
            np.save(path, pose) 

