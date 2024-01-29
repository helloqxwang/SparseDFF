import numpy as np
import cv2
# from prune.tools import  get_dino_features
from typing import List
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from camera.sam import Sam_Detector, vis_mask_image
import torch
import open3d as o3d
import yaml
import json
from scipy.spatial.transform import Rotation 
import os
import skimage

CAM = {
    "cam0": '000299113912',
    "cam1": '000272313912',
    "cam2": '000285613912',
    "cam3": '000262413912',
}
CAM_INDEX = [CAM['cam0'], CAM['cam1'], CAM['cam2'], CAM['cam3']]

def get_dino_features(img_raw:np.ndarray, scale:int=3)->torch.Tensor:
    """get dino features for only one img

    Args:
        img (np.ndarray): (h, w, 3)
        scale (int, optional): _description_. Defaults to 3.

    Returns:
        torch.Tensor: (h, w, F)
    """
    img_raw = img_raw.astype('float32') / 255.
    img_raw = skimage.img_as_float32(img_raw)
    if os.path.isdir('./thirdparty_module/dinov2'):
        torch.hub.set_dir( "./")
        model = torch.hub.load('./thirdparty_module/dinov2', 'dinov2_vitb14', source='local', pretrained=False).cuda()
        model.load_state_dict(torch.load('./thirdparty_module/dinov2_vitb14_pretrain.pth'))
    else:
        torch.hub.set_dir("../")
        model = torch.hub.load('../thirdparty_module/dinov2', 'dinov2_vitb14', source='local', pretrained=False).cuda()
        model.load_state_dict(torch.load('../thirdparty_module/dinov2_vitb14_pretrain.pth'))
    h, w = img_raw.shape[0] // 14 * 14,  img_raw.shape[1] // 14 * 14
    img = skimage.transform.resize(
                img_raw,
                (img_raw.shape[0] // 14 * 14 , img_raw.shape[1] // 14 * 14 )
            ).astype('float32')
    img = torch.from_numpy(img)
    img = img[None, :].cuda()
    # print('Picture for Dino size:', img.shape)
    with torch.no_grad():
        ### (batch size, 3, height, width)
        ret = model.forward_features(img.permute(0, 3, 1, 2))
    features = ret['x_norm_patchtokens']
    N, _, F = features.shape
    ### (height, width, features) torch.Tensor np.float32
    features = features.reshape(img.shape[1] // 14, img.shape[2] // 14, F).permute(2, 0, 1)
    features = torch.nn.functional.interpolate(features.unsqueeze(0), size=(img_raw.shape[0] // scale , img_raw.shape[1] // scale ), mode='bilinear', align_corners=False)
    features = features.squeeze(0).permute(1, 2, 0)
    return features

def depth2pt_K_numpy(depths:np.ndarray, K:np.ndarray , R:np.ndarray, xyz_images=True)->np.ndarray:
    """
    The batch_K_version  of depth2pt, but without the auto-scale of the camera parameters.
    So K MUST MATCH the depths.

    If xyz_images is True: 
        Translate a pile of depth images ( N x H x W) to xyz images ( N x H x W x 3)
    Else: 
        Translate a pile of depth images ( N x H x W) to 3D points ( Num x 3) filtered by the depth < 0
        bath_sign (Num, ) is the Sign of the batch, noting every selected points which batch it belongs to
        zero_filter (Num, ) is the filter of the points which depth > 0

    Args:
        depths (np.ndarray): (n, h, w)
        K (np.ndarray): the intrinsics (n, 3, 3)
        R (np.ndarray): the rotation matrix (n, 4, 4)

    Returns:
        If xyz_images is True: 
            xyz_imges: np.ndarray (n, h, w, 3)
        Else: 
            points: np.ndarray (num, 3)
            batch_sign: np.ndarray (num, )
            zero_filter: np.ndarray (n*h*w, )
    """
    batch_size, h, w = depths.shape
    fx = K[:, 0, 0][:, None, None]
    fy = K[:, 1, 1][:, None, None]
    x_offset = K[:, 0, 2][:, None, None]
    y_offset = K[:, 1, 2][:, None, None]
    ### 对于index(a0, a1, ..., an) shape：(n, a0, a1, ..., an)
    ### transpose 改变了坐标的打包方式，从"所有行坐标在一起，所有列坐标在一起"变为"每个点的行列坐标在一起"。
    ### 事实上，indice 坐标是以通道(channel)的方式分开的，即第一个通道存储所有的行坐标，第二个通道存储所有的列坐标。
    ### 第二维和第三维度就是行列坐标
    ### 但我们希望将点的所有横坐标放在一起处理，所有纵坐标放在一起处理，所以要将通道的维度放到最后。
    # (3, batch_size, rey, resx) -> (batch_size, rey, resx, 3)
    # indices = torch.stack(torch.meshgrid(torch.arange(batch_size), torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).to(dtype=torch.float32).to(device)
    indices = np.indices((batch_size ,h, w), dtype=np.float32).transpose(1, 2, 3, 0)
    # depths[depths < 0] = 0
    z_e = depths
    x_e = (indices[..., -1] - x_offset) * z_e / fx
    y_e = (indices[..., -2] - y_offset) * z_e / fy
    homogenerous = np.ones((batch_size, h, w))
    xyz_img = np.stack([x_e, y_e, z_e, homogenerous], axis=-1)  # Shape: [n, H, W, 4]
    ### (n, 4, 4) * (n, h, w, 4) --> (n, h, w, 4)
    # xyz_img_trans = np.stack([np.matmul(R[i], xyz_img[i].reshape(-1, 4).T).T.reshape(h, w, 4) for i in range(R.shape[0])], axis=0)
    xyz_img_trans = np.matmul(R, xyz_img.reshape(xyz_img.shape[0], -1, 4).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(batch_size, h, w, 4)
    if xyz_images:
        return xyz_img_trans[..., :3]
    else:
        batch_sign = np.zeros((batch_size, h, w))
        for i in range(batch_size):
            batch_sign[i] = i + 1
        zero_filter = (depths != 0).reshape(-1)
        batch_sign = batch_sign.reshape(-1)[zero_filter]
        return xyz_img_trans[..., :3].reshape(-1, 3)[zero_filter], batch_sign, zero_filter

def load_cddi(data_path):
    """load colors, depths, distortions, intrinsics from a folder contain multicamera

    Args:
        data_path (str): path
    Return:
        colors, depths, distortion, intrinsics (np.ndarray) (cam_num, h, w, 3) for colors
    """
    if not os.path.isdir(data_path):
        raise ValueError("data_path should be a folder")
    colors_ls, depths_ls, distortion_ls, intrinsics_ls = [], [], [], []
    for serial_num in CAM_INDEX:
        path = os.path.join(data_path, serial_num)
        if not os.path.isdir(path):
            raise ValueError(f"Cannot find {path}")
        colors = np.load(os.path.join(path, 'colors.npy'))
        depths = np.load(os.path.join(path, 'depth.npy'))
        distortion = np.load(os.path.join(path, 'distortion.npy'))
        intrinsics = np.load(os.path.join(path, 'intrinsic.npy'))
        colors_ls.append(colors)
        depths_ls.append(depths)
        distortion_ls.append(distortion)
        intrinsics_ls.append(intrinsics)
    colors = np.stack(colors_ls, axis=0)
    depths = np.stack(depths_ls, axis=0)
    distortion = np.stack(distortion_ls, axis=0)
    intrinsics = np.stack(intrinsics_ls, axis=0)
    return colors, depths, distortion, intrinsics

def read_tranformation(data_path:str='./camera/transform.yaml'):
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


def get_extrinsics_from_json(path:str, transformation_path:str='./camera/transform.yaml'):
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
    if os.path.isfile(transformation_path):
        table2cam, cam2base = read_tranformation(transformation_path)
    else:
        table2cam, cam2base = read_tranformation('../camera/transform.yaml')
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

def get_foregroundmark(features:np.ndarray, threshold=0.3):
    """get the **foreground** mask using PCA and dino_feature

    Args:
        features (np.ndarray): (cam_num, H, W, dim)
        threshold (float, optional): _description_. Defaults to 0.6.
    """
    pca = PCA(n_components=1)
    reduced_feat = pca.fit_transform(features.reshape(-1, features.shape[-1]))
    norm_feat = minmax_scale(reduced_feat)
    norm_feat = norm_feat.reshape(features.shape[:-1])
    return norm_feat < threshold



def undistort(colors:np.ndarray, depths:np.ndarray, intrinsics:np.ndarray, distortion:np.ndarray)->(np.ndarray, np.ndarray):
    """get the undistorted depths and colors

    Args:
        colors:np.ndarray,  (num, h, w, 3)
        depths:np.ndarray,  (num, h, w)
        intrinsics:np.ndarray, (num, 3, 3)
        distortion:np.ndarray (num, 8)

    Returns:
        colors_undistort: np.ndarray (cam_num, h, w, 3)
        depth_undistort: np.ndarray (cam_num, h, w)
    """
    color_undistort_ls = []
    depth_undistort_ls = []
    for i in range(colors.shape[0]):
        map1, map2 = cv2.initUndistortRectifyMap(intrinsics[i], distortion[i], np.eye(3), intrinsics[i], (colors.shape[2], colors.shape[1]), cv2.CV_32FC1)
        color_undistort = cv2.remap(colors[i], map1, map2, cv2.INTER_NEAREST)
        depth_undistort = cv2.remap(depths[i], map1, map2, cv2.INTER_NEAREST)
        color_undistort_ls.append(color_undistort)
        depth_undistort_ls.append(depth_undistort)

    colors_undistort = np.stack(color_undistort_ls, axis=0)
    depths_undistort = np.stack(depth_undistort_ls, axis=0)
    
    return colors_undistort, depths_undistort

def get_distort_points(path:str, extrinsics:np.ndarray)->(np.ndarray, np.ndarray):
    from .camera_tools import get_extrinsics_from_json, load_color_pc,\
transform_points, vis_color_pc, save_colorpc, vis_img, load_depths,\
load_cddi
    points, colors = load_color_pc('./data/20230815_165149', mix=False)
    points = points.reshape(points.shape[0], -1, 3)
    points_trans = transform_points(points, extrinsics)
    return points_trans, colors

def get_index_from_range(points:np.ndarray, x=[-420, 420], y=[-560, 560], z=[-200, 800], return_mask:bool = False)->np.ndarray:
    """get the index of the points in the range

    Args:
        points (np.ndarray): (w ,h, 3)
        x
        y
        z

    Returns:
        index (tuple): (y_idx, x_idx)
    """
    mask = (points[..., 0] > x[0]) & (points[..., 0] < x[1]) & (points[..., 1] > y[0]) & (points[..., 1] < y[1]) & (points[..., 2] > z[0]) & (points[..., 2] < z[1])
    if return_mask:
        return mask
    index = np.nonzero(mask)
    return index

def line_dist(points:torch.tensor, line:torch.tensor)->torch.tensor:
        """compute the distance of the points to the line

        Args:
            point (torch.tensor): (num, 3)
            line (torch.tensor): (4, ) [a, b, c, d]

        Returns:
            dist (toch.tensor); (num, )
        """
        dist = np.matmul(points, line[:-1].reshape(3, 1)) + line[-1]
        return dist.squeeze()

def pipeline(data_path:str, extrinsics_path:str, scale:int=3, save:bool=True, name = 'mm', prune_method='sam', key:int=0, verbose:bool=True, samckp_path:str='./thirdparty_module/sam_vit_h_4b8939.pth')->(np.ndarray, np.ndarray, np.ndarray):
    """
    the pipeline of the data loading/capturing then processing
    
    The depths has shrinked according to the scale.

    Args:
        data_path (str): the path of the capture_3d
            if path is None:
                capture new data
        extrinsics_path (str): path of the extrinsics (json path)
        downsample_size (tuple): the down_sampled size of each image
        scale: the shrink scale
    Returns:
        points: (n, 3) torch.Tensor 
        features: (n, F) torch.Tensor
        colors: (n, 3) np.ndarray
        batch_sign: (n, ) torch.Tensor
        points_undistort: (n', ) np.ndarray used for points_ref


    """
    ### get extrinsics(world2cam) and world2base
    extrinsics, world2base = get_extrinsics_from_json(extrinsics_path)
    if data_path:
        colors_distort, depths_distort, distortion, intrinsics = load_cddi(data_path)
        colors, depths = undistort(colors_distort, depths_distort ,intrinsics, distortion) 
    else:
        from .capture_3d import capture_auto
        points_distort, colors_distort, depths_distort, intrinsics, distortion = capture_auto(save=save, name = name)
        colors, depths = undistort(colors_distort, depths_distort, intrinsics, distortion)

    colors_pile = colors[..., (2, 1, 0)]
    depths[depths < 0] = 0
    points_undistort = depth2pt_K_numpy(depths, intrinsics, np.linalg.inv(extrinsics), xyz_images=True)
    detector = Sam_Detector(sam_checkpoint=samckp_path)
    points_ls = []
    features_ls = []
    batch_sign_ls = []
    colors_ls = []
    ### attention! the unit now is mm
    for idx in range(points_undistort.shape[0]):
        points = points_undistort[idx]
        colors = colors_pile[idx]
        depth = depths[idx]

        if prune_method == 'sam':
            posi_num = 4
            posi_index = get_index_from_range(points, x=[-200, 200], y = [-200, 200], z=[20, 1200])
            posi_select_id = np.random.choice(len(posi_index[0]), posi_num // 2)
            posi_index_ = get_index_from_range(points, x=[-200, 200], y = [-200, 200], z=[5, 20])
            posi_select_id_ = np.random.choice(len(posi_index_[0]), posi_num // 2)
            
            
            posi_select_id = np.concatenate([posi_select_id, posi_select_id_])
            posi_index = np.array([[posi_index[1][i], posi_index[0][i]] for i in posi_select_id])
            

            neg_num = 2 
            if key == 0:
                neg_index = get_index_from_range(points, x=[-400, 400], y = [370, 460], z=[-10, 20])
                if neg_index[0].shape[0] < neg_num:
                    neg_index = get_index_from_range(points, x=[-400, 400], y = [- 370, - 460], z=[-150, 900])
            elif key == 1:
                neg_index = get_index_from_range(points, x=[-400, 400], y = [370, 460], z=[-10, 90])
                if neg_index[0].shape[0] < neg_num:
                    neg_index = get_index_from_range(points, x=[-400, 400], y = [- 370, - 460], z=[-150, 900])
            else:
                raise NotImplementedError
            neg_select_id = np.random.choice(len(neg_index[0]), neg_num)
            neg_index = np.array([[neg_index[1][i], neg_index[0][i]] for i in neg_select_id])

            ref_points = np.concatenate([posi_index, neg_index], axis=0)
            labels = np.array([1, 0]).repeat([posi_num, neg_num])
            if verbose:
                print('Color size:', colors.shape)
            mask_sam = detector.get_mask(colors, ref_points, labels)
            if save:
                vis_mask_image(colors, mask_sam, ref_points, labels, save_path=f'./data/sam{idx}.png')
            
            mask_physics = get_index_from_range(points, return_mask=True)
            mask = mask_sam & mask_physics
            index = np.nonzero(mask)
        elif prune_method == 'physics':
            mask_physics = get_index_from_range(points, x=[-455, 455], y=[-545, 545], z=[-200, 800],return_mask = True)
            mask = (depth!=0) & mask_physics
            index = np.nonzero(mask)
            if save:
                vis_mask_image(colors, mask, None, None, save_path=f'./data/physics{idx}.png')
        else:
            raise NotImplementedError
        
        bb = np.array([np.min(index[1]) , np.min(index[0]) , np.max(index[1]) , np.max(index[0]) ])
        pruned_colors = colors[bb[1]:bb[3], bb[0]:bb[2]]
        prune_points = points[bb[1]:bb[3], bb[0]:bb[2]]
        pruned_mask = mask[bb[1]:bb[3], bb[0]:bb[2]].astype('float32')
        pruned_depth = depth[bb[1]:bb[3], bb[0]:bb[2]]
        
        h, w, _ = pruned_colors.shape
        h, w = h // scale, w // scale

        features:torch.tensor = get_dino_features(pruned_colors, scale=scale)
        if save:
            cv2.imwrite(f'./data/dino_color{idx}.png', pruned_colors[..., (2, 1, 0)])
            np.save(f'./data/dino_features{idx}.npy', features.cpu().numpy())
        downsampled_points = cv2.resize(prune_points, (w, h), interpolation=cv2.INTER_NEAREST)
        downsampled_colors = cv2.resize(pruned_colors, (w, h), interpolation=cv2.INTER_NEAREST)
        downsampled_mask = cv2.resize(pruned_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype('bool')
        downsample_depth = cv2.resize(pruned_depth, (w, h), interpolation=cv2.INTER_NEAREST)
        downsampled_mask = downsampled_mask & (downsample_depth != 0)
        if verbose:
            print('Downsampled mask size:', downsampled_mask.shape)
            print('features size:', features.shape)

        masked_features = features[torch.from_numpy(downsampled_mask)]
        masked_points = downsampled_points[downsampled_mask]
        masked_colors = downsampled_colors[downsampled_mask]
        batch_sign = np.ones((masked_points.shape[0],)) * (idx + 1)
        
        if prune_method == 'physics':
            ### to prune the table top off
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(masked_points)
            plane_model, inliers = pcd.segment_plane(distance_threshold=10,
                                                ransac_n=3,
                                                num_iterations=2000)
            plane = np.array(plane_model) / np.linalg.norm(plane_model[:3])
            dist_plane = line_dist(masked_points, plane)
            distance = 20
            if np.count_nonzero(dist_plane < - distance) > np.count_nonzero(dist_plane >  distance):
                ### make sure the norm_vec of the plane point to the bear side
                plane = - plane
                index_prune_plane = dist_plane < - distance
            else:
                index_prune_plane = dist_plane >  distance
            masked_features = masked_features[index_prune_plane]
            masked_points = masked_points[index_prune_plane]
            masked_colors = masked_colors[index_prune_plane]
            batch_sign = batch_sign[index_prune_plane]

        batch_sign_ls.append(batch_sign)
        points_ls.append(masked_points)
        features_ls.append(masked_features)
        colors_ls.append(masked_colors)

    points = torch.from_numpy(np.concatenate(points_ls, axis=0).astype('float32')) / 1000. # from mm to m
    features = torch.cat(features_ls, axis=0)
    batch_sign = torch.from_numpy(np.concatenate(batch_sign_ls, axis=0))
    colors = np.concatenate(colors_ls, axis=0)

    return points, features, colors, batch_sign, points_undistort.reshape(-1, 3) / 1000.


