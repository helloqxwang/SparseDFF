import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import os
import trimesh
from matplotlib import cm
from typing import List
from unified_optimize import home_made_feature_interpolator
from prune.tools import get_features
import torch
import cv2
import math
from camera import get_main_bbs_from_piles, undistort
from camera.camera_tools import get_extrinsics_from_json, load_color_pc,\
transform_points, vis_color_pc, save_colorpc, vis_img, load_depths,\
get_intrinsics_distortion_from_npy, pt_vis
from camera.capture_3d import capture_auto
from prune.tools import downsample, depth2pt_K_numpy, get_dino_features
import warnings

# Turn off PyTorch warnings
warnings.filterwarnings("ignore")


def trimesh_show(np_pcd_list:List[np.ndarray], mesh_list:list, full_color_list:list=None, color_list:list=None, show=True, size=1):
    colormap = cm.get_cmap('brg', len(np_pcd_list))
    # colormap = cm.get_cmap('gist_ncar_r', len(np_pcd_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(np_pcd_list))
    ]
    if color_list is None:
        color_list = colors

    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        if full_color_list is not None and i < len(full_color_list):
            tpcd.colors =full_color_list[i]
        else:
            tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)

    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    scene.add_geometry(mesh_list)

    if show:
        scene.show()
    # if name:
    #     folder, img_name = name.split(':')
    #     dir_path = './test/visualization/{}'.format(folder)
    #     os.makedirs(dir_path, exist_ok=True)
    #     img_name = img_name + '.png'
    #     path = os.path.join(dir_path, img_name)
    #     img = scene.save_image((480, 480))
    #     with open(path, 'wb') as f:
    #         f.write(img) 
    return scene

def normalize(points:np.ndarray):
    points = points - np.mean(points, axis=0, keepdims=True)
    return points

def get_parrallel_vec(vec:np.ndarray, plane_norm:int)->np.ndarray:
    """get a vector parrallel to the plane

    Args:
        vec (np.ndarray): the vector to the plane (3,)
        plane (int): the plane index (3,)

    Returns:
        np.ndarray: the parrallel vector (unnormalized) (3,)
    """
    plane_norm = plane_norm / np.linalg.norm(plane_norm)
    vec = vec / np.linalg.norm(vec)
    parrallel_vec = vec - np.dot(vec, plane_norm) * plane_norm
    return parrallel_vec 


def get_square_coordinates(norm_vector:np.ndarray, side_length:int, k:int)->np.ndarray:
    """get the mesh grid coordinates of a square on a plane

    Args:
        norm_vector (np.ndarray): the vector os the plane (3, ) (a, b, c) 
        side_length (int): the side length of the square
        k (int): the number of the vertices on each side

    Returns:
        the coordinates in the 3D frame (np.ndarray): (40, 40, 3)
    """
    norm_vector /= np.linalg.norm(norm_vector)

    v1 = get_parrallel_vec(np.array([0, 0, -1]), norm_vector)
    if np.linalg.norm(v1) < 1e-6:
        v1 = get_parrallel_vec(np.array([1, 0, 0]), norm_vector)
        v2 = np.cross(norm_vector, v1)
    else:
        v2 = np.cross(norm_vector, v1)
    

    # Step 3: Normalize v1 and v2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Step 4: Calculate the coordinates of the four vertices
    step = side_length / k
    i, j = np.meshgrid(np.arange(-k//2, k/2), np.arange(-k//2, k/2), indexing='ij')
    vertices = (v1 * ((i * step))[..., np.newaxis] +
                v2 * ((j * step))[..., np.newaxis])

    return vertices

def vis_3D(points:np.ndarray, features:np.ndarray, ref_idx:int, plane:np.ndarray, res:int=200):
    points_norm = normalize(points)
    grids = get_square_coordinates(plane, 0.6, res)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    interpolator = home_made_feature_interpolator(points_norm, features, device=device)
    grids = torch.from_numpy(grids).float().to(device)
    interpolated_features:np.ndarray = interpolator.predict(grids).cpu().numpy().reshape(-1, features.shape[-1])
    ref_features = features[ref_idx]
    feat_dis:np.ndarray = np.linalg.norm(interpolated_features - ref_features, axis=1)
    print(feat_dis.max())
    # feat_dis = feat_dis.clip(0, 40)
    feat_dis = feat_dis.reshape(res, res)
    feat_dis = (255 - (feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)
    print(feat_dis.max())
    print(feat_dis.min())
    print(feat_dis.mean())
    cv2.imwrite('feat_dis_ori.jpg', feat_dis)
    feat_dis = cv2.applyColorMap(feat_dis, cv2.COLORMAP_JET)
    cv2.imwrite('feat_dis.jpg', feat_dis)

def vis_2D(features:np.ndarray, ref_idx:(int, int), mark:np.ndarray, clip:float=None):
    """visulize the 2D feature distance in a image

    Args:
        features (np.ndarray): (H, W, dim)
        ref_idx (int, int): the index of the reference point
        mark (np.ndarray): _description_
        clip (float, optional): clip too far distance. Defaults to None.
    """
    ref_features = features[ref_idx]
    feat_dis:np.ndarray = np.linalg.norm(features - ref_features, axis=-1)
    print('feat_dis max:', feat_dis.max())
    if clip:
        feat_dis = feat_dis.clip(0, clip)
    color_proj = (255 - (feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)

    # print(color_proj.max())
    # print(color_proj.min())
    # print(color_proj.mean())
    color_proj = cv2.applyColorMap(color_proj, cv2.COLORMAP_JET)
    cv2.circle(color_proj, (ref_idx[1], ref_idx[0]), 3, (0,0,0), -1)
    cv2.imwrite('vis_dis.jpg', color_proj)

def vis_2D_multiview(features:np.ndarray, ref_idx:(int, int, int), mark:np.ndarray, clip:float=None):
    """visulize the 2D feature distance amoung multiple images

    Args:
        features (np.ndarray): (n, H, W, dim)
        ref_idx (int, int, int): the index of the reference point
        mark (np.ndarray): (n, H, W)
        clip (float, optional): clip too far distance. Defaults to None.
    """
    ref_features = features[ref_idx]
    feat_dis:np.ndarray = np.linalg.norm(features - ref_features, axis=-1)
    print('feat_dis max:', feat_dis.max())
    if clip:
        feat_dis = feat_dis.clip(0, clip)
    color_proj = (255 - (feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)
    # color_proj = (255 - (feat_dis - feat_dis.min(axis=(1, 2),keepdims=True)) / (feat_dis.max(axis=(1, 2),keepdims=True) - feat_dis.min(axis=(1, 2),keepdims=True)) * 255).astype(np.uint8)
    # print(color_proj.max())
    # print(color_proj.min())
    # print(color_proj.mean())
    img_num = features.shape[0]
    img_ls = []
    for idx, img in enumerate(color_proj):
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        if ref_idx[0] == idx:
            cv2.circle(img, (ref_idx[2], ref_idx[1]), 1, (0,0,0), -1)
        img_ls.append(img)

    row = int(np.sqrt(img_num))
    import math
    col = math.ceil(float(img_num) / row)
    img = np.zeros((row * img_ls[0].shape[0], col * img_ls[0].shape[1], 3), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            if idx < img_num:
                img[i * img_ls[0].shape[0]:(i+1) * img_ls[0].shape[0], j * img_ls[0].shape[1]:(j+1) * img_ls[0].shape[1]] = img_ls[idx]
    cv2.imwrite('vis_dis_multi.jpg', img)

def load_color_depth_pile(path:str, obj_ls:List[str]):
    color_ls = []
    depth_ls = []
    for obj_path in obj_ls:
        path_ = os.path.join(path, obj_path)
        color = np.load(os.path.join(path_, 'colors.npy'))
        depth = np.load(os.path.join(path_, 'depth.npy'))
        color_ls.append(color)
        depth_ls.append(depth)
    colors = np.stack(color_ls, axis=0)
    depths = np.stack(depth_ls, axis=0)
    return colors, depths

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
def get_foregroundmark(features:np.ndarray, threshold=0.65):
    """get the **foreground** mask using PCA and dino_feature

    Args:
        features (np.ndarray): (cam_num, H, W, dim)
        threshold (float, optional): _description_. Defaults to 0.6.
    """
    pca = PCA(n_components=1)
    reduced_feat = pca.fit_transform(features.reshape(-1, features.shape[-1]))
    norm_feat = minmax_scale(reduced_feat)
    norm_feat = norm_feat.reshape(features.shape[:-1])
    return norm_feat > threshold

def get_2D_pca(features:np.ndarray, for_pc=False, clip:float=None):
    masks = get_foregroundmark(features)
    pca = PCA(n_components=3)
    reduced_feat = pca.fit_transform(features[masks].reshape(-1, features.shape[-1]))
    norm_feat = minmax_scale(reduced_feat)
    pictures = np.zeros(features.shape[:-1] + (3, ))
    if for_pc:
        return norm_feat, masks
    else:
        pictures[masks] = norm_feat
        # return as type uint8 for cv2 [0, 255]
        pictures = (pictures * 255).astype(np.uint8)
        return pictures

def visnpic(pictures:np.ndarray):
    """visualize 2D pictures using cv2

    Args:
        pictures (np.ndarray): (cam_num, H, W, dim)
    """
    img_num = pictures.shape[0]
    row = int(np.sqrt(img_num))
    col = math.ceil(float(img_num) / row)
    img = np.zeros((row * pictures[0].shape[0], col * pictures[0].shape[1], 3), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            if idx < img_num:
                img[i * pictures[0].shape[0]:(i+1) * pictures[0].shape[0], j * pictures[0].shape[1]:(j+1) * pictures[0].shape[1]] = pictures[idx]
    print(img.shape)
    cv2.imwrite('vis_dis_multi.jpg', img)

def normalize_(tensor:np.ndarray):
    tensor = (tensor - tensor.mean(axis=(1, 2), keepdims=True)) / tensor.std(axis=(1, 2), keepdims=True)
    return tensor

def get_color_depth_pt_und(data_path):
    extrinsics = get_extrinsics_from_json('/home/user/wangqx/stanford/kinect/workspace/calibration.json')
    colors_distort, depths_distort = load_color_depth_pile(data_path, obj_ls=args.object_ls)
    intrinsics, distortion = get_intrinsics_distortion_from_npy('/home/user/wangqx/stanford/kinect/workspace/', standard=False)
    colors, depths = undistort(colors_distort, depths_distort ,intrinsics, distortion) 
    points_undistort = depth2pt_K_numpy(depths, intrinsics, np.linalg.inv(extrinsics), xyz_images=True)
    return colors, depths, points_undistort
    
def get_pca_feat_points(data_path, scale = 14, key=0):
    colors0, depths0, points0 = get_color_depth_pt_und(data_path[0])
    colors1, depths1, points1 = get_color_depth_pt_und(data_path[1])
    colors = np.concatenate([colors0, colors1], axis=0)
    depths = np.concatenate([depths0, depths1], axis=0)
    points_undistort = np.concatenate([points0, points1], axis=0)
    bbs = get_main_bbs_from_piles(colors)
    colors = np.array([colors[i, bb[1]:bb[3], bb[0]:bb[2]] for i, bb in enumerate(bbs)])
    depths = np.array([depths[i, bb[1]:bb[3], bb[0]:bb[2]] for i, bb in enumerate(bbs)])
    points_undistort = np.array([points_undistort[i, bb[1]:bb[3], bb[0]:bb[2]] for i, bb in enumerate(bbs)])
    
    features = get_features(colors, scale, img_num = colors.shape[0], key=key, name=os.path.basename(args.data_path)+args.object0).squeeze().cpu().numpy()
    batch_size, h, w, _ = features.shape
    points_dd = np.array([cv2.resize(points_undistort[idx], (w, h), interpolation=cv2.INTER_NEAREST) for idx in range(points_undistort.shape[0])])
    depths_dd = np.array([cv2.resize(depths[idx], (w, h), interpolation=cv2.INTER_NEAREST) for idx in range(depths.shape[0])])
    batch_sign = np.zeros((batch_size, h, w))
    for i in range(batch_size):
        batch_sign[i] = i + 1
    features, masks = get_2D_pca(features, for_pc=True)
    ob1_num = np.count_nonzero(masks[:4])
    batch_sign = batch_sign[masks]
    points_dd = points_dd[masks]
    return features, points_dd, batch_sign, ob1_num

def process_f(pp, features0, mask=None):
    features0 = pp.transform(features0)
    features0 = (features0 - features0.min(axis=0, keepdims=True)) / (features0.max(axis=0, keepdims=True) - features0.min(axis=0, keepdims=True))
    features0 = features0 * 0.8 + 0.2
    features0 = features0[:, [0, 2, 1]]
    if mask is not None:
        features0[~mask] = np.zeros(3)
    return features0

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='/home/user/wangqx/stanford/kinect/data/20230827_145333')
    argparser.add_argument('--object0','-o0', type=str, default='000262413912')
    argparser.add_argument('--object1','-o1', type=str, default='000272313912')
    argparser.add_argument('--object_ls', type=list, default=['000299113912', '000262413912', '000285613912', '000272313912'])
    argparser.add_argument('--ref_idx', type=lambda x: tuple(map(int, x.split(','))), default=(0, 49, 95))
    argparser.add_argument('--plane_norm', type=lambda x: list(map(int, x.split(','))), default=[0, 1, 0])
    argparser.add_argument('--mode', type=str, default='2D_similarity_multi')
    argparser.add_argument('--similarity', type=str, default='dot')
    args = argparser.parse_args()


    # features = np.load('/home/user/wangqx/stanford/Learning_based_method/features_und_xyz.npy')
    # print(features.shape)
    # vis_2D_multiview(features, args.ref_idx, None, clip=None)
    # exit()

    if args.mode == '2D_similarity':
        path = os.path.join(args.data_path, args.object1)
        colors = np.load(os.path.join(path, 'colors.npy'))
        depths = np.load(os.path.join(path, 'depth.npy'))
        scale = 4
        features = get_features(colors[None, :], scale, img_num = 1).squeeze().cpu().numpy()
        vis_2D(features, args.ref_idx, None, clip=None)
    elif args.mode == '2D_similarity_multi':
        # (0, 49, 95)
        # (0, 55, 120)
        # (0, 58, 128)
        # (0, 92, 106)
        colors, depths = load_color_depth_pile(args.data_path, obj_ls=args.object_ls)
        scale = 4
        colors = colors[:, :-200, 350:-350, :]
        # colors = colors.astype(np.float32) / 255.
        features = get_features(colors, scale, img_num = colors.shape[0], key=24, name=os.path.basename(args.data_path)+args.object0).squeeze().cpu().numpy()
        print(features.shape)

        features0 = np.load('/home/user/wangqx/stanford/Learning_based_method/data4/features_0_0.npy')
        features1 = np.load('/home/user/wangqx/stanford/Learning_based_method/data4/features_0_1.npy')
        features2 = np.load('/home/user/wangqx/stanford/Learning_based_method/data4/features_0_2.npy')
        features3 = np.load('/home/user/wangqx/stanford/Learning_based_method/data4/features_0_3.npy')
        # (4, 768)
        means = np.stack([features0.mean(axis=0), features1.mean(axis=0), features2.mean(axis=0), features3.mean(axis=0)], axis=0)
        std = np.stack([features0.std(axis=0), features1.std(axis=0), features2.std(axis=0), features3.std(axis=0)], axis=0)

        print('means', means.shape)
        # features = (features - means[:, None, None, :]) / std[:, None, None, :]

        vis_2D_multiview(features, args.ref_idx, None, clip=None)
    elif args.mode == '3D_similarity_picture':
        args.data_path = '/home/user/wangqx/stanford/data'
        args.object1 = 'bear_points0.npy'
        path = os.path.join(args.data_path, args.object1)
        points = np.load(path)
        points = normalize(points)
        features = np.load(path.replace('points', 'features'))
        print(points.shape)
        print(features.shape)
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud('test0.ply', pt)
        vis_3D(points, features, args.ref_idx[0], np.array(args.plane_norm).astype(np.float32))
    elif args.mode == '3D_similarity':
        key = 0
        clip = None
        
        points = np.load(f'./data_b0/monkey_points_{key}.npy')
        features = np.load(f'./data_b0/monkey_features_{key}.npy')
        points_ = np.load(f'./data_b0/monkey_points_{key + 1}.npy')
        features_ = np.load(f'./data_b0/monkey_features_{key + 1}.npy')
        if args.similarity == 'dot':
            features /= np.linalg.norm(features, axis=-1, keepdims=True)
            ref_features = features[args.ref_idx[0]]
            feat_dis = (features * ref_features).sum(axis=-1).squeeze()
        elif args.similarity == 'l2':
            ref_features = features[args.ref_idx[0]]
            feat_dis:np.ndarray = np.linalg.norm(features - ref_features, axis=-1).squeeze()
            feat_dis_:np.ndarray = np.linalg.norm(features_ - ref_features, axis=-1).squeeze()
        
        print('feat_dis min:', feat_dis.min())
        print('feat_dis max:', feat_dis.max())
        if clip is not None:
            feat_dis = feat_dis.clip(*clip)
            feat_dis_ = feat_dis_.clip(*clip)
        if args.similarity == 'dot':
            color_proj = ((feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)
        elif args.similarity == 'l2':
            color_proj = (255 - (feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)
            color_proj_ = (255 - (feat_dis_ - feat_dis_.min()) / (feat_dis_.max() - feat_dis_.min()) * 255).astype(np.uint8)
        
        # print(color_proj.max())
        # print(color_proj.min())
        # print(color_proj.mean())
        # print(color_proj.shape)
        np.save('color_proj_.npy', color_proj_)
        color_proj = cv2.applyColorMap(color_proj, cv2.COLORMAP_JET).squeeze()
        color_proj_ = cv2.applyColorMap(color_proj_, cv2.COLORMAP_JET).squeeze()
        print(color_proj.shape)
        color_proj = color_proj.astype(np.float32) / 255.
        color_proj_ = color_proj_.astype(np.float32) / 255.
        color_proj = color_proj[:, [2, 1, 0]]
        color_proj_ = color_proj_[:, [2, 1, 0]]
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(points)
        pt.colors = o3d.utility.Vector3dVector(color_proj)
        pt_ = o3d.geometry.PointCloud()
        pt_.points = o3d.utility.Vector3dVector(points_)
        pt_.colors = o3d.utility.Vector3dVector(color_proj_)
        # trimesh_show([points], [], [color_proj])
        o3d.io.write_point_cloud('test0.ply', pt)
        o3d.io.write_point_cloud('test1.ply', pt_)

    elif args.mode == '2D_pca_multiview':
        # color0 = cv2.imread('./cat0.jpg')
        # color1 = cv2.imread('./cat1.jpg')
        # color2 = cv2.imread('./cat2.jpg')
        # color3 = cv2.imread('./cat3.jpg')
        # colors = np.stack([color0, color1, color2, color3], axis=0)
        # colors0, depths0 = load_color_depth_pile(args.data_path, obj_ls=args.object_ls)
        # colors1, depths1 = load_color_depth_pile('/home/user/wangqx/stanford/kinect/data/20230827_145354', obj_ls=args.object_ls)
        # # colors0, depths0 = load_color_depth_pile('/home/user/wangqx/stanford/kinect/data/20230827_150027', obj_ls=args.object_ls)
        # # colors1, depths1 = load_color_depth_pile('/home/user/wangqx/stanford/kinect/data/20230827_150211', obj_ls=args.object_ls)
        # colors = np.concatenate([colors0, colors1], axis=0)
        # bbs = get_main_bbs_from_piles(colors)
        # colors = np.stack([colors[idx, bb[1]:bb[3], bb[0]:bb[2], :] for idx, bb in enumerate(bbs)], axis=0)
        # # colors = colors[:, :-200, 350:-350, :]
        # scale = 14
        # print(colors.shape)
        # colors = colors.astype(np.float32) / 255.

        # features = get_features(colors, scale, img_num = colors.shape[0], key=24, name=os.path.basename(args.data_path)+args.object0).squeeze().cpu().numpy()
        features0 = np.load('/home/user/wangqx/stanford/data_monkey/dino_features0.npy')
        features1 = np.load('/home/user/wangqx/stanford/data_monkey/dino_features1.npy')
        features2 = np.load('/home/user/wangqx/stanford/data_monkey/dino_features2.npy')
        features3 = np.load('/home/user/wangqx/stanford/data_monkey/dino_features3.npy')
        shape0 = features0.shape
        shape1 = features1.shape
        shape2 = features2.shape
        shape3 = features3.shape

        # pic0 = cv2.imread('/home/user/wangqx/stanford/kinect/data/monkey0/000262413912/colors.png')
        # pic1 = cv2.imread('/home/user/wangqx/stanford/kinect/data/monkey0/000272313912/colors.png')
        # pic2 = cv2.imread('/home/user/wangqx/stanford/kinect/data/monkey0/000285613912/colors.png')
        # pic3 = cv2.imread('/home/user/wangqx/stanford/kinect/data/monkey0/000299113912/colors.png')
        # print(pic0.shape)
        # features0 = get_dino_features(pic0, 14).squeeze().cpu().numpy()
        # features1 = get_dino_features(pic1, 14).squeeze().cpu().numpy()
        # features2 = get_dino_features(pic2, 14).squeeze().cpu().numpy()
        # features3 = get_dino_features(pic3, 14).squeeze().cpu().numpy()
        features0 = features0.reshape(-1, 768)
        features1 = features1.reshape(-1, 768)
        features2 = features2.reshape(-1, 768)
        features3 = features3.reshape(-1, 768)
        features = np.concatenate([features0.reshape(-1, 768), features1.reshape(-1, 768), features2.reshape(-1, 768), features3.reshape(-1, 768)], axis=0)
        pca1 = PCA(n_components=1)
        pca3 = PCA(n_components=3)

        mask = get_foregroundmark(features.reshape(-1, 768))
        print(mask.shape)

        pp = pca3.fit(features[mask])
        p0 = process_f(pp, features0, mask[0:features0.shape[0]])
        p1 = process_f(pp, features1, mask[features0.shape[0]:features0.shape[0]+features1.shape[0]])
        p2 = process_f(pp, features2, mask[features0.shape[0]+features1.shape[0]:features0.shape[0]+features1.shape[0]+features2.shape[0]])
        p3 = process_f(pp, features3, mask[-features3.shape[0]:])
        cv2.imwrite('p0.jpg', p0.reshape(shape0[0], shape0[1], -1) * 255)
        cv2.imwrite('p1.jpg', p1.reshape(shape1[0], shape1[1], -1) * 255)
        cv2.imwrite('p2.jpg', p2.reshape(shape2[0], shape2[1], -1) * 255)
        cv2.imwrite('p3.jpg', p3.reshape(shape3[0], shape3[1], -1) * 255)



        # print(features.shape)
        # from pdb import set_trace; set_trace()
        # pictures = get_2D_pca(features)
        # visnpic(pictures)
    elif args.mode == '3D_pca':
        colors, points, batch_signs, num1= get_pca_feat_points(['/home/user/wangqx/stanford/kinect/data/20230827_150211',
                                                           '/home/user/wangqx/stanford/kinect/data/20230827_145854']
                                                          , scale=3, key=28)
        
        pt_ = o3d.geometry.PointCloud()
        pt_.points = o3d.utility.Vector3dVector(points[:num1])
        pt_.colors = o3d.utility.Vector3dVector(colors[:num1])
        o3d.io.write_point_cloud('test0.ply', pt_)

        pt_ = o3d.geometry.PointCloud()
        pt_.points = o3d.utility.Vector3dVector(points[num1:])
        pt_.colors = o3d.utility.Vector3dVector(colors[num1:])
        o3d.io.write_point_cloud('test1.ply', pt_)

    else:
        raise NotImplementedError

    # dis = np.array([0.8, 0, 0])
    # points1 = points1 + dis
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(np.concatenate((points0, points1), axis=0))
    # pcd1.colors = o3d.utility.Vector3dVector(np.concatenate((features0, features1), axis=0)/ 255.)
    # o3d.visualization.draw_geometries([pcd1])
    # o3d.io.write_point_cloud('test0.ply', pcd1)