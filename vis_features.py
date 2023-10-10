import numpy as np
import open3d as o3d
import argparse
import os
import trimesh
from typing import List
from unified_optimize import home_made_feature_interpolator
import cv2
from camera import  undistort
from camera.camera_tools import get_extrinsics_from_json, load_color_pc,\
get_intrinsics_distortion_from_npy, pt_vis
from prune.tools import depth2pt_K_numpy, get_dino_features
import warnings
from sklearn.preprocessing import minmax_scale
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import torch
from optimize.hand_model import HandModelMJCF
from sklearn.decomposition import PCA

def plot_obj(mesh, opacity=1):
    return go.Mesh3d(
        x=mesh.vertices[:,0], 
        y=mesh.vertices[:,1], 
        z=mesh.vertices[:,2], 
        i=mesh.faces[:,0], 
        j=mesh.faces[:,1], 
        k=mesh.faces[:,2], 
        opacity=opacity,
        color='royalblue')

def plot_hand(verts, faces):
    return go.Mesh3d(
        x=verts[:,0], 
        y=verts[:,1], 
        z=verts[:,2], 
        i=faces[:,0], 
        j=faces[:,1], 
        k=faces[:,2], 
        color='lightpink')

def plot_points(pts, colors):
    return go.Scatter3d(
        x=pts[:,0],
        y=pts[:,1],
        z=pts[:,2],
        mode='markers',
        marker=dict(
            size=2.5,
            color=colors,      # 根据颜色数组设置每个点的颜色
            opacity=1
        )
    )

def plot_points_pure(pts):
    return go.Scatter3d(
        x=pts[:,0],
        y=pts[:,1],
        z=pts[:,2],
        mode='markers',
        marker=dict(
            size=2.5,
            color='lightgreen',      # 根据颜色数组设置每个点的颜色
            opacity=1
        )
        # surfacecolor=,
    )

def plot_contact_points(pts, grad):

    return go.Cone(x=pts[:,0], y=pts[:,1], z=pts[:,2], u=-grad[:,0], v=-grad[:,1], w=-grad[:,2], anchor='tip',
                            colorscale=[(0,'lightpink'), (1,'lightpink')], sizemode='absolute', sizeref=0.2, opacity=0.5)

def get_pca(ff_ls:list):
    """
    ffs (list): list of np.ndarray (n, 3)
    """
    ffs = np.concatenate(ff_ls, axis=0)
    pca = PCA(n_components=3)
    pca_fit = pca.fit(ffs)
    ff_result = []
    for ff in ff_ls:
        ff_pca = pca_fit.transform(ff)
        ff_pca = (ff_pca - ff_pca.min(axis=0, keepdims=True)) / (ff_pca.max(axis=0, keepdims=True) - ff_pca.min(axis=0, keepdims=True))
        ff_pca = ff_pca * 0.8 + 0.2
        ff_pca = ff_pca[:, [0, 2, 1]]
        ff_result.append(ff_pca)
    return ff_result

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='./data')
    argparser.add_argument('--mode', type=str, default='result_vis')
    argparser.add_argument('--ref_idx', type=lambda x: tuple(map(int, x.split(','))), default=(0, 49, 95))
    argparser.add_argument('--similarity', type=str, default='l2')
    argparser.add_argument('--key', type=int, default=0)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--with_hand', action='store_true')
    args = argparser.parse_args()

    if args.mode == '3Dsim_point':
        ### select your own ref_idx
        key = args.key
        clip = None
        points = np.load(f'./data/points_{key}.npy')
        features = np.load(f'./data/features_{key}.npy')
        points_ = np.load(f'./data/points_{key + 1}.npy')
        features_ = np.load(f'./data/features_{key + 1}.npy')
        if args.similarity == 'dot':
            features /= np.linalg.norm(features, axis=-1, keepdims=True)
            ref_features = features[args.ref_idx[0]]
            feat_dis:np.ndarray = (features * ref_features).sum(axis=-1).squeeze()
            feat_dis_:np.ndarray = (features_ * ref_features).sum(axis=-1).squeeze()
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
            color_proj_ = ((feat_dis_ - feat_dis_.min()) / (feat_dis_.max() - feat_dis_.min()) * 255).astype(np.uint8)
        elif args.similarity == 'l2':
            color_proj = (255 - (feat_dis - feat_dis.min()) / (feat_dis.max() - feat_dis.min()) * 255).astype(np.uint8)
            color_proj_ = (255 - (feat_dis_ - feat_dis_.min()) / (feat_dis_.max() - feat_dis_.min()) * 255).astype(np.uint8)

        color_proj = cv2.applyColorMap(color_proj, cv2.COLORMAP_JET).squeeze()
        color_proj_ = cv2.applyColorMap(color_proj_, cv2.COLORMAP_JET).squeeze()
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
        o3d.io.write_point_cloud('./visualize/3D_similarity0.ply', pt)
        o3d.io.write_point_cloud('./visualize/3D_similarity1.ply', pt_)

    elif args.mode == '3Dsim_volume':
        key = args.key
        path = args.data_path

        ### Load hand and the gt hand pose
        hand = HandModelMJCF("mjcf/shadow_hand_vis.xml", "mjcf/meshes", 
                            n_surface_points=500, device='cuda', tip_aug=2, ref_points=None)
        hand_gt_pose = np.load(os.path.join(path, f'des_ori.npy'))
        rotm = Rotation.from_rotvec(hand_gt_pose[3:6]).as_matrix()
        hand_gt_pose = np.concatenate([hand_gt_pose[0:3], rotm[:, 0], rotm[:, 1], hand_gt_pose[8:]], axis=0)
        hand.set_parameters(torch.from_numpy(hand_gt_pose[None, :]).to('cuda').to(torch.float32), retarget=False, robust=False)
        hand_mesh = hand.get_trimesh_data(0)
        if key == 1:
            hand_mesh = trimesh.load_mesh(os.path.join(path, 'X_mesh.stl'))
        
        ### Load the points, colors and features
        pts0 = np.load(os.path.join(path, f'points_0.npy'))
        pts1 = np.load(os.path.join(path, f'points_1.npy'))
        feat0 = np.load(os.path.join(path, f'features_0.npy'))
        feat1 = np.load(os.path.join(path, f'features_1.npy'))
        colors_ref = np.load(os.path.join(path, f'colors_{key}.npy')).astype(np.float32) / 255

        ### Calculate the distance in the feature field
        query_point = hand.get_surface_points()[0].cpu().numpy().mean(axis=0) # (3, )
        query_weight = (1 / (np.linalg.norm(pts0 - query_point[None, :], axis=-1) + 1e-8) ** 2) # (n)
        query_feat = np.sum(query_weight[:, None] * feat0, axis=0) / query_weight.sum() # (768, )

        ### Calculate the distance in the feature field
        lx, rx = eval(f'pts{key}')[:, 0].min(), eval(f'pts{key}')[:, 0].max()
        ly, ry = eval(f'pts{key}')[:, 1].min(), eval(f'pts{key}')[:, 1].max()
        lz, rz = eval(f'pts{key}')[:, 2].min(), eval(f'pts{key}')[:, 2].max()
        step = 0.01


        X, Y, Z = np.mgrid[lx:rx:step, ly:ry:step, lz:rz:step]
        grid = np.mgrid[lx:rx:step, ly:ry:step, lz:rz:step].reshape(3, -1).T
        grid_weight = (1 / (np.linalg.norm(pts1[None, ...] - grid[:, None, :], axis=-1) + 1e-8) ** 2) # (n, f1)
        grid_feat = np.matmul(grid_weight, feat1) / grid_weight.sum(axis=-1, keepdims=True) # (n, 768)
        feat_dis = np.linalg.norm(grid_feat - query_feat[None, :], axis=-1) # (n, )

        ### Visualize
        layout = go.Layout(
            scene=dict(
                xaxis_visible=False,
                yaxis_visible=False, 
                zaxis_visible=False,  
            )
        )
        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value= 1 / feat_dis,
        opacity=0.1,
        surface_count=12,
        ),layout=layout )
        fig.update_layout(scene=dict(
                        aspectmode='data'
                        ))

        fig.write_html(f'./visualize/volume_{key}.html')
    
    elif args.mode == 'result_vis':
        key = args.key
        path = args.data_path

        ### Load hand and the gt hand pose
        hand = HandModelMJCF("mjcf/shadow_hand_vis.xml", "mjcf/meshes", 
                            n_surface_points=500, device='cuda', tip_aug=2, ref_points=None)
        hand_gt_pose = np.load(os.path.join(path, f'des_ori.npy'))
        rotm = Rotation.from_rotvec(hand_gt_pose[3:6]).as_matrix()
        hand_gt_pose = np.concatenate([hand_gt_pose[0:3], rotm[:, 0], rotm[:, 1], hand_gt_pose[8:]], axis=0)
        hand.set_parameters(torch.from_numpy(hand_gt_pose[None, :]).to('cuda').to(torch.float32), retarget=False, robust=False)
        hand_mesh = hand.get_trimesh_data(0)
        if key == 1:
            hand_mesh = trimesh.load_mesh(os.path.join(path, 'X_mesh.stl'))
        
        ### Load the points, colors and features
        pts0 = np.load(os.path.join(path, f'points_0.npy'))
        pts1 = np.load(os.path.join(path, f'points_1.npy'))
        feat0 = np.load(os.path.join(path, f'features_0.npy'))
        feat1 = np.load(os.path.join(path, f'features_1.npy'))
        colors_ref = np.load(os.path.join(path, f'colors_{key}.npy')).astype(np.float32) / 255
        f_pca = get_pca([feat0, feat1])
        layout = go.Layout(
            scene=dict(
                xaxis_visible=False,
                yaxis_visible=False, 
                zaxis_visible=False,  
            )
        )

        ### Visualize
        if args.pca:
            if args.with_hand:
                fig = go.Figure([plot_obj(hand_mesh), plot_points(eval(f'pts{key}'), f_pca[key])], layout=layout)
            else:
                fig = go.Figure([plot_points(eval(f'pts{key}'), f_pca[key])], layout=layout)
        else:
            if args.with_hand:
                fig = go.Figure([plot_obj(hand_mesh), plot_points(eval(f'pts{key}'), colors_ref)], layout=layout)
            else:
                fig = go.Figure([plot_points(eval(f'pts{key}'), colors_ref)], layout=layout)
        fig.update_layout(scene=dict(
                        aspectmode='data'
                        ))
        fig.write_html(f'./visualize/result_vis_{key}.html')

    else:
        raise NotImplementedError
