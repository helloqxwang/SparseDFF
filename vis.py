
import plotly.graph_objects as go
import numpy as np
import trimesh
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
import torch
from optimize.hand_model import HandModelMJCF
from sklearn.decomposition import PCA


path = '/home/qianxu/wangqx/data/stanford/data_previous/data_927_MonkeyToBear/data_b0'

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