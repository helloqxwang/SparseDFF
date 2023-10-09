import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
from pytorch3d.io import load_objs_as_meshes
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
import skimage

def pt_vis(points:np.ndarray, size=None):
    """vicsualize the point cloud"""
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points=o3d.utility.Vector3dVector(points)
    if size:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pointcloud, axis])
    else:
        o3d.visualization.draw_geometries([pointcloud])

    return

def mesh2rgbd(mesh_path:list , device=torch.device('cuda'), scale=1.4, num=20, img_size=256):
    """smampling the mesh to rgbd images

    Args:
        mesh_path (list): the path of a mesh file
        device (_type_): the device to run the model
        num (int, optional): the number of views. Defaults to 20.
        img_size (int, optional): the size of the image. Defaults to 512.

    Returns:
        images, depths: torch.tensor of shape (N, H, W, 3) and (N, H, W)
        c21, K: np.ndarray of shape (N, 4, 4) and (N, 3, 3)
        camera_params: dict of camera parameters

    """
    ### add a texture to the mesh
    mesh = load_objs_as_meshes(mesh_path, device=device)
    verts = mesh.verts_list()[0]
    rot = Rotation.from_euler('zyx', [np.pi, 0, - np.pi/2], degrees=False)
    matrix = torch.tensor(rot.as_matrix(), dtype=verts.dtype).to(device)
    verts = (matrix @ verts.T).T

    mesh.verts_list()[0] = verts * 1.4
    tex = torch.ones_like(verts)[None, ...].to(device)  # (1, V, 3)
    mesh.textures = TexturesVertex(verts_features=tex)

    meshes = mesh.extend(num)
    elev = 30
    azim = torch.linspace(-180, 180, num + 1)[:-1]
    ### R is the pytorch3d2world rotation matrix, T is the world2pytorch3d translation
    R, T = look_at_view_transform(dist=0.91, elev=elev, azim=azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=img_size, 
        bin_size=0,
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    ) 

    # We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
    # so the renderer does not need to be reinitialized if any of the settings change.
    images = renderer(meshes, cameras=cameras, lights=lights)
    fragments = renderer.rasterizer(meshes, cameras=cameras)
    depths = fragments.zbuf.mean(dim=-1)

    fx = fy = img_size / (2 * np.tan(np.pi / 3 / 2))
    cx = cy = img_size / 2
    ### construct the Inrinsics matrix
    K = np.asarray([
     [fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]])
    camera_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'xres': img_size, 'yres': img_size}
    ### construct the homogeneous coordinates
    # camera coordinates to pytorch3d camera coordinates
    c2c_pytorch3d = np.eye(4)
    c2c_pytorch3d[0, 0] = -1
    c2c_pytorch3d[1, 1] = -1
    # world coordinates to pytorch3d camera coordinates
    w2c_pytorch3d = np.eye(4)[None, ...].repeat(num, axis=0)
    w2c_pytorch3d[:, :3, :3] = R.mT
    w2c_pytorch3d[:, :3, 3] = T
    # camera coordinates to world coordinates
    c2w = np.linalg.inv(w2c_pytorch3d) @ c2c_pytorch3d

    return images[..., :3], depths, c2w, K, camera_params

def load_dataset_img(path, load:list, device='cuda'):
    """load imgs from npy files and return them as torch tensors

    Args:
        path (_type_): _description_
        num (int, optional): _description_. Defaults to 1.
        device (str, optional): _description_. Defaults to 'cuda'.
    Return:
        [num, height, width, channels]
    """
    npy_files = glob.glob(os.path.join(path, '*.npy'))
    img_ls = []
    for i in range(min(len(load), len(npy_files))):
        # Load the i-th image
        img = np.load(npy_files[load[i]])
        img = torch.from_numpy(img).to(device)
        img_ls.append(img)
    imgs = torch.stack(img_ls, dim=0)
    return imgs

def save_imgs(path_, imgs):
    """save the images to the path_/images/
    """
    path = os.path.join(path_, './images')
    os.makedirs(path, exist_ok=True)
    if imgs.dim() == 3:
        for i in range(imgs.shape[0]):
            cv2.imwrite(os.path.join(path, f'{i}.png'), imgs[i].cpu().numpy())
    else:
        cv2.imwrite(os.path.join(path, f'0.png'), imgs.cpu().numpy())

def view_smiliarity(img1, img2, device='cuda', mark1=None, mark2=None, ref1=None, ref2=None):
    """link the marthed points and view the two images"""
    assert img1.dim() == 3 and img2.dim() == 3
    h, w, c = img1.shape
    m1, m2 = find_match(img1, img2)
    img1 = torch.zeros((h*w)).to(device)
    img2 = torch.zeros((h*w)).to(device)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if ref1 is not None:
        ref1 = np.asarray(cv2.resize(ref1.cpu().numpy(), (h , w), interpolation=cv2.INTER_NEAREST) )
        print(ref2.shape)
        ref2 = np.asarray(cv2.resize(ref2.cpu().numpy(), (h , w), interpolation=cv2.INTER_NEAREST) )
        print(ref1.shape)
        img = np.concatenate([ref1, ref2], axis=1)
        ax.imshow(img)
    else:
        img = np.concatenate([img1.reshape(h, w).cpu().numpy(), img2.reshape(h, w).cpu().numpy()], axis=1)
        ax.imshow(img, cmap='gray')

    for i, j in zip(m1, m2):
        y1, x1 = i // w, i % w
        y2, x2 = j // w, j % w
        x2 += w
        ax.plot(x1, y1, 'ro', markersize=1)
        ax.plot(x2, y2, 'ro', markersize=1)
        if mark1[y1, x1] != -1 or mark2[y2, x2 - w] != -1:
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=0.5)
    fig.savefig('matching_points.png')

def load_cam(path, load:list, device='cuda'):
    """load the camera parameters

    Args:
        path (str): the path of the camera parameters
        load (list): the indices of the camera parameters to load
        device (str, optional): Defaults to 'cuda'.

    Returns:
        list: list of the camera parameters dicts
    """
    npy_files = glob.glob(os.path.join(path, '*.npy'))
    cam_ls = []
    for i in range(min(len(load), len(npy_files))):
        camera_dict = np.load(os.path.join(path, 'camera_info{}.npy'.format(load[i])), allow_pickle=True).item()
        cam_ls.append(camera_dict)
    return cam_ls

def match_cache(func):
    """cache the matches between the images
    """
    cache = dict()
    def wrapper(img1:torch.Tensor, img2:torch.Tensor, mask1:torch.Tensor, mask2:torch.Tensor, device='cuda'):
        i = hash(img1.data_ptr())
        j = hash(img2.data_ptr())
        if (i, j) in cache:
            print('using cache!!!')
            return cache[(i, j)]
        elif (j, i) in cache:
            print('using cache!!!')
            m2, m1 = cache[(j, i)]
            return m1, m2
        else:
            cache[(i, j)] = func(img1, img2, mask1, mask2, device)
        return cache[(i, j)]
    return wrapper

### it's very different to put this into batch version
@match_cache
def find_match(img1, img2, device='cuda'):
    """find the matches between **TWO** images using the feature vectors

    Args:
        img1 (torch.tensor): _description_
        img2 (torhc.tensor): _description_
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        match1, match2: the indices of the matches
    """
    # Calculate the pairwise distances between the feature vectors

    dists = torch.cdist(img1.reshape(-1, img1.shape[-1]), img2.reshape(-1, img2.shape[-1])).to(device)
    idx1 = torch.argmin(dists, dim=1)
    idx2 = torch.argmin(dists, dim=0)

    # Find the matches between the two images
    matches1 = []
    matches2 = []
    for i in range(idx1.shape[0]):
        if idx2[idx1[i]] == i:
            matches1.append(i)
            matches2.append(idx1[i].item())

    return torch.tensor(matches1).to(device), torch.tensor(matches2).to(device)

@match_cache
def find_match_masked(img1:torch.Tensor, img2:torch.Tensor, mask1:torch.Tensor, mask2:torch.Tensor, device='cuda', use_cpu=False):
    """find the matches between two images using the feature vectors
    
    masked the useless part of the images using the mask(always the depth) (FALSE means useless)

    Args:
        img1 (torch.tensor): (height, width, channels)
        img2 (torch.tensor): (height, width, channels)
        mask1 (torch.tensor): (height, width)
        mask2 (torch.tensor): (height, width)
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        match1, match2: the indices of the matches
    """
    h, w, c = img1.shape
    if use_cpu:
        dists = torch.cdist(img1.reshape(-1, img1.shape[-1]).to('cpu'), img2.reshape(-1, img2.shape[-1]).to('cpu'))
    else:
        dists = torch.cdist(img1.reshape(-1, img1.shape[-1]).to(torch.float16), img2.reshape(-1, img2.shape[-1]).to(torch.float16))
    idx1 = torch.argmin(dists, dim=1)
    idx2 = torch.argmin(dists, dim=0)

    # Find the matches between the two images
    match1 = []
    match2 = []
    for i in range(idx1.shape[0]):
        if idx2[idx1[i]] == i:
            y1, x1 = i // w, i % w
            y2, x2 = idx1[i] // w, idx1[i] % w
            if mask1[y1, x1]  and mask2[y2, x2] :
                # print(mask1[y1, x1], mask2[y2, x2])
                match1.append(i)
                match2.append(idx1[i].item())
    
    return torch.tensor(match1).to(device), torch.tensor(match2).to(device)
    
def depth2pt(depths:torch.Tensor, camera_params:dict, R:torch.Tensor, xyz_images=True, device='cuda')->torch.Tensor:
    """
    If xyz_images is True: 
        Translate a pile of depth images ( N x H x W) to xyz images ( N x H x W x 3)
    Else: 
        Translate a pile of depth images ( N x H x W) to 3D points ( Num x 3) filtered by the depth < 0
        bath_sign (Num, ) is the Sign of the batch, noting every selected points which batch it belongs to
        zero_filter (Num, ) is the filter of the points which depth > 0

    Args:
        depths (_type_): (n, h, w)
        camera_params (_type_): the camera parameters
        R (_type_): the rotation matrix (n, 4, 4)

    Returns:
        If xyz_images is True: 
            torch.tensor (n, h, w, 3)
        Else: 
            torch.tensor (num, 3), batch_sign (num, ), zero_filter (n*h*w, )
    """
    # scale camera parameters
    batch_size, h, w = depths.shape
    scale_x = w / camera_params['xres']
    scale_y = h / camera_params['yres']
    fx = camera_params['fx'] * scale_x
    fy = camera_params['fy'] * scale_y
    x_offset = camera_params['cx'] * scale_x
    y_offset = camera_params['cy'] * scale_y
    ### 对于index(a0, a1, ..., an) shape：(n, a0, a1, ..., an)
    ### transpose 改变了坐标的打包方式，从"所有行坐标在一起，所有列坐标在一起"变为"每个点的行列坐标在一起"。
    ### 事实上，indice 坐标是以通道(channel)的方式分开的，即第一个通道存储所有的行坐标，第二个通道存储所有的列坐标。
    ### 第二维和第三维度就是行列坐标
    ### 但我们希望将点的所有横坐标放在一起处理，所有纵坐标放在一起处理，所以要将通道的维度放到最后。
    # (3, batch_size, rey, resx) -> (batch_size, rey, resx, 3)
    indices = torch.stack(torch.meshgrid(torch.arange(batch_size), torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).to(dtype=torch.float32).to(device)
    # indices = np.indices((batch_size ,h, w), dtype=np.float32).transpose(1, 2, 3, 0)
    depths[depths < 0] = 0
    z_e = depths
    x_e = (indices[..., -1] - x_offset) * z_e / fx
    y_e = (indices[..., -2] - y_offset) * z_e / fy
    homogenerous = torch.ones((batch_size, h, w)).to(device)
    xyz_img = torch.stack([x_e, y_e, z_e, homogenerous], axis=-1)  # Shape: [n, H, W, 4]
    ### (n, 4, 4) * (n, h, w, 4) --> (n, h, w, 4)
    # xyz_img_trans = np.stack([np.matmul(R[i], xyz_img[i].reshape(-1, 4).T).T.reshape(h, w, 4) for i in range(R.shape[0])], axis=0)
    xyz_img_trans = torch.matmul(R, xyz_img.reshape(xyz_img.shape[0], -1, 4).transpose(2, 1)).transpose(2, 1).reshape(batch_size, h, w, 4)
    if xyz_images:
        return xyz_img_trans[..., :3]
    else:
        batch_sign = torch.zeros((batch_size, h, w)).to(device)
        for i in range(batch_size):
            batch_sign[i] = i + 1
        zero_filter = (depths != 0).reshape(-1)
        batch_sign = batch_sign.reshape(-1)[zero_filter]
        return xyz_img_trans[..., :3].reshape(-1, 3)[zero_filter], batch_sign, zero_filter

def depth2pt_K(depths:torch.Tensor, K:torch.Tensor , R:torch.Tensor, xyz_images=True, device='cuda')->torch.Tensor:
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
        depths (torch.Tensor): (n, h, w)
        K (torch.Tensor): the intrinsics (n, 3, 3)
        R (_type_): the rotation matrix (n, 4, 4)

    Returns:
        If xyz_images is True: 
            torch.tensor (n, h, w, 3)
        Else: 
            torch.tensor (num, 3), batch_sign (num, ), zero_filter (n*h*w, )
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
    indices = torch.stack(torch.meshgrid(torch.arange(batch_size), torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).to(dtype=torch.float32).to(device)
    # indices = np.indices((batch_size ,h, w), dtype=np.float32).transpose(1, 2, 3, 0)
    # depths[depths < 0] = 0
    z_e = depths
    x_e = (indices[..., -1] - x_offset) * z_e / fx
    y_e = (indices[..., -2] - y_offset) * z_e / fy
    homogenerous = torch.ones((batch_size, h, w)).to(device)
    xyz_img = torch.stack([x_e, y_e, z_e, homogenerous], axis=-1)  # Shape: [n, H, W, 4]
    ### (n, 4, 4) * (n, h, w, 4) --> (n, h, w, 4)
    # xyz_img_trans = np.stack([np.matmul(R[i], xyz_img[i].reshape(-1, 4).T).T.reshape(h, w, 4) for i in range(R.shape[0])], axis=0)
    xyz_img_trans = torch.matmul(R, xyz_img.reshape(xyz_img.shape[0], -1, 4).transpose(2, 1)).transpose(2, 1).reshape(batch_size, h, w, 4)
    if xyz_images:
        return xyz_img_trans[..., :3]
    else:
        batch_sign = torch.zeros((batch_size, h, w)).to(device)
        for i in range(batch_size):
            batch_sign[i] = i + 1
        zero_filter = (depths != 0).reshape(-1)
        batch_sign = batch_sign.reshape(-1)[zero_filter]
        return xyz_img_trans[..., :3].reshape(-1, 3)[zero_filter], batch_sign, zero_filter

def depth2pt_numpy(depths:np.ndarray, camera_params:dict, R:np.ndarray, xyz_images=True)->np.ndarray:
    """
    If xyz_images is True: 
        Translate a pile of depth images ( N x H x W) to xyz images ( N x H x W x 3)
    Else: 
        Translate a pile of depth images ( N x H x W) to 3D points ( Num x 3) filtered by the depth < 0
        bath_sign (Num, ) is the Sign of the batch, noting every selected points which batch it belongs to
        zero_filter (Num, ) is the filter of the points which depth > 0

    Args:
        depths (_type_): (n, h, w)
        camera_params (_type_): the camera parameters
        R (_type_): the rotation matrix (n, 4, 4)

    Returns:
        If xyz_images is True: 
            torch.tensor (n, h, w, 3)
        Else: 
            torch.tensor (num, 3), batch_sign (num, ), zero_filter (n*h*w, )
    """
    # scale camera parameters
    batch_size, h, w = depths.shape
    scale_x = w / camera_params['xres']
    scale_y = h / camera_params['yres']
    fx = camera_params['fx'] * scale_x
    fy = camera_params['fy'] * scale_y
    x_offset = camera_params['cx'] * scale_x
    y_offset = camera_params['cy'] * scale_y
    ### 对于index(a0, a1, ..., an) shape：(n, a0, a1, ..., an)
    ### transpose 改变了坐标的打包方式，从"所有行坐标在一起，所有列坐标在一起"变为"每个点的行列坐标在一起"。
    ### 事实上，indice 坐标是以通道(channel)的方式分开的，即第一个通道存储所有的行坐标，第二个通道存储所有的列坐标。
    ### 第二维和第三维度就是行列坐标
    ### 但我们希望将点的所有横坐标放在一起处理，所有纵坐标放在一起处理，所以要将通道的维度放到最后。
    # (3, batch_size, rey, resx) -> (batch_size, rey, resx, 3)
    # indices = torch.stack(torch.meshgrid(torch.arange(batch_size), torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).to(dtype=torch.float32).to(device)
    indices = np.indices((batch_size ,h, w)).transpose(1, 2, 3, 0)
    depths[depths < 0] = 0
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
        batch_sign = np.zeros((batch_size, h, w)).to(device)
        for i in range(batch_size):
            batch_sign[i] = i + 1
        zero_filter = (depths != 0).reshape(-1)
        batch_sign = batch_sign.reshape(-1)[zero_filter]
        return xyz_img_trans[..., :3].reshape(-1, 3)[zero_filter], batch_sign, zero_filter

def find_match_from_ref(img:torch.Tensor, mask:torch.Tensor, ref:list, ref_mask:list, device='cuda')->torch.Tensor:
    """
    using a list of reference images to find the matched points in the image
    Args:
        img (torch.tensor): (height, width, channels)
        mask (torch.tensor): (height, width)
        ref (list): list of torch.Tensor: (height, width, channels)
        ref_mask (list): list of torch.Tensor: (height, width)
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        torch.tensor: (num_points, channels)
    """
    h, w, c = img.shape
    match = None
    for index in range(len(ref)):
        if mask == None:
            m1, m2 = find_match(img, ref[index], device)
        else:
            m1, m2 = find_match_masked(img, ref[index], mask, ref_mask[index], device)
        m1 = m1.cpu().numpy()
        if index == 0:
            match = m1
        else:
            match = np.intersect1d(match, m1)
    ### finally the `match` will be the indices of the matched points (in img)
    return match

def get_matched_pt_ft(imgs:torch.Tensor, depths:torch.Tensor, points:torch.Tensor, device='cuda'):
    """
    **BATCH VERSION**

    Args:
        img (torch.tensor): (batch_size, height, width, channels)
        depth (torch.tensor): (batch_size, height, width)
        points (torch.tensor): (batch_size, height, width, 3)
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        torch.tensor: (num_points, channels)
    """
    batch_size, h, w, c = imgs.shape
    match_points = []
    match_features = []
    for index in range(imgs.shape[0]):
        img = imgs[index]
        point = points[index]
        ref = []
        ref_mask = []
        if index == 0:
            ref.append(imgs[index + 1])
            ref_mask.append(depths[index + 1])
        elif index == imgs.shape[0] - 1:
            ref.append(imgs[index - 1])
            ref_mask.append(depths[index - 1])
        else:
            ref.append(imgs[index - 1])
            ref_mask.append(depths[index - 1])
            ref.append(imgs[index + 1])
            ref_mask.append(depths[index + 1])
        match = find_match_from_ref(img, depths[index], ref, ref_mask, device)
        match_point = point.reshape(-1, 3)[match]
        match_feature = img.reshape(-1, c)[match]
        print('match point shape: ', match_point.shape)
        match_points.append(match_point)
        match_features.append(match_feature)
    return torch.cat(match_points, dim=0), torch.cat(match_features, dim=0)

def features_cache(func):

    def wrapper(imgs, scale, key=0, img_num=0, img_size=224, name='bunny'):
        dr_path = os.path.join('/home/user/wangqx/stanford/cache_data', '{}_{}'.format(name, key))
        os.makedirs(dr_path, exist_ok=True)
        dd_path = os.path.join(dr_path, 'feats{}_{}_{}_dino.npy'.format(img_num, img_size, scale))
        print(dd_path)
        if os.path.exists(dd_path):
            features = np.load(dd_path)
            features = torch.from_numpy(features).to(torch.device('cuda'))
        else:
            torch.hub.set_dir( "/home/user/wangqx/stanford/")
            model = torch.hub.load('/home/user/wangqx/stanford/Learning_based_method/dinov2', 'dinov2_vitb14', source='local', pretrained=False).cuda()
            model.load_state_dict(torch.load('/home/user/wangqx/ndf_robot/dinov2_vitb14_pretrain.pth'))
            if isinstance(imgs, np.ndarray):
                features = func(model, imgs, scale)
            else:
                features = func(model, imgs.cpu().numpy(), scale)
            np.save(dd_path, features.cpu().numpy())
        return features
    return wrapper

@features_cache
def get_features(model, imgs:np.ndarray, scale:int=4, name='bunny')->torch.Tensor:
    """
    the imgs should be of shape (N, H, W, 3)

    Args:
        model (_type_): the dinov2 model
        rgb (_type_): a stack of rgb images
        scale (int, optional): the scale to shrink the imgs. Defaults to 4.

    Returns:
        _type_: torch.Tensor of shape (N, H, W, F)
    """
    imgs = skimage.img_as_float32(imgs)
    feat_ls = []
    for rgb in imgs:
        img_raw = skimage.transform.resize(
                rgb,
                (rgb.shape[0] // 14 * 14 , rgb.shape[1] // 14 * 14 )
            )
        imgs = torch.from_numpy(img_raw)
        imgs = imgs[None, :].cuda()
        # print('img.shape', imgs.shape)
        with torch.no_grad():
            ### (batch size, 3, height, width)
            ret = model.forward_features(imgs.permute(0, 3, 1, 2))
        features = ret['x_norm_patchtokens']
        N, _, F = features.shape
        ### (height, width, features) torch.Tensor np.float32
        features = features.reshape(imgs.shape[1] // 14, imgs.shape[2] // 14, F).permute(2, 0, 1)
        features = torch.nn.functional.interpolate(features.unsqueeze(0), size=(rgb.shape[0] // scale , rgb.shape[1] // scale ), mode='bilinear', align_corners=False)
        features = features.squeeze(0).permute(1, 2, 0)
        feat_ls.append(features)
    features = torch.stack(feat_ls, dim=0)
    return features

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
    torch.hub.set_dir( "/home/user/wangqx/stanford/")
    model = torch.hub.load('/home/user/wangqx/stanford/Learning_based_method/dinov2', 'dinov2_vitb14', source='local', pretrained=False).cuda()
    model.load_state_dict(torch.load('/home/user/wangqx/ndf_robot/dinov2_vitb14_pretrain.pth'))
    h, w = img_raw.shape[0] // 14 * 14,  img_raw.shape[1] // 14 * 14
    img = skimage.transform.resize(
                img_raw,
                (img_raw.shape[0] // 14 * 14 , img_raw.shape[1] // 14 * 14 )
            ).astype('float32')
    img = torch.from_numpy(img)
    img = img[None, :].cuda()
    print('Picture for Dino size:', img.shape)
    # print('img.shape', imgs.shape)
    with torch.no_grad():
        ### (batch size, 3, height, width)
        ret = model.forward_features(img.permute(0, 3, 1, 2))
    # from pdb import set_trace; set_trace()
    features = ret['x_norm_patchtokens']
    N, _, F = features.shape
    ### (height, width, features) torch.Tensor np.float32
    features = features.reshape(img.shape[1] // 14, img.shape[2] // 14, F).permute(2, 0, 1)
    features = torch.nn.functional.interpolate(features.unsqueeze(0), size=(img_raw.shape[0] // scale , img_raw.shape[1] // scale ), mode='bilinear', align_corners=False)
    features = features.squeeze(0).permute(1, 2, 0)
    return features


def downsample(pc:torch.Tensor, res=0.005):
    """Down sample the point cloud using o3d voxel_down_sample_and_trace

    Args:
        pc (torch.Tensor): The point cloud to be downsampled (num, 3)
        res (int): _description_

    Returns:
        down_sampled_pt (torch.Tensor): The downsampled point cloud (num, 3)
        index (np.ndarray): The index of the downsampled points in the original point cloud
    """
    device = pc.device
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
    _, _, idxs = pcd.voxel_down_sample_and_trace(res, pcd.get_min_bound(), pcd.get_max_bound())
    res = []
    for idx in idxs:
        res.append(np.random.choice(np.array(idx)))
    index = np.array(res)
    down_sampled_pt = torch.Tensor(pc[index]).to(device)
    return down_sampled_pt, index

def prune_sphere(points:torch.Tensor, center:torch.Tensor, radius:int = 0.3)->(torch.Tensor, torch.Tensor):
    """prune the useless points outside a shpere

    Args:
        points (torch.Tensor): (num, 3)
        center (torch.Tensor): (3, )
        center (int, optional): _description_. Defaults to 0.3)

    Returns:
        pruned_points: _description_
        index: the index of the pruned points in points
    """

    distence = torch.norm((points-center), p=2, dim=-1)
    index = distence < radius
    return points[index], index

def prune_box(points:torch.Tensor, x:list, y:list, z:list):
    """prune the useless points outside a box

    Args:
        points (torch.Tensor): (num, 3)
        x (list): [x_min, x_max]
        y (list): [y_min, y_max]
        z (list): [z_min, z_max]

    Returns:
        pruned_points: _description_
        index: the index of the pruned points in points
    """
    index = (points[:, 0] > x[0]) & (points[:, 0] < x[1]) & (points[:, 1] > y[0]) & (points[:, 1] < y[1]) & (points[:, 2] > z[0]) & (points[:, 2] < z[1])
    return points[index], index

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    key = 0
    mesh_path = '/home/user/wangqx/stanford/bunnyQ_Attack1_{}.obj'.format(key)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    img_num = 4
    scale = 4
    img_size = 1024
    imgs, depths_ori, c2w, K, camera_params = mesh2rgbd([mesh_path], device, num=img_num, img_size=img_size)
    print('#############')
    features = get_features(imgs, scale, key, img_num, img_size=img_size)
    n, h, w, c= features.shape
    depths = torch.from_numpy(np.array([cv2.resize(depth, (h , w), interpolation=cv2.INTER_NEAREST) for depth in depths_ori.cpu().numpy()])).to(device)
    points = depth2pt(depths, camera_params, torch.from_numpy(c2w).to(device).to(torch.float32), device=device)
    match_points, match_features = get_matched_pt_ft(features, depths, points, device=device)
    print('match_points.shape: ', match_points.shape)
    pt_vis(match_points.cpu().numpy())
    