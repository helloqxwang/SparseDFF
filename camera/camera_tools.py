import open3d as o3d
import numpy as np
import json
import os
import cv2
from pyk4a import Config, PyK4A, connected_device_count, Calibration
import pyk4a
import yaml
from scipy.spatial.transform import Rotation 
import open3d as o3d


CAM = {
    "cam0": '000299113912',
    "cam1": '000272313912',
    "cam2": '000285613912',
    "cam3": '000262413912',
}
CAM_INDEX = [CAM['cam0'], CAM['cam1'], CAM['cam2'], CAM['cam3']]

# CAM_INDEX = [ CAM['cam1']]

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

def vis_color_pc(points:np.ndarray, colors:np.ndarray, size:int=0.1, save=False, rot=None):
    """view the point cloud with colors

    Args:
        points (np.ndarray): (n, 3)
        colors (np.ndarray): (n, 3)
    """
    if size == 0:
        size = 0.1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    if rot is not None:
        axis.rotate(rot, center=[0, 0, 0])
    if save:
        o3d.io.write_point_cloud('./vis_pcd.ply', pcd)
    else:
        o3d.visualization.draw_geometries([pcd, axis])


def vis_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_colorpc(points, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(path, pcd)

def load_color_pc(path, mix=False):
    """Load the colors and the points sperately
    and then stack them together
    n must be equal to h*w

    Args:
        path (str): the path of the point cloud and the color

    Returns:
        points: np.ndarray shape: (cam_num, n, 3)
        colors: np.ndarray shape: (cam_num, h, w, 3)
        if mix:
            points: np.ndarray shape: (cam_num*n, 3)
            colors: np.ndarray shape: (cam_num*h*w, 3)
    """
    colors_ls = []
    points_ls = []
    for serial_num in CAM_INDEX:
        points = np.load(os.path.join(path, serial_num, 'points.npy'))
        colors = np.load(os.path.join(path, serial_num, 'colors.npy'))
        colors_ls.append(colors)
        points_ls.append(points)
    points = np.stack(points_ls, axis=0)
    colors = np.stack(colors_ls, axis=0)
    if mix:
        points = points.reshape((-1, 3))
        colors = colors.reshape((-1, 3))
    return points, colors

def load_depths(path:str) -> np.ndarray:
    """load the depths of the cameras

    Args:
        path (str): path

    Returns:
        np.ndarray: (cam_num, h, w)
    """

    depths_ls = []
    for serial_num in CAM_INDEX:
        depths = np.load(os.path.join(path, serial_num, 'depth.npy'))
        depths_ls.append(depths)
    depths = np.stack(depths_ls, axis=0)
    return depths


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


def get_extrinsics_from_json(path:str):
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

    table2cam, cam2base = read_tranformation()
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

def transform_points(points:np.ndarray, extrinsics:np.ndarray):
    """transform the points from their own camera frame to the world frame

    Args:
        points (np.ndarray): points in the camera frame shape PER CAMERA (cam_num, n, 3)
        extrinsics (np.ndarray): CAM2WORLD of the cams (cam_num, 4, 4)

    Returns:
        np.ndarray: points in the world frame shape: (cam_num*n, 3)
    """
    extrinsics = np.linalg.inv(extrinsics)
    points = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=-1)
    points = np.matmul(extrinsics, points.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(-1, 4)
    return points[..., :3]

def convert_opencv_distortion_to_multical(distortion:np.ndarray) -> np.ndarray:
    """convert the distortion from opencv version(From the pyk4a) to multical version
    (k1, k2, k3, p1, p2)
    
    """
    standard_distortion = np.zeros((1, 5))
    standard_distortion[0, 0] = distortion[0]
    standard_distortion[0, 1] = distortion[1]
    standard_distortion[0, 2] = distortion[4]
    standard_distortion[0, 3] = distortion[2]
    standard_distortion[0, 4] = distortion[3]
    return standard_distortion

def get_intrinsics_distortion_from_cams(res=pyk4a.ColorResolution.RES_1080P) -> (np.ndarray, np.ndarray):
    """get the intrinsics and distortion of the cameras

    Returns:
        intrinsics :np.ndarray (cam_num, 3, 3)
        distortion :np.ndarray (cam_num, 5)
    """

    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        exit()
    print(f"Available devices: {cnt}")
    intrinsics_ls = []
    distortion_ls = []
    for device_id in range(cnt):
        device = PyK4A(
            Config(
                color_resolution=res,
                camera_fps=pyk4a.FPS.FPS_5,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            ), 
            device_id=device_id,
        )
        device.open()
        print(f"{device_id}: {device.serial}")
        device.start()
        intrinsics = device.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion = device.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
        standard_distortion = convert_opencv_distortion_to_multical(distortion)
        intrinsics_ls.append(intrinsics)
        distortion_ls.append(standard_distortion)
    return np.stack(intrinsics_ls, axis=0), np.stack(distortion_ls, axis=0)

def get_intrinsics_distortion_from_npy(path:str, max_load_cams:int = 20, zero:bool=False, standard:bool=False) -> (np.ndarray, np.ndarray):
    """get the intrinsics and distortion of the cameras from path (workplace)

    Returns:
        intrinsics :np.ndarray (cam_num, 3, 3)
        distortion :np.ndarray (cam_num, 5)
    """
    intrinsics_ls = []
    distortion_ls = []
    for i in range(max_load_cams):
        if not os.path.isdir(os.path.join(path, f"cam{i}")):
            break
        intrinsics = np.load(os.path.join(path, f"cam{i}", 'intrinsic.npy'))
        distortion = np.load(os.path.join(path, f"cam{i}", 'distortion.npy'))
        intrinsics_ls.append(intrinsics)
        standard_distortion = convert_opencv_distortion_to_multical(distortion)
        if standard:
            distortion_ls.append(standard_distortion)
        else:
            distortion_ls.append(distortion)
    if zero:
        return np.stack(intrinsics_ls, axis=0), np.zeros_like(np.stack(distortion_ls, axis=0))
    else:
        return np.stack(intrinsics_ls, axis=0), np.stack(distortion_ls, axis=0)

def write_intrinsics_distortion2json(intrinsics:np.ndarray, distortion:np.ndarray, json_path:str):
    """
    write the intrinsics and distortion to the multical intrinsics json file

    Args:
        intrinsics (np.ndarray): (cam_num, 3, 3)
        distortion (np.ndarray): (cam_num, 5)
        path (str): path to save the json file
    """
    assert intrinsics.shape[0] == distortion.shape[0], "cam_num of intrinsics and distortion should be the same"
    with open(json_path, 'r') as f:
        data = json.load(f)
    for index in range(intrinsics.shape[0]):
        data['cameras'][f'cam{index}']['K'] = intrinsics[index].tolist()
        data['cameras'][f'cam{index}']['dist'] = distortion[index].tolist()
    dir_name, file_name = os.path.split(json_path)
    with open(os.path.join(dir_name, 'intrinsic_opt.json'), 'w') as f:
        json.dump(data, f)

def read_tranformation(data_path:str='/home/user/wangqx/stanford/kinect/tranform.yaml'):
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

def read_hand_arm(forder_path:str='/home/user/wangqx/stanford/kinect/hand_arm', name=None):
    """
    read the hand and arm points from the forder

    Args:
        forder_path (str, optional): path. Defaults to '/home/user/wangqx/stanford/kinect/hand_arm'.

    Returns:
        np.ndarray: shape (n, 1, 3 + 6 + hand_dofs) n is the number of poses
    """
    file_ls = os.listdir(forder_path)
    hand_path_ls:list = [item for item in file_ls if 'hand' in item]
    arm_path_ls:list = [item for item in file_ls if 'arm' in item]
    pose_ls = []
    for i in range(len(hand_path_ls)):
        hand_path = os.path.join(forder_path, hand_path_ls[i])
        arm_path = os.path.join(forder_path, arm_path_ls[i])
        hand = np.load(hand_path)
        arm = np.load(arm_path)
        posi = arm[:3]
        rot_matrix = Rotation.from_rotvec(arm[3:]).as_matrix()
        # this is in the base_link frame (TCP2baselink)
        rot_m = np.eye(4)
        rot_m[:3, :3] = rot_matrix
        rot_m[:3, 3] = posi

        baselink2base = np.eye(4)
        baselink2base[:3, :3] = Rotation.from_euler('xyz', [0, 0, np.pi]).as_matrix()
        rot_m = baselink2base @ rot_m


        table2cam, cam2base = read_tranformation()
        base2world = np.linalg.inv(table2cam) @ np.linalg.inv(cam2base)
        # this is in the world frame (TCP2world)
        rot_m = base2world @ rot_m

        # hand2EE
        hand2EE = np.eye(4)
        hand2EE[:3, :3] = Rotation.from_euler('xyz', [0, 0, np.pi]).as_matrix()
        hand2EE[:3, 3] = [0, -0.01, 0.247]
        rot_m = rot_m @ hand2EE 


        rot_world = rot_m[:3, :3]
        posi_world = rot_m[:3, 3]

        a1 = rot_world[0]
        a2 = rot_world[1]
        rot_6d = np.concatenate([a1, a2])
        hand_pose = np.concatenate([posi_world, rot_6d, hand[2:]])[None, :]
        pose_ls.append(hand_pose)
    poses = np.stack(pose_ls, axis=0)
    if name is not None:
        idx = hand_path_ls.index(f'hand_{name}.npy')
        return poses[idx]
    return poses


if __name__ == '__main__':
    table2cam, cam2base = read_tranformation()
    frame_table = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    frame_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    frame_cam = frame_cam.transform(np.linalg.inv(table2cam))
    frame_base.transform(np.linalg.inv(cam2base)).transform(np.linalg.inv(table2cam))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(frame_table)
    vis.add_geometry(frame_cam)
    vis.add_geometry(frame_base)

    vis.run()
    vis.destroy_window()
