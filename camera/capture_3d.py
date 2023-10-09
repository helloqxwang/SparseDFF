import datetime
import os
import numpy as np
import argparse
import pyk4a
from pyk4a import Config, PyK4A, connected_device_count, Calibration
import open3d as o3d
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .camera_tools import vis_color_pc
# from match_train.match_tools import 


def capture_mannul(args):
    
    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        exit()
    print(f"Available devices: {cnt}")
    now = datetime.datetime.now()
    str_time = now.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.path, str_time)
    for device_id in range(cnt):
        device = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_2160P,
                camera_fps=pyk4a.FPS.FPS_5,
                depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                synchronized_images_only=True,
            ), 
            device_id=device_id,
        )
        device.open()
        print(f"{device_id}: {device.serial}")
        device.start()
        # getters and setters directly get and set on device
        device.whitebalance = 4500
        assert device.whitebalance == 4500
        device.whitebalance = 4510
        assert device.whitebalance == 4510
        
        for i in range(20):
            capture = device.get_capture()
            # if np.any(capture.depth) and np.any(capture.color):
            #     break
        store_path = os.path.join(path, f"{device.serial}")
        os.makedirs(store_path, exist_ok=True)
        points = capture.transformed_depth_point_cloud
        depths = capture.transformed_depth
        intrinsic = device.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion = device.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
        colors = capture.color[..., :3]
        print('points shape: ', points.shape)
        print('colors shape: ', colors.shape)
        print('depths shape: ', depths.shape)
        print('intrinsic shape: ', intrinsic.shape)
        print('distortion shape: ', distortion.shape)

        if args.save:
            np.save(os.path.join(store_path, 'intrinsic.npy'), intrinsic)
            np.save(os.path.join(store_path, 'distortion.npy'),distortion)
            np.save(os.path.join(store_path, 'points.npy'), points)
            np.save(os.path.join(store_path, 'colors.npy'), colors)
            np.save(os.path.join(store_path, 'depth.npy'), depths)
            cv2.imwrite(os.path.join(store_path, 'colors.png'), capture.color)

        if args.vis_3d:
            vis_color_pc(points.reshape(-1,3), capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)))
        if args.vis_2d:
            cv2.imshow('color', colors)
            cv2.imshow('depth', capture.depth)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def capture_auto(save=True, path='/home/user/wangqx/stanford/kinect/data', name='mm'):
    """return the points, colors, depths, intrinsic, distortion

    Args:
        save (bool, optional): _description_. Defaults to False.
        path (str, optional): _description_. Defaults to '/home/user/wangqx/stanford/kinect/data'.

    Returns:
        points: np.ndarray (cam_num, h, w, 3)
        colors: np.ndarray (cam_num, h, w, 3)
        depths: np.ndarray (cam_num, h, w)
        intrinsic: np.ndarray (cam_num, 3, 3)
        distortion: np.ndarray (cam_num, 8)
    """
    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        exit()
    print(f"Available devices: {cnt}")
    now = datetime.datetime.now()
    str_time = name
    path = os.path.join(path, str_time)
    points_list = []
    colors_list = []
    depths_list = []
    intrinsic_list = []
    distortion_list = []
    for device_id in range(cnt):
        device = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_2160P,
                camera_fps=pyk4a.FPS.FPS_5,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            ), 
            device_id=device_id,
        )
        device.open()
        print(f"{device_id}: {device.serial}")
        device.start()
        # getters and setters directly get and set on device
        device.whitebalance = 4500
        assert device.whitebalance == 4500
        device.whitebalance = 4510
        assert device.whitebalance == 4510
        
        for i in range(20):
            capture = device.get_capture()
            # if np.any(capture.depth) and np.any(capture.color):
            #     break
        store_path = os.path.join(path, f"{device.serial}")
        os.makedirs(store_path, exist_ok=True)
        points = capture.transformed_depth_point_cloud
        depths = capture.transformed_depth
        intrinsic = device.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion = device.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
        colors = capture.color[..., :3]
        print('points shape: ', points.shape)
        print('colors shape: ', colors.shape)
        print('depths shape: ', depths.shape)
        print('intrinsic shape: ', intrinsic.shape)
        print('distortion shape: ', distortion.shape)
        points_list.append(points)
        colors_list.append(colors)
        depths_list.append(depths)
        intrinsic_list.append(intrinsic)
        distortion_list.append(distortion)
        if save:
            print(os.path.join(store_path, 'intrinsic.npy'))
            np.save(os.path.join(store_path, 'intrinsic.npy'), intrinsic)
            np.save(os.path.join(store_path, 'distortion.npy'),distortion)
            np.save(os.path.join(store_path, 'points.npy'), points)
            np.save(os.path.join(store_path, 'colors.npy'), colors)
            np.save(os.path.join(store_path, 'depth.npy'), depths)
            cv2.imwrite(os.path.join(store_path, 'colors.png'), capture.color)
    points = np.stack(points_list, axis=0)
    colors = np.stack(colors_list, axis=0)
    depths = np.stack(depths_list, axis=0)
    intrinsic = np.stack(intrinsic_list, axis=0)
    distortion = np.stack(distortion_list, axis=0)
    
    return points, colors, depths, intrinsic, distortion

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default='./data')
    argparser.add_argument('--vis_3d', action='store_true')
    argparser.add_argument('--vis_2d', action='store_true')
    argparser.add_argument('--save', action='store_true')
    args = argparser.parse_args()
    capture_mannul(args)