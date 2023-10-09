import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple
import os
import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, connected_device_count
import numpy as np

class Workspace(object):
    def __init__(self, save_dir:str, serial_numbers:List[str] ):
        self.save_dir = save_dir
        self.serial_numbers = serial_numbers
        self.folders = [os.path.join(self.save_dir, 'cam{}'.format(idx)) for idx, serial_number in enumerate(self.serial_numbers)]
        for idx in range(len(self.serial_numbers)):
            print('cam{}: {}'.format(idx, self.serial_numbers[idx]))
        self.img_folders = [os.path.join(folder, "image") for folder in self.folders]
        self.depth_folders = [os.path.join(folder, "depth") for folder in self.folders]
        for folder in self.img_folders:
            os.makedirs(folder, exist_ok=True)
        for folder in self.depth_folders:
            os.makedirs(folder, exist_ok=True)
        self.color_count = np.zeros(len(self.serial_numbers), dtype=np.int32)
        self.depth_count = np.zeros(len(self.serial_numbers), dtype=np.int32)

    def get_cam_path(self, idx):
        return self.folders[idx]
    
    def get_color_path(self, idx, color_only=True):
        if color_only:
            path = os.path.join(self.folders[idx], f"{self.color_count[idx]}.png")
        else:
            path = os.path.join(self.img_folders[idx], f"{self.color_count[idx]}.png")
        self.color_count[idx] += 1
        return path
    
    def get_depth_path(self, idx):
        path = os.path.join(self.depth_folders[idx], f"{self.depth_count[idx]}.png") 
        self.depth_count[idx] += 1
        return path

class MultiCameraManager(object):
    serial_numbers: List[str]

    def __init__(
        self,
        save_dir: Path = Path("."),
        img_res_lev:int = pyk4a.ColorResolution.RES_2160P,
        fps_lev: int = pyk4a.FPS.FPS_5,
        depth_mode: int = pyk4a.DepthMode.NFOV_UNBINNED,
        enable_color_stream: bool = True,
        enable_depth_stream: bool = True,
        enable_hardware_sync: bool = False,
        tranformed_color: bool = False,
        color_only:bool = True, 
        align_to: str = "color",
    ) -> None:
        
        assert align_to == "color"
        self.save_dir = save_dir
        self.img_res_lev = img_res_lev
        self.fps_lev = fps_lev
        self.depth_mode = depth_mode
        self.tranformed_color = tranformed_color
        self.color_only = color_only

        self.enable_color_stream = enable_color_stream
        self.enable_depth_stream = enable_depth_stream
        self.enable_hardware_sync = enable_hardware_sync

        # ct    os.makedirs(path, exist_ok=True)
        self.n_device = connected_device_count()
        self.devices:List[PyK4A] = []
        self.serial_numbers = []
        if not self.n_device:
            print("No devices available")
            exit()
        print(f"Available devices: {self.n_device:}")
        for device_id in range(self.n_device):
            device = PyK4A(
                Config(
                    color_resolution=self.img_res_lev,
                    camera_fps=self.fps_lev,
                    depth_mode=self.depth_mode,
                    synchronized_images_only=True,
                ), 
                device_id=device_id,
            )
            device.open()
            print(f"{device_id}: {device.serial}")
            device.start()
            self.devices.append(device)
            self.serial_numbers.append(device.serial)

        ### not implemented yet
        if self.enable_hardware_sync:
            self.hardware_sync()
        
        self.workspace:Workspace = Workspace(self.save_dir, self.serial_numbers)
        self.save_intrinsics_distortion()

    def save_intrinsics_distortion(self):
        for idx, device in enumerate(self.devices):
            intrinsic = device.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
            distortion = device.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
            np.save(os.path.join(self.workspace.get_cam_path(idx), 'intrinsic.npy'), intrinsic)
            np.save(os.path.join(self.workspace.get_cam_path(idx), 'distortion.npy'), distortion)

    def hardware_sync(self):
        if self.n_device == 1:
            return

        raise NotImplementedError


    def wait_for_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        depth_frames: list = []
        color_frames: list = []

        for i, device in enumerate(self.devices):
            capture = device.get_capture()

            # points = capture.depth_point_cloud.reshape((-1, 3))
            if self.tranformed_color:
                color_frame = capture.transformed_color
                print(color_frame.shape)
            else:
                color_frame = capture.color
            if not self.color_only:
                depth_frame = capture.depth
                depth_frames.append(depth_frame)
            color_frames.append(color_frame)
            cv2.namedWindow(f"frame_{i}",cv2.WINDOW_NORMAL)
            cv2.imshow(f"frame_{i}", color_frame)
            time.sleep(0.5)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("c"):
            print(self.enable_color_stream)
            if self.enable_color_stream:
                for i, frame in enumerate(color_frames):
                    cv2.imwrite(
                        str(
                            self.workspace.get_color_path(i)
                        ),
                        frame,
                    )
            if not self.color_only:
                if self.enable_depth_stream:
                    for i, frame in enumerate(depth_frames):
                        cv2.imwrite(
                            str(
                                self.workspace.get_depth_path(i)
                            ),
                            frame,
                        )

        return depth_frames, color_frames
