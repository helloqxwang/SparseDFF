import numpy as np
from camera_tools import get_intrinsics_distortion_from_cams, get_intrinsics_distortion_from_npy, write_intrinsics_distortion2json
from pyk4a import Config, PyK4A, connected_device_count, Calibration
import pyk4a

intrinsics, distortion = get_intrinsics_distortion_from_npy('./workspace', standard=False, zero=False)
write_intrinsics_distortion2json(intrinsics, distortion, './workspace/intrinsic.json')

