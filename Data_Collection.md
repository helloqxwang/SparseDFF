# Overview of the Data Collection Setup

This section outlines the Kinect environment setup used in our research. The setup integrates various tools and packages, enabling efficient interaction and calibration processes. The environment is implemented on **Ubuntu 20.04** with **ROS-noetic**.

## Components

- **Kinect Official SDK**: Provides the foundational environment.
- **[pyk4a](https://github.com/etiennedub/pyk4a)**: Facilitates interaction with the Kinect using Python.
- **[easyhandeye](https://github.com/IFL-CAMP/easy_handeye)**: Utilized for eye-on-base or eye-in-hand calibration.
- **[multical](https://pypi.org/project/multical/)**: Employs multi-calibration for Kinects.



There are also some docs/scripts I think probably helpful for your task!

- A script for taking simultaneous photos with all Kinects, including command arguments for 2D/3D result visualization: [capture_3d.py](https://github.com/Halowangqx/SparseDFF/blob/main/camera/capture_3d.py).
- A script for visualizing and organizing the RGB input from all Kinects, useful for multi-calibration with the `multical` package: [capture.py](https://github.com/Halowangqx/SparseDFF/blob/main/camera/capture.py).



# Installing All You Need for Kinect!

## Kinect SDK Installation

#### On Ubuntu 18.04
Installation on **ubuntu 18.04** is easy. You can simply follow the [official guide][https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md] then go ahead! :D

The following long Intrustrution is for poor guys using **ubuntu20.04** like me.

#### On Ubuntu 20.04

1. **Compilation from Source:**
   ```bash
   git clone git@github.com:microsoft/Azure-Kinect-Sensor-SDK.git
   cd Azure-Kinect-Sensor-SDK
   mkdir build && cd build
   cmake .. -GNinja
   ninja
   ```

   **Troubleshooting During Compilation:**
   - For issues related to `libudev.h`:
     ```bash
     sudo apt-get install libusb-1.0-0-dev
     sudo apt-get install libudev-dev
     ```

   - For errors regarding 'LIBSOUNDIO_LIB-NOTFOUND':
     ```bash
     sudo apt install libsoundio-dev
     ```

2. **SDK Installation:**
   ```bash
   curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
   sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
   curl -sSL https://packages.microsoft.com/config/ubuntu/18.04/prod.list | sudo tee /etc/apt/sources.list.d/microsoft-prod.list
   sudo apt-get update
   sudo apt install libk4a1.3-dev
   sudo apt install libk4abt1.0-dev
   sudo apt install k4a-tools=1.3.0
   sudo cp scripts/99-k4a.rules /etc/udev/rules.d/ 
   reboot
   ```

   **Error Resolution:**
   - In case of specific errors (e.g., the one shown in the provided image), manually install `libk4a1.3-dev`, `libk4abt1.0-dev`, and `k4a-tools=1.3.0` via `.deb` files.



## ROS SDK Compilation

Follow the [official guide](https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/building.md) for ROS SDK compilation.



## Installing the pyk4a Module

```bash
# Activate the working conda environment
conda activate sparsedff
pip install pyk4a
```



# Calibration

**Hand-Eye Calibration with easyhandeye**

The [easyhandeye](https://github.com/IFL-CAMP/easy_handeye) module is essential for precise hand-eye calibration between cameras and robotic arms. This calibration is critical in aligning the robot's movements with the camera's perspective, ensuring accurate and synchronized operations. By utilizing `easyhandeye`, we facilitate seamless interaction between the camera's vision and the robot's actions, enhancing the efficiency of our automated processes.

**Multi-Camera Calibration with multical**

The [multical](https://pypi.org/project/multical/) module plays a pivotal role in multi-camera calibration. This process involves aggregating and harmonizing data from multiple camera feeds, a task essential for comprehensive spatial understanding and data integration. 

Our approach includes collecting numerous sets of camera images, a process streamlined by the [`camera/capture.py`](https://github.com/Halowangqx/SparseDFF/blob/main/camera/capture.py)script. This script automates the photo capturing by responding to a simple key press ('C'), significantly simplifying the data collection phase of multi-camera calibration.

**Calibration Data Storage Structure**

The organization of calibration data is critical for efficient data retrieval and analysis. Our storage structure is systematically organized as follows:

```bash
|____capture.py # Colect data for multi-view calibration
|____sam.py
|____capture_3d.py
|____hand_arm
| |____ref_pose_monkey.npy
|____manager.py
|____camera_tools.py
|____transform.yaml # Store the result of eye-on-base calibration
|____workspace # Dir where the multi-view calibration performed
| |____intrinsic.json # intrinsic for cameras
| |____calibration.json # extrinsic for cameras 
| |____boards.yaml # aruco board
|______init__.py
```
