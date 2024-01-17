# Kinect_Doc

We use:
- Kinect Official SDK to provide the base env.
- pyk4a https://github.com/etiennedub/pyk4a to interact with kinect using python
- easyhandeye https://github.com/IFL-CAMP/easy_handeye to perform eye-on-base or eye-in-hand calibration.
- multical https://pypi.org/project/multical/ to perform multi-calibration for kinects.

We use all the above packages in Unbuntu20.04 with ROS-noetic.(That's because Kinect SDK is not compatable with Ubuntu22.04 and18.04 is too old :D)


There are also some docs/scripts I think probably helpful for your task~
- https://github.com/Halowangqx/SparseDFF/blob/main/camera/capture_3d.py This piece of code makes all Kinects take a photo once and containing some cmd arguments to visualize the result in 2d/3d.
- https://github.com/Halowangqx/SparseDFF/blob/main/camera/capture.py This piece of code can vis all the kinects' rgb input in a fixed fps and save them in an organized way every time you press 'C' in the keyboard. That's very helpful when doing multicalibration with `multical` package.
