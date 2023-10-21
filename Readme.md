# SparseDFF: Sparse-View Feature Distillation for One-Shot Dexterous Manipulation
We introduce a novel method for acquiring view-consistent 3D DFF s from sparse RGBD observations, enabling one-shot learning of dexterous manipulations that are transferable to novel scenes. 

## Overview
### Constructing the Feature Field
Initially, we map the image features to the 3D point cloud, allowing for propagation across the 3D space to establish a dense feature field.

Then, A lightweight feature refinement network optimizes with a contrastive loss between pairwise views after back-projecting the image features onto the 3D point cloud. 

Additionally, we implement a point-pruning mechanism to augment feature continuity within each local neighborhood. 

### Optimize the EE Pose
By establishing coherent feature fields on both source and target scenes, we devise an energy function that facilitates the minimization of feature discrepancies w.r.t. the end-effector parameters between the demonstration and the target manipulation. 

## Installation
We provide sample data that allows you to directly conduct manipulation transfer within our collected data, and offer visualization code for visualizing the experimental results. Additionally, we also provide code for data collection using Kinect. For additional environment configuration and instructions, please refer to [Data Collection](#data-collection) Part.


We provide bash script for example data download and pretrained model download.
```
git clone --recurse-submodules git@github.com:Halowangqx/SparseDFF.git
conda create -n sparsedff python=3.9
cd ./SparseDFF/thirdparty_module/dinov2
pip install -r requirements.txt
cd ../pytorch_kinematics
pip install -e .
cd ../pytorch3d
pip install -e .
cd ../segment-anything
pip install -e .

sudo apt install xvfb 
pip install open3d opencv-python scikit-image trimesh lxml pyvirtualdisplay pyglet

# download the pretrained-model for sam and dinov2
bash pth_download.sh
# download the example data
bash download.sh
```



## Usage
### Test with our given models
All configurations are managed by the `config.yaml`. The usage of specific parameters can be found in the comments within the file. 

Test with our default example data:
```
python unified_optimize.py 
```

After finishing the test, a clear visulization using trimesh will be presented if `visualize` is `true` in `config.yaml``. Otherwise, a picture of the visulization will be saved.

### Data Collection
We use four [Azure Kinects](https://azure.microsoft.com/en-us/products/kinect-dk) for data collection. We provide `capture_3d.py` for capturing the data required for the model . `capture.py` is used for convenient image collection required for multi-calibration (with [multical](https://pypi.org/project/multical/)). Additionally, you can use [easy_handeye](https://github.com/IFL-CAMP/easy_handeye) for eye-on-base calibration, and save the calibration matrix, along with the pose matrix of the experiment table, in the tranform.yaml folder. The code will automatically read the saved information during runtime.

The file struction is as:
```bash
+-- camera
|  +-- capture_3d.py  
|  +-- capture.py
|  +-- transform.yaml
```
### Test our model with data captured in real time
Additionally, we provide a pipeline that allows for automatic capturing of images during manipulation transfer at runtime. After configuring the camera environment, you only need to change the value of `data2` in the `config.yaml` file to `null`.

## Visualize

We provide the `vis_features.py` script in order to perform further fine-grained visualization on the data in the  `./data `directory.

All the Visualizations will be saved in `./visulize` directory. 

Visualize the feature field using the similarity between 
```
python vis_features.py --mode 3Dsim_volume --key 0
```

Visualize the optimization Result 
```
python vis_features.py --mode result_vis --key=1 --pca --with_hand
```

Visualize the feature similarity between all points in reference pointcloud, test pointcloud and *a point in reference point cloud assigned by `--ref_idx`.

```
python vis_features.py --mode 3D_similarity --ref_idx 496 --similarity l2
```
