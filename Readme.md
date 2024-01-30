# SparseDFF: Sparse-View Feature Distillation for One-Shot Dexterous Manipulation
We introduce a novel method for acquiring view-consistent 3D DFFs from sparse RGBD observations, enabling one-shot learning of dexterous manipulations that are transferable to novel scenes. 

***[ICLR 2024](https://iclr.cc/) Accepted***

 [Project](https://helloqxwang.github.io/SparseDFF/) | [Full Paper](https://arxiv.org/abs/2310.16838) 

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhelloqxwang%2FSparseDFF&count_bg=%23F3A7D2&title_bg=%23F96E28&icon=microbit.svg&icon_color=%2325A837&title=Hi%21&edge_flat=false)](https://hits.seeyoufarm.com)

## What's Inside This Repository!

- A brief Introduction to SparseDFF.
- Everything mentioned in the paper and a **step-by-step guide** to use it!
- A Friendly and Useful [guide](./Data_Collection.md) for **Kinect Installation** and **everything you need for data colection**!. Additional Codes for *automatic image capturing during manipulation transfer at runtime*.
- Additional codes for EE besides Shadow Hand (Comming soon!)



## Method Overview
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

# Install submodules
cd ./SparseDFF/thirdparty_module/dinov2
pip install -r requirements.txt
cd ../pytorch_kinematics
pip install -e .
cd ../pytorch3d
pip install -e .
cd ../segment-anything
pip install -e .

# download relevant packages
pip install open3d opencv-python scikit-image trimesh lxml pyvirtualdisplay pyglet==1.5.15

# download the pretrained-model for sam and dinov2
bash pth_download.sh
# download the example data
bash download.sh
```



## Camera Installation

>  If you simply wish to run our model (including both training and inference) on pre-captured data, there's no need for the following installation steps. Go ahead and start playing with SparseDFF directly! 
>
> However, if you intend to collect your own data using Azure Kinect, please proceed with the following setup.

[Friendly and Useful Installation guide for Kinect SDK](./Data_Collection.md)



## Step-by-step Tutorial

### Quick Start: Testing with Our Provided Data & Models

If you'd like to quickly experience the results of our model, you can directly use our default example data for testing.

#### Testing Steps

1. **Configuration Management**: All configurations are managed via the `config.yaml` file. You can find detailed usage instructions and comments for each parameter within this file.

2. **Running the Test**: To test using the default example data, execute the following command:

   ```
   python unified_optimize.py
   ```

3. **A brief Visualization of Result **: After completing the test, if `visualize` is set to `true` in `config.yaml`, a clear trimesh visualization will be presented. Otherwise, the visualization result will be saved as an image.



### Play with your Own Data

If you wish to test with your own data, follow these steps!

#### Data Collection

> You need to complete the [configuration](./Data_Collection.md) of the Kinect Camera before running the subsequent code. You can also collect data in your own way and then organize it into the same data structure.

We utilize four [Azure Kinects](https://azure.microsoft.com/en-us/products/kinect-dk) for data collection. The steps are as follows:

**Capturing 3D Data**: Use`camera/capture_3d.py` to collect the data required for the model:

```bash
cd camera
python capture_3d.py --save
```

***Structure of the Data*:**

```bash
|____20231010_monkey_original # name
| |____000262413912 # the serial number of the camera
| | |____depth.npy # depth img
| | |____colors.npy # rgb img
| | |____intrinsic.npy # intrinsic of the camera
| | |____points.npy # 3D points calculated from the depth and colors
| | |____colors.png # rgb img (.png for quick view)
| | |____distortion.npy # Distortion
| | ... # Data of other cameras
```

#### Training the Refinement Model

To train the refinement model corresponding to your data, follow these steps:

1. **Data Preparation**: Navigate to the refinement directory and execute the following commands:

   ```bash
   cd refinement
   python read.py --img_data_path 20231010_monkey_original
   ```

2. **Start Training**: Run the following command to train the model:

   ```bash
   python train_linear_probe.py --mode glayer --key 0
   ```

3. **Model Configuration**: After training, you can find the corresponding checkpoint in the refinement directory, and then update the `model_path` in `config.yaml` to the appropriate path.

### Testing with Your Data

Before proceeding with the testing, it's important to ensure that your newly collected reference data, the trained refinement network, and the test data are correctly specified in the `config.yaml` file. Once these elements are confirmed and properly set up in the configuration, you can directly execute：

```bash
python unified_optimize.py
```

We also provide a pipeline for **automatic image capturing during manipulation transfer at runtime**.

1. **Configure the Camera Environment**: Set up the camera environment as described in the data collection section.
2. **Automatic Image Capture**: After configuring the camera environment, change the value of `data2` in the `config.yaml` file to `null` to enable automatic image capturing during reference.

With these steps, you can either quickly test our model with provided materials or delve deeper into experiments and model training with your own data. We look forward to seeing the innovative applications you create using our model!



# Visualize

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
