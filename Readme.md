# SparseDDF:Sparse-View Feature Distillation for One-Shot Dexterous Manipulation
We introduce a novel method for acquiring view-consistent 3D DFF s from sparse RGBD observations, enabling one-shot learning of dexterous manipulations that are transferable to novel scenes. 

## Overview
### Constructing the Feature Field
Initially, we map the image features to the 3D point cloud, allowing for propagation across the 3D space to establish a dense feature field.

Then, A lightweight feature refinement network optimizes with a contrastive loss between pairwise views after back-projecting the image features onto the 3D point cloud. 

Additionally, we implement a point-pruning mechanism to augment feature continuity within each local neighborhood. 

### Optimize the EE Pose
By establishing coherent feature fields on both source and target scenes, we devise an energy function that facilitates the minimization of feature discrepancies w.r.t. the end-effector parameters between the demonstration and the target manipulation. 

## Installation
```
conda create -n sparsedff python=3.9
conda env update -f conda.yml
```

## Usage
### Test with our given models

> All configurations is managed in `config.yaml`.
An original config.yaml is offered.

Test with our default test data:
```
python unified_optimize.py 
```
After finishing the test, a clear visulization using trimesh will be presented if `visualize true` in config.yaml. Otherwise, a picture of the visulization will be saved.

## Visualize

We provide the `vis_features.py` script in order to perform further fine-grained visualization on the data in the  `./data `directory.

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
