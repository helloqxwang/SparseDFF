#! /bin/bash

if [ -f ./thirdparty_module/dinov2_vitb14_pretrain.pth ]
then 
    echo "dino pth exsits"
else
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -P ./thirdparty_module/
fi

if [ -f ./thirdparty_module/sam_vit_h_4b8939.pth ]
then 
    echo "sam pth exsits"
else
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./thirdparty_module/
fi