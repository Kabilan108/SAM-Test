#!/bin/bash

# Check if you are in the `SAM-Test` directory
if [ $(pwd) != *"/SAM-Test" ]; then
    echo "Please run this script from the root directory of the repository"
    exit 1
fi

# Download the models
curl "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o "models/sam_vit_h_4b8939.pth"
curl "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" -o "models/sam_vit_l_0b3195.pth"
curl "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -o "models/sam_vit_b_01ec64.pth"

# Download images
mkdir -p "notebooks/images"
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg
