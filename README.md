# AutoSAM2-VSOD
AutoSAM2-VSOD: Adapting SAM2 to VSOD by Overloading the Prompt Encoder

## Overview

This work adapts the Segment Anything Model 2 (SAM2) for Video Saliency Object Detection (VSOD) by replacing its conditioning mechanism with an image-based encoder. Without further fine-tuning SAM, this modification trains SAM2 to segment the salient object or objects in images. 

## Paper

The paper associated with this repository can be found [here]().

## Datasets

We used the following datasets linked to in the following repositories in our experiments:

[DAVSOD](https://github.com/DengPingFan/DAVSOD)
[ViDSOD-100](https://github.com/jhl-Det/RGBD_Video_SOD)

## SAM2 checkpoints

Linked to in original SAM2 repository:
[SAM2](https://github.com/facebookresearch/sam2)

## Usage

To use AutoSAM2-VSOD, follow these steps:

training:

python train.py --root_data_dir <~/dataset_directory>

inference:

python inference.py --root_data_dir <~/dataset_directory> --folder <train model folder number>

## Sources and Referenced Repositories

We started our project using the code from the AutoSAM project github page. We also worked with the code for SAM, SAM2, and ESANet. Noise augmentations were based on those from the ViDSOD-100 code.
Links to the above are included below:
https://github.com/talshaharabany/AutoSAM
https://github.com/facebookresearch/segment-anything
https://github.com/facebookresearch/sam2
https://github.com/TUI-NICR/ESANet
https://github.com/jhl-Det/RGBD_Video_SOD
