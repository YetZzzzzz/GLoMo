import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

DEVICE = torch.device("cuda:1")

# MOSI SETTING
#ACOUSTIC_DIM = 74
#VISUAL_DIM = 47
#TEXT_DIM = 768

# MOSEI SETTING
#ACOUSTIC_DIM = 74
#VISUAL_DIM = 35
#TEXT_DIM = 768

## URFUNNY SETTING
#ACOUSTIC_DIM = 81
#VISUAL_DIM = 371
#TEXT_DIM = 768
### URFUNNY-HKT SETTING
#ACOUSTIC_DIM = 81
#VISUAL_DIM = 371
#TEXT_DIM = 768
#
### SARCASM-HKT SETTING
#ACOUSTIC_DIM = 81
#VISUAL_DIM = 371
#TEXT_DIM = 768


## HKT-SETTING
visual_features_list=list(range(55,91))
acoustic_features_list=list(range(0,60))

ACOUSTIC_DIM = len(acoustic_features_list)
VISUAL_DIM = len(visual_features_list)
HCF_DIM=4
TEXT_DIM=768
VISUAL_DIM_ALL = 91
ACOUSTIC_DIM_ALL = 81
DATASET_LOCATION = "../datasets/"
