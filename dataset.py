# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

import tqdm
import monai

import itertools
from collections.abc import Sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final
from sklearn.model_selection import train_test_split

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.nets.swin_unetr import SwinTransformer, MERGING_MODE

from monai.networks.nets import SEResNet50, SEResNet101
from monai.networks.blocks.squeeze_and_excitation import SEBottleneck, SEResNetBottleneck


class ResampleZ:
    def __init__(self, new_depth):
        self.new_depth = new_depth

    def __call__(self, img):
        # Assuming img has shape (D, H, W)
        d, h, w = img.shape
        
        # Add channel dimension (C, D, H, W) where C=1
        img = img.unsqueeze(0)

        # Interpolate
        img = F.interpolate(img.unsqueeze(0), size=(self.new_depth, h, w), mode='trilinear', align_corners=False)

        # Remove added dimensions
        img = img.squeeze(0).squeeze(0)

        return img


class RSNADataset(monai.data.Dataset):
    def __init__(self, df=None, train_series_descriptions=None, is_train=True):
        self.is_train = is_train
        self.df = df
        
        if is_train:
            self.study_ids = list(set(df['study_id'].values.tolist()))
        else:
            self.study_ids = list(set(train_series_descriptions['study_id'].values.tolist()))

        self.train_series_descriptions = train_series_descriptions
        self.labels = [
            'spinal_canal_stenosis_l1_l2', 
            'spinal_canal_stenosis_l2_l3',
            'spinal_canal_stenosis_l3_l4', 
            'spinal_canal_stenosis_l4_l5',
            'spinal_canal_stenosis_l5_s1', 
            'left_neural_foraminal_narrowing_l1_l2',
            'left_neural_foraminal_narrowing_l2_l3',
            'left_neural_foraminal_narrowing_l3_l4',
            'left_neural_foraminal_narrowing_l4_l5',
            'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2',
            'right_neural_foraminal_narrowing_l2_l3',
            'right_neural_foraminal_narrowing_l3_l4',
            'right_neural_foraminal_narrowing_l4_l5',
            'right_neural_foraminal_narrowing_l5_s1',
            'left_subarticular_stenosis_l1_l2', 
            'left_subarticular_stenosis_l2_l3',
            'left_subarticular_stenosis_l3_l4', 
            'left_subarticular_stenosis_l4_l5',
            'left_subarticular_stenosis_l5_s1', 
            'right_subarticular_stenosis_l1_l2',
            'right_subarticular_stenosis_l2_l3',
            'right_subarticular_stenosis_l3_l4',
            'right_subarticular_stenosis_l4_l5',
            'right_subarticular_stenosis_l5_s1'
        ]
        self.target_mapping = {
            "Normal/Mild": [1, 0, 0],
            "Moderate":[0, 1, 0],
            "Severe": [0, 0, 1]
        }
        self.ss_mapping = {
            0: "normal_mild", 1: "moderate", 2: "severe"
        }
        self.data_dir = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        series_descriptions = self.train_series_descriptions.loc[self.train_series_descriptions.study_id == study_id]  # series_id, series_description

        data = {}
        for series_id, series_description in series_descriptions[["series_id", "series_description"]].values:
            if series_description == "Sagittal T2/STIR":
                
                if self.is_train:
                    path_to_dicom_dir = f"{self.data_dir}/train_images/{study_id}/{series_id}"
                else:
                    path_to_dicom_dir = f"{self.data_dir}/test_images/{study_id}/{series_id}"
                    
                dicom = self.transform(path_to_dicom_dir)
                data[series_description] = dicom

        data["study_id"] = study_id
        
        if self.is_train:
            data["target"] = self.target_processing(index)

        return data
    
    def target_processing(self, index):
        row = self.df.iloc[[index]]
        
        data = []
        for label in self.labels:
            severity = row[[label]].values[0][0]
            target = self.target_mapping[severity]
            data.append(target)
            
        return np.asarray(data)
    
    
    def transform(self, path_to_dicom_dir):
        transform = monai.transforms.Compose([
            monai.transforms.LoadImage(),
            monai.transforms.EnsureChannelFirst(channel_dim=-1),
            ResampleZ(new_depth=7),
            monai.transforms.Resize((384, 384), mode='trilinear'),
            monai.transforms.NormalizeIntensity(),
            monai.transforms.ToTensor()
        ])
        return transform(path_to_dicom_dir)
