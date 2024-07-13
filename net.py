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

from config import args, device

class SEResNet101Custom(SEResNet101):
    """SEResNet50 based on `Squeeze-and-Excitation Networks` with optional pretrained support when spatial_dims is 2."""

    def __init__(
        self,
        layers: Sequence[int] = (3, 4, 6, 3),
        groups: int = 1,
        reduction: int = 16,
        dropout_prob: float | None = None,
        inplanes: int = 64,
        downsample_kernel_size: int = 1,
        input_3x3: bool = False,
        pretrained: bool = True,
        progress: bool = True,
        **kwargs,
    ) -> None:
        block=SEResNetBottleneck
        super().__init__(
            layers=layers,
            groups=groups,
            reduction=reduction,
            dropout_prob=dropout_prob,
            inplanes=inplanes,
            downsample_kernel_size=downsample_kernel_size,
            input_3x3=input_3x3,
            **kwargs,
        )
        self.last_linear = nn.Linear(512 * block.expansion, 75)
        
    def logits(self, x: torch.Tensor):
        x = self.adaptive_avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        x = x.view(x.size(0), 25, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)
        return x
    
    
class improved_SEResNet101Custom(SEResNet101):
    """SEResNet101 based on `Squeeze-and-Excitation Networks` with optional pretrained support when spatial_dims is 2."""

    def __init__(
        self,
        layers: Sequence[int] = (3, 4, 23, 3),
        groups: int = 1,
        reduction: int = 16,
        dropout_prob: float | None = None,
        inplanes: int = 64,
        downsample_kernel_size: int = 1,
        input_3x3: bool = False,
        pretrained: bool = True,
        progress: bool = True,
        **kwargs,
    ) -> None:
        block = SEResNetBottleneck
        super().__init__(
            layers=layers,
            groups=groups,
            reduction=reduction,
            dropout_prob=dropout_prob,
            inplanes=inplanes,
            downsample_kernel_size=downsample_kernel_size,
            input_3x3=input_3x3,
            **kwargs,
        )
        self.feature_dim = 512 * block.expansion
        self.last_linear = nn.Linear(self.feature_dim, 75)

        # Additional layers for combining features
        self.conv1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(self.feature_dim, num_heads=8)
        self.final_conv = nn.Conv2d(self.feature_dim, 75, kernel_size=1)

    def logits(self, x: torch.Tensor):
        x = self.adaptive_avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        x = x.view(x.size(0), 25, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming input x has shape (1, 7, 356, 356)
        x = x.squeeze(0)  # Shape (7, 356, 356)
        feature_maps = []
        for i in range(x.size(0)):  # Process each slice individually
            slice_i = x[i].unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 356, 356)
            features_i = self.features(slice_i)  # Shape (1, feature_dim, H, W)
            feature_maps.append(features_i)
        
        # Concatenate feature maps along the channel dimension
        combined_features = torch.cat(feature_maps, dim=0)  # Shape (7, feature_dim, H, W)

        # Apply additional convolutions
        combined_features = self.conv1(combined_features)
        combined_features = nn.GELU()(combined_features)
        combined_features = self.conv2(combined_features)
        combined_features = nn.GELU()(combined_features)

        # Apply self-attention
        B, C, H, W = combined_features.size()
        combined_features_flat = combined_features.view(B, C, -1).permute(2, 0, 1)  # Shape (H*W, B, C)
        attn_output, _ = self.attention(combined_features_flat, combined_features_flat, combined_features_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # Shape (B, C, H, W)

        # Sum feature maps
        combined_features = combined_features + attn_output
        combined_features = combined_features.sum(dim=0, keepdim=True)  # Shape (1, feature_dim, H, W)

        # Apply final convolution
        logits = self.final_conv(combined_features)  # Shape (1, 75, H, W)

        logits = logits.mean(dim=[2, 3])  # Global average pooling
        logits = logits.view(logits.size(0), 25, 3)  # Reshape to (1, 25, 3)

        return logits


def load_net():
    if args.model_name == "improved_SEResNet101Custom":
        net = improved_SEResNet101Custom(
            in_channels=args.in_channels, 
            spatial_dims=args.spatial_dims, 
            layers=args.layers, 
            dropout_prob=args.dropout_prob, 
            inplanes=args.inplanes
        )

    return net