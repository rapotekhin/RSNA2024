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
from dataset import RSNADataset
from net import load_net
from train_model import train_net
from utils import label_smoothing, validate_model_with_submission_format

# Validation function
def inference_model(net, test_dataloader, ss, labels):
    
    new_ss = pd.DataFrame(columns=ss.columns)

    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch in test_dataloader:
            in_tensor = batch[args.modality].view(batch[args.modality].shape[1], 1, *batch[args.modality].shape[2:])
            study_id = int(batch['study_id'][0])

            predict = net(in_tensor).median(dim=0).values.softmax(dim=-1)
            
            for i in range(25):
                label = labels[i]
                row_id = f"{study_id}_{label}"
                normal_mild_value = float(predict[i][0])
                moderate_value = float(predict[i][1])
                severe_value = float(predict[i][2])
                
                data = [[row_id, normal_mild_value, moderate_value, severe_value]]
                data_df = pd.DataFrame(data, columns=ss.columns)
                
                # Concatenate the new DataFrame to new_ss
                new_ss = pd.concat([new_ss, data_df], axis=0, ignore_index=True)

    return new_ss

if __name__ == "__main__":
    net = load_net()
    net = net.to(device)
    if device == 'cuda':
        net = torch.compile(net)
        net(torch.randn(1, 224, 224).to(device))
    net.load_state_dict(torch.load(os.path.join(args.workdir, "model_best.pth"), map_location="cpu")["state_dict"])

    test_series_descriptions = pd.read_csv("test_series_descriptions.csv")
    ss = pd.read_csv("sample_submission.csv")

    test_dataset = RSNADataset(df=None, train_series_descriptions=test_series_descriptions, is_train=False)
    test_dataloader = monai.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    new_ss = inference_model(net, test_dataloader, ss, test_dataset.labels)
    new_ss.to_csv("submission.csv", index=False)