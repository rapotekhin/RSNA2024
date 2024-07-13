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
def format_val_to_submission(val, val_predictions, labels):
    submission = []

    for idx, row in val.iterrows():
        study_id = row['study_id']
        predictions = val_predictions[idx % len(val_predictions)]

        for i, label in enumerate(labels):
            row_id = f"{study_id}_{label}"
            normal_mild_value = float(predictions[i][0])
            moderate_value = float(predictions[i][1])
            severe_value = float(predictions[i][2])
            submission.append([row_id, normal_mild_value, moderate_value, severe_value])
    
    submission_df = pd.DataFrame(submission, columns=['row_id', 'normal_mild', 'moderate', 'severe'])
    return submission_df

def validate_model_with_submission_format(net, val_dataloader, val_df, labels):
    net.eval()  # Set the model to evaluation mode
    val_predictions = []
    val_targets = []

    with torch.no_grad():  # Disable gradient computation
        for batch in val_dataloader:
#             in_tensor = batch['Sagittal T2/STIR'].view(batch['Sagittal T2/STIR'].shape[1], 1, *batch['Sagittal T2/STIR'].shape[2:])
            in_tensor = batch['Sagittal T2/STIR']
            target = batch['target']
            
            in_tensor = in_tensor.to(device)
            target = target.to(device).squeeze(0)  # Adjust target shape to match predict
            
            predict = net(in_tensor).softmax(dim=-1)
            predict = torch.median(predict, dim=0).values  # Apply median after softmax
            
            val_predictions.append(predict.cpu().numpy())
            val_targets.append(target.cpu().numpy())
    
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)

    # Format the validation data to submission format
    formatted_submission = format_val_to_submission(val_df.reset_index(drop=True), val_predictions, labels)

    # Calculate log loss
    target_labels = []
    predict_labels = []
    sample_weights = []

    for i, row in formatted_submission.iterrows():
        row_id_parts = row['row_id'].split('_')
        study_id = row_id_parts[0]
        label = '_'.join(row_id_parts[1:])
        
        true_values = val_df[val_df['study_id'] == int(study_id)][label].values[0]
        true_class = ['Normal/Mild', 'Moderate', 'Severe'].index(true_values)
        target_labels.append(true_class)
        
        predict_values = row[['normal_mild', 'moderate', 'severe']].values
        predict_labels.append(predict_values)
        
        mapping = {0: 1, 1: 2, 2: 4}
        sample_weights.append(mapping[true_class])

    target_labels = np.array(target_labels)
    predict_labels = np.array(predict_labels)

    log_loss_value = sklearn.metrics.log_loss(
        y_true=target_labels,
        y_pred=predict_labels,
        labels=[0, 1, 2],
        sample_weight=sample_weights
    )

    return log_loss_value

def label_smoothing(target, epsilon=0.1):
    num_classes = target.size(-1)
    return (1 - epsilon) * target + epsilon / num_classes