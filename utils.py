# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import pandas as pd
import sklearn.metrics
import hashlib
import tqdm

import warnings
warnings.filterwarnings("ignore")

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

def prepare_tensor(tensor):
    # Check if tensor has shape (25, 3), expand dims to (1, 25, 3)
    if tensor.shape == torch.Size([25, 3]):
        tensor = tensor.unsqueeze(0)  # Adds a batch dimension

    return tensor

def validate_model_with_submission_format(net, val_dataloader, val_df, labels):
    net.eval()  # Set the model to evaluation mode
    val_predictions = []
    val_targets = []

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm.tqdm(val_dataloader, desc="Val", total=len(val_dataloader)):
#             in_tensor = batch['Sagittal T2/STIR'].view(batch['Sagittal T2/STIR'].shape[1], 1, *batch['Sagittal T2/STIR'].shape[2:])
            in_tensor = batch[args.modality]
            target = batch['target']
            
            in_tensor = in_tensor.to(device)
            target = target.to(device).squeeze(0)  # Adjust target shape to match predict
            
            predict = net(in_tensor).softmax(dim=-1)
            # predict = torch.median(predict, dim=0).values  # Apply median after softmax

            predict = prepare_tensor(predict)
            target = prepare_tensor(target)

            val_predictions.append(predict)
            val_targets.append(target)

    # Concatenate all tensors in the lists along the batch dimension
    val_predictions = torch.cat(val_predictions, dim=0)  # Shape (N, 25, 3)
    val_targets = torch.cat(val_targets, dim=0)    # Shape (N, 25, 3)

    val_predictions = val_predictions.cpu().numpy()
    val_targets = val_targets.cpu().numpy()

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

def f_name_hash(data):
    fname = data["study_id"]
    return hashlib.md5(fname.encode("utf-8")).hexdigest().encode()