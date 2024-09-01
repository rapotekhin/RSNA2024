# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import torch

import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import tqdm

from config import args, device
from utils import label_smoothing, validate_model_with_submission_format

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Compute the BCE with logits loss
        bce_loss = self.bce_with_logits(inputs, targets)

        # Apply weights
        if self.alpha is not None:
            alpha = self.alpha.expand_as(targets)
            bce_loss = alpha * bce_loss

        # Get the probabilities for the true classes
        p_t = torch.exp(-bce_loss)

        # Compute the focal loss
        focal_loss = (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_net(net, train_dataloader, train, train_dataset, val_dataloader, val, val_dataset, clearml):
    print("START TRAINING")
    # Initialize the network, loss function, optimizer, and scheduler
    epochs = args.epochs
    accumulation_steps = args.accumulation_steps  # Number of batches to accumulate gradients
    label_smoothing_epsilon = args.label_smoothing_epsilon

    net.train()

    # Define class weights
    class_weights = torch.tensor(args.class_weights).to(device)

    # Initialize BCEWithLogitsLoss with class weights
    # lossbce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    lossbce = FocalLoss(alpha=class_weights)
    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)  # Example scheduler with T_max=epochs

    scaler = GradScaler()  # For mixed precision training

    # Trackers for loss and learning rate
    loss_tracker = []
    lr_tracker = []
    best_val_metric = 100

    running_loss = 0.0
    running_metric = 0.0

    for epoch in range(epochs):
        net.train()  # Ensure the model is in training mode

        optimizer.zero_grad()  # Zero the parameter gradients at the beginning of the epoch

        with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                in_tensor = batch[args.modality]
                target = batch['target']
                in_tensor = in_tensor.to(device)
                target = target.to(device).float()  # Ensure target is of type float
                
                # Apply label smoothing
                target = label_smoothing(target, epsilon=label_smoothing_epsilon)

                with autocast():  # Mixed precision training
                    logits = net(in_tensor)
                    predict = logits.softmax(dim=-1)  # Apply softmax for logits
                    
                    loss = lossbce(
                        logits.view(-1, logits.shape[-1]),
                        target.view(-1, target.shape[-1])
                    )
                
                scaler.scale(loss).backward()  # Backpropagation with scaler

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # Optimize with scaler
                    scaler.update()
                    optimizer.zero_grad()  # Zero the parameter gradients

                running_loss += loss.item()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    print_loss = running_loss / (epoch * len(train_dataloader) + batch_idx)
                    pbar.set_postfix({"Train Loss": f"{print_loss:.4f}"})
                    clearml.get_logger().report_scalar("Loss", lossbce.__class__.__name__, print_loss, epoch * len(train_dataloader) + batch_idx)

                pbar.update(1)

                
            # Validate the model
            val_metric = validate_model_with_submission_format(net, val_dataloader, val, val_dataset.data.labels)
            clearml.get_logger().report_scalar("Val LogLoss", "", val_metric, epoch)

        # Save the best model
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save({"state_dict": net.state_dict(), "val_metric": val_metric}, os.path.join(args.workdir, "model_best.pth"))

        # Average loss and metric for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_metric = running_metric / len(train_dataloader)
        loss_tracker.append(epoch_loss)
        lr_tracker.append(optimizer.param_groups[0]['lr'])

        # Step the scheduler
        scheduler.step()
        
        # Logging
        message = f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Val Metric: {val_metric:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        with open(os.path.join(args.workdir, "train_log.txt"), "a") as f:
            f.write(message + "\n")
        clearml.get_logger().report_scalar("Learning Rate", "", optimizer.param_groups[0]['lr'], epoch)
        print(message)

    # Optionally, plot the tracked loss and learning rates
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_tracker, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(lr_tracker, label='Learning Rate')
    # plt.xlabel('Epoch')
    # plt.ylabel('LR')
    # plt.legend()

    # plt.show()

    print(f"BEST VAL METRIC: {best_val_metric}")

    return net
