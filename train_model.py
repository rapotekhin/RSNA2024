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
    lossbce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)  # Example scheduler with T_max=epochs

    scaler = GradScaler()  # For mixed precision training

    # Trackers for loss and learning rate
    loss_tracker = []
    lr_tracker = []
    best_val_metric = 100

    for epoch in tqdm.trange(epochs, desc="Epochs"):
        net.train()  # Ensure the model is in training mode
        running_loss = 0.0
        running_metric = 0.0
        
        optimizer.zero_grad()  # Zero the parameter gradients at the beginning of the epoch

        for batch_idx, batch in enumerate(train_dataloader):
    #         in_tensor = batch[args.modality].view(batch[args.modality].shape[1], 1, *batch[args.modality].shape[2:])
            in_tensor = batch[args.modality]
    #         target = batch['target'].repeat(in_tensor.shape[0], 1, 1)  # Repeat target to match input batch siz

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
            
            if (batch_idx + 1) % (len(train_dataloader) // 5) == 0:
                print_loss = running_loss / (epoch * len(train_dataloader) + batch_idx)
                message = f"Epoch {epoch}, Train  Loss: {print_loss:.4f}"
                print(message, flush=True)

                clearml.get_logger().report_scalar("Loss", "BCE", print_loss, epoch * len(train_dataloader) + batch_idx)

            
        # Validate the model
        val_metric = validate_model_with_submission_format(net, val_dataloader, val, val_dataset.labels)
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
