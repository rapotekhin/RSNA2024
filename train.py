# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


import monai
from sklearn.model_selection import train_test_split

from clearml import Task

from config import args, device
from dataset import RSNADataset
from net import load_net
from train_model import train_net
from utils import validate_model_with_submission_format

if __name__ == "__main__":
    clearml = Task.init(
        project_name=args.project_name, 
        task_name=args.task_name, 
        auto_connect_frameworks={"pytorch": False},
        task_type=Task.TaskTypes.training
    )

    train = pd.read_csv("train.csv")
    train_label_coordinates = pd.read_csv("train_label_coordinates.csv")
    train_series_descriptions = pd.read_csv("train_series_descriptions.csv")

    train = train.dropna()

    train, val = train_test_split(train, test_size=args.test_size, random_state=args.random_state)

    train_dataset = RSNADataset(df=train, train_series_descriptions=train_series_descriptions)
    val_dataset = RSNADataset(df=val, train_series_descriptions=train_series_descriptions)

    train_dataloader = monai.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader = monai.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    net = load_net()
    # net = torch.nn.DataParallel(net)
    if os.path.isfile(args.checkpoint):
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"], strict=False)
    net = net.to(device)

    if device == 'cuda':
        net = torch.compile(net)
        net(torch.randn(1, 224, 224).to(device))

    net = train_net(net, train_dataloader, train, train_dataset, val_dataloader, val, val_dataset, clearml)

    net.load_state_dict(torch.load(os.path.join(args.workdir, "model_best.pth"), map_location="cpu")["state_dict"])
    net = net.to(device)

    val_metric = validate_model_with_submission_format(net, val_dataloader, val, val_dataset.labels)

    with open(os.path.join(args.workdir, "metrics.txt"), 'w') as f:
        f.write(f"VAL LOGLOS: {val_metric}")
    # test_series_descriptions = pd.read_csv("test_series_descriptions.csv")
    # ss = pd.read_csv("sample_submission.csv")

    # test_dataset = RSNADataset(df=None, train_series_descriptions=test_series_descriptions, is_train=False)
    # test_dataloader = monai.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    # new_ss = inference_model(net, test_dataloader, ss, test_dataset.labels)
    # new_ss.to_csv("submission.csv", index=False)