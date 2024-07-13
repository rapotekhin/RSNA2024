from tap import Tap
import torch
import shutil, os

class SimpleArgumentParser(Tap):
    work_dir = "./workdir"
    task_name = "sample"
    project_name = "kaggle_rsna2024"

    # model config
    in_channels=1
    spatial_dims=2
    layers=(3, 4, 23, 3)
    dropout_prob=0.2
    inplanes=64
    model_name = "improved_SEResNet101Custom"
    checkpoint = None

    # data config
    modality = "Sagittal T2/STIR"
    test_size = 0.2
    random_state = 42
    batch_size = 1
    num_workers = 0
    image_size = (384, 384)
    resample_z_slices = 7

    # train config
    epochs = 10
    accumulation_steps = 4  # Number of batches to accumulate gradients
    label_smoothing_epsilon = 0.1
    lr = 0.001
    class_weights = [1.0, 2.0, 4.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = SimpleArgumentParser().parse_args()
args.workdir = f"{args.workdir}/{args.task_name}"

os.makedirs(args.workdir, exist_ok=True)
shutil.copy(
    "config.py", os.path.join(args.workdir, "config.py")
)
shutil.copy(
    "dataset.py", os.path.join(args.workdir, "dataset.py")
)
shutil.copy(
    "net.py", os.path.join(args.workdir, "net.py")
)
shutil.copy(
    "inference.py", os.path.join(args.workdir, "inference.py")
)