# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import monai.transforms
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import monai

import torch.nn.functional as F

from config import args
from utils import f_name_hash
import os



class CropBySpineMRI(monai.transforms.MapTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def crop_volume_by_black_border(self, vol, threshold=30):
        """
        Crops the input volume along the h axis based on the given threshold.
        
        Parameters:
        vol (np.ndarray): Input volume with shape (d, h, w)
        threshold (float): Threshold value for cropping (default is 50)
        
        Returns:
        np.ndarray: Cropped volume
        """

        def crop_by_h(vol, thr):
            # Compute the mean along the h axis
            means = np.mean(vol, axis=(0, 2))
            
            # Apply the threshold
            above_threshold = means > thr
            
            # Find the first and last indices where the condition is True
            start = np.argmax(above_threshold)
            end = len(above_threshold) - np.argmax(above_threshold[::-1])
            
            # Crop the volume
            cropped_vol = vol[:, start:end, :]
            
            return cropped_vol

        def crop_by_w(vol, thr):
            # Compute the mean along the h axis
            means = np.mean(vol, axis=(0, 1))
            
            # Apply the threshold
            above_threshold = means > thr
            
            # Find the first and last indices where the condition is True
            start = np.argmax(above_threshold)
            end = len(above_threshold) - np.argmax(above_threshold[::-1])
            
            # Crop the volume
            cropped_vol = vol[:, :, start:end]
            
            return cropped_vol

        vol = crop_by_h(vol, threshold)
        vol = crop_by_w(vol, threshold)
        return vol

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            dim = len(img.shape)

            if dim == 4:
                img = np.squeeze(img, axis=0)

            # Assuming img has shape (D, H, W)
            new_img = self.crop_volume_by_black_border(img)
            _, h, w = new_img.shape

            if key == "Sagittal T1":
                new_img = new_img[:, int(h * 0.3):int(h * 0.7), :]
            elif key == "Sagittal T2/STIR":
                new_img = new_img[:, int(h * 0.3):int(h * 0.7), :]
            elif key == "Axial T2":
                pass
            else:
                raise ValueError(f"{key} not modality")

            if dim == 4:
                new_img = np.expand_dims(new_img, axis=0)

            d[key] = new_img

        return d


class ExpandChannelFirstd(monai.transforms.MapTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]
            dim = len(img.shape)
            # Assuming img has shape (D, H, W)

            if dim == 3 and args.spatial_dims == 3:
                img = img.unsqueeze(0)

            d[key] = img

        return d
    

class ResampleZ(monai.transforms.MapTransform):
    def __init__(self, new_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_depth = new_depth

    def __call__(self, data):
        d = dict(data)

        if self.new_depth is None:
            return d

        for key in self.keys:
            img = d[key]
            dim = len(img.shape)
            # Assuming img has shape (D, H, W)
            if dim == 3:
                _, h, w = img.shape
                # Add channel dimension (C, D, H, W) where C=1
                img = img.unsqueeze(0)
            else:
                _, _, h, w = img.shape

            # Interpolate
            img = F.interpolate(img.unsqueeze(0), size=(self.new_depth, h, w), mode='trilinear', align_corners=False)

            # Remove added dimensions
            if dim == 3:
                img = img.squeeze(0).squeeze(0)

            d[key] = img

        return d


class RSNADataset(monai.data.Dataset):
    def __init__(self, df=None, train_series_descriptions=None, is_train=True, cache_dir=None, hash_func=None):
        self.is_train = is_train
        self.df = df
        self.cache_dir = cache_dir
        self.hash_func = hash_func
        
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
        self.data_dir = args.data_dir
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        series_descriptions = self.train_series_descriptions.loc[self.train_series_descriptions.study_id == study_id]  # series_id, series_description

        data = {}
        data["study_id"] = str(study_id)

        if self.cache_dir:
            fpath_hash = self.hash_func(data).decode("utf-8")
            hashfile = os.path.join(self.cache_dir, f"{fpath_hash}.pt")
            if os.path.isfile(hashfile):
                return data

        for series_id, series_description in series_descriptions[["series_id", "series_description"]].values:
            if series_description == args.modality:

                if self.is_train:
                    path_to_dicom_dir = f"{self.data_dir}/train_images/{study_id}/{series_id}"
                else:
                    path_to_dicom_dir = f"{self.data_dir}/test_images/{study_id}/{series_id}"
                # dicom = self.apply_transform(path_to_dicom_dir)
                # data[series_description] = dicom

                data[series_description] = path_to_dicom_dir
        
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

    def apply_transform(self, path_to_dicom_dir):
        return self.transform(path_to_dicom_dir)

    def get_transform(self):
        return monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=[args.modality]),
            monai.transforms.EnsureChannelFirstd(keys=[args.modality], channel_dim=-1),
            ResampleZ(keys=[args.modality], new_depth=args.resample_z_slices),
            CropBySpineMRI(keys=[args.modality]),

            monai.transforms.OneOf(
                transforms=[
                    monai.transforms.RandGaussianNoised(keys=[args.modality], mean=0, std=0.015, prob=0.25),
                    monai.transforms.RandGaussianSmoothd(
                        keys=[args.modality], sigma_x=(0.1, 0.3), sigma_y=(0.1, 0.3), sigma_z=(0.1, 0.3), prob=0.25
                    ),
                ]
            ),
            monai.transforms.OneOf(
                transforms=[
                    monai.transforms.RandGridDistortiond(
                        keys=[args.modality],
                        mode=["bicubic"],
                        num_cells=5,
                        prob=1,
                        distort_limit=(-0.05, 0.05),
                    ),
                    monai.transforms.RandGridDistortiond(
                        keys=[args.modality],
                        mode=["bicubic"],
                        num_cells=5,
                        prob=1,
                        distort_limit=(0, 0),
                    ),
                ],
                weights=[0.25, 0.75],
            ),
            # monai.transforms.RandCoarseDropoutd(keys=[args.modality], holes=4, spatial_size=8, prob=0.3),

            ExpandChannelFirstd(keys=[args.modality]),
            monai.transforms.Resized(keys=[args.modality], spatial_size=args.image_size, mode='trilinear'),
            monai.transforms.NormalizeIntensityd(keys=[args.modality]),
            monai.transforms.ToTensord(keys=[args.modality])
        ])

def get_dataset(df, train_series_descriptions, is_train=True):
    ds = RSNADataset(
        df=df, 
        train_series_descriptions=train_series_descriptions, 
        is_train=is_train, 
        cache_dir=args.cache_dir,
        hash_func=f_name_hash
    )

    if args.cache_dir is not None:
        return monai.data.PersistentDataset(
            data=ds,
            transform=ds.get_transform(),
            cache_dir=args.cache_dir,
            hash_func=f_name_hash,
        )
    else:
        return monai.data.Dataset(data=ds, transform=ds.get_transform())