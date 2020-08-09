# Python STL
import os
import logging
from typing import Any, Dict
# Image Processing
import numpy as np
import cv2
import random
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
# Data augmentation
from albumentations.augmentations import transforms as tf
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2


# Root folder of dataset
_DIRNAME: str = os.path.dirname(__file__)
DATA_FOLDER: str = os.path.join(_DIRNAME, "dataset", "raw")


class PanNukeDataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 phase: str,
                 args: Dict[str, Any],
                 num_classes: int = 2,
                 class_dict: Dict[int, int] = (0, 255),):
        """Create an API for the dataset

        Parameters
        ----------
        data_folder : str
            Root folder of dataset
        phase : str
            Phase of learning
            In ['train', 'val']
        num_classes : int
            Number of classes (including background)
        class_dict : dict[int, int]
            Dictionary mapping brightness to class indices
        args : Dict[str, Any]
            Extra arguments
            in_channels : int
                Number of input channels
            image_size : int
                Final size of output images
        """

        logger = logging.getLogger(__name__)
        logger.info(f"Creating {phase} dataset")

        # Root folder of the dataset
        if not os.path.isdir(data_folder):
            raise NotADirectoryError(f"{data_folder} is not a directory or "
                                     f"it doesn't exist.")
        logger.info(f"Datafolder: {data_folder}")
        self.root = data_folder

        # Phase of learning
        if phase not in ['train', 'val']:
            raise ValueError("Provide any one of ['train', 'val'] as phase.")
        logger.info(f"Phase: {phase}")
        self.phase = phase

        # Data Augmentations and tensor transformations
        self.transforms = PanNukeDataset.get_transforms(self.phase, args)
        self.images = np.load(os.path.join(self.root, self.phase, "images.npy"))
        self.masks = np.load(os.path.join(self.root, self.phase, "masks.npy"))

        # Number of classes in the segmentation target
        self.classes = np.load(os.path.join(self.root, self.phase, "types.npy"))
        self.num_classes = len(set(self.classes))
        logger.info(f"Number of classes: {self.num_classes}")

        # CLI args
        self.args = args

    def __getitem__(self, idx: int):
        image = self.images[idx]
        mask = self.masks[idx]

        # Data Augmentation for image and mask
        augmented = self.transforms['aug'](image=image, mask=mask)
        new_image = self.transforms['img_only'](image=augmented['image'])
        new_mask = self.transforms['mask_only'](image=augmented['mask'])
        aug_tensors = self.transforms['final'](image=new_image['image'],
                                               mask=new_mask['image'])
        image = aug_tensors['image']
        mask = aug_tensors['mask']

        # Add a channel dimension (C in [N C H W]) if required
        if self.num_classes == 2:
            mask = torch.unsqueeze(mask, dim=0)  # [H, W] => [H, W]

        # Return tuple of tensors
        return image, mask

    def __len__(self):
        return self.images.shape[0]

    @staticmethod
    def get_transforms(phase: str, args: Dict[str, Any]) -> Dict[str, Compose]:
        """Get composed albumentations augmentations

        Parameters
        ----------
        phase : str
            Phase of learning
            In ['train', 'val']
        args : Dict[str, Any]
            Extra arguments

        Returns
        -------
        transforms: dict[str, albumentations.core.composition.Compose]
            Composed list of transforms
        """
        aug_transforms = []
        im_sz = (args['image_size'], args['image_size'])

        if phase == "train":
            # Data augmentation for training only
            aug_transforms.extend([
                tf.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5),
                tf.Flip(p=0.5),
                tf.RandomRotate90(p=0.5),
            ])
            # Exotic Augmentations for train only 🤤
            aug_transforms.extend([
                tf.RandomBrightnessContrast(p=0.5),
                tf.ElasticTransform(p=0.5),
                tf.MultiplicativeNoise(multiplier=(0.5, 1.5),
                                       per_channel=True, p=0.2),
            ])
        aug_transforms.extend([
            tf.RandomSizedCrop(min_max_height=im_sz,
                               height=im_sz[0],
                               width=im_sz[1],
                               w2h_ratio=1.0,
                               interpolation=cv2.INTER_LINEAR,
                               p=1.0),
        ])
        aug_transforms = Compose(aug_transforms)

        mask_only_transforms = Compose([
            tf.Normalize(mean=0, std=1, always_apply=True)
        ])
        image_only_transforms = Compose([
            tf.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                         always_apply=True)
        ])
        final_transforms = Compose([
            ToTensorV2()
        ])

        transforms = {
            'aug': aug_transforms,
            'img_only': image_only_transforms,
            'mask_only': mask_only_transforms,
            'final': final_transforms
        }
        return transforms


def provider(data_folder: str,
             phase: str,
             args: Dict[str, Any],
             batch_size: int = 8,
             num_workers: int = 4, ) -> DataLoader:
    """Return dataloader for the dataset

    Parameters
    ----------
    data_folder : str
        Root folder of the dataset
    phase : str
        Phase of learning
        In ['train', 'val']
    batch_size : int
        Batch size
    num_workers : int
        Number of workers
    args : Dict[str, Any]
        Extra arguments

    Returns
    -------
    dataloader: DataLoader
        DataLoader for loading data from CPU to GPU
    """
    image_dataset = PanNukeDataset(data_folder, phase, args)
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {phase} dataloader")
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
