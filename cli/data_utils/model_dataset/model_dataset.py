from pathlib import PosixPath

from depth_estimation.utils.data import (
    InputTargetDataset,
    IntPILToTensor,
    FloatPILToTensor,
    MutualRandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms

import csv


def get_model_dataset(
    samples_idx_file, train=False, shuffle=False, device="cpu"
):

    # filenames
    rgb_depth_priors_tuples = []

    if isinstance(samples_idx_file, PosixPath):
        try:
            lines = csv.reader(open(samples_idx_file).read().splitlines())
            rgb_depth_priors_tuples += [i[1:] for i in lines]
        except FileNotFoundError:
            print(f"{samples_idx_file} not found, skipping...")
    
    elif isinstance(samples_idx_file, list):
        for file in samples_idx_file:
            try:
                lines = csv.reader(open(file).read().splitlines())
                rgb_depth_priors_tuples += [i[1:] for i in lines]
            except FileNotFoundError:
                print(f"{samples_idx_file} not found, skipping...")

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])
        target_samples_transform = transforms.Compose(
             [
                 MutualRandomFactor(factor_range=(0.8, 1.2)),
             ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [IntPILToTensor(type="uint8", device=device)]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = None
        target_samples_transform = None

    dataset = InputTargetDataset(
        rgb_depth_priors_tuples=rgb_depth_priors_tuples,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        target_samples_transform=target_samples_transform,
        max_priors=200,
        shuffle=shuffle,
    )

    return dataset
