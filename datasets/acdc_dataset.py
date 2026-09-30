import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class SelfDefineDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        image_size=(256, 256),
        val_ratio=0.2,
        seed=23,
        train_dir="training",
        test_dir="testing",
        in_channels=1,
    ):
        super().__init__()

        root = Path(root)
        self.image_size = tuple(image_size)
        self.in_channels = in_channels

        if split == "test":
            self.patient_dirs = sorted((root / test_dir).glob("patient*"))
        else:
            patient_dirs = sorted((root / train_dir).glob("patient*"))
            random.Random(seed).shuffle(patient_dirs)
            val_count = int(len(patient_dirs) * val_ratio)
            self.patient_dirs = {
                "train": patient_dirs[val_count:],
                "val": patient_dirs[:val_count],
            }[split]
        self.samples = []
        for patient_dir in self.patient_dirs:
            for mask_path in sorted(patient_dir.glob("*_frame*_gt.nii.gz")):
                image_path = mask_path.with_name(
                    mask_path.name.replace("_gt.nii.gz", ".nii.gz")
                )

                # 三维图像形如(H,W,Z)，沿第三维提取切片
                num_slices = nib.load(image_path).shape[2]

                for z in range(num_slices):
                    self.samples.append((image_path, mask_path, z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, z = self.samples[index]

        image = np.array(nib.load(image_path).dataobj[:, :, z], dtype=np.float32)
        mask = np.array(nib.load(mask_path).dataobj[:, :, z], dtype=np.float32)
        image = (image - image.min()) / (
            image.max() - image.min() + 1e-6
        )  # 最小最大归一化

        image = torch.from_numpy(image)[None, None]
        mask = torch.from_numpy(mask)[None, None]

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]

        mask = F.interpolate(mask, size=self.image_size, mode="nearest-exact")[
            0, 0
        ].long()
        if self.in_channels == 3:
            image = image.repeat(3, 1, 1)

        return image, mask
