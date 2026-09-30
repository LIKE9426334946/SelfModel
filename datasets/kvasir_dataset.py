from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode


# 定义数据集类，实现图片和mask的读取
class SelfDefineDataset(Dataset):
    def __init__(self, root, images, masks, image_size, in_channels=3):
        super().__init__()

        self.image_dir = Path(root) / images
        self.mask_dir = Path(root) / masks

        self.image_mode = {1: "L", 3: "RGB"}[in_channels]

        self.image_paths = sorted(
            self.image_dir.glob("*.jpg")
        )  # 注意，这里要使用通配符*来匹配所有文件，返回的是以image_dir开头的路径

        # 对图片进行预处理
        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        # 对标签进行处理 ?
        self.mask_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, interpolation=InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # mask文件名和图片是一一对应的
        mask_path = self.mask_dir / image_path.name

        with Image.open(image_path) as file:
            image = file.convert(self.image_mode)

        with Image.open(mask_path) as file:
            mask = file.convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask >= 0.5).float()

        return image, mask
