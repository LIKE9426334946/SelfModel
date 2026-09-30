import torch
from torch.utils.data import random_split


def build_dataset(config, split):
    dataset_name = config["dataset"]
    data_config = config["datasets"][dataset_name]
    seed = config["training"]["seed"]

    if dataset_name == "kvasir-seg":
        from .kvasir_dataset import SelfDefineDataset

        dataset = SelfDefineDataset(
            root=data_config["root"],
            images=data_config["images"],
            masks=data_config["masks"],
            image_size=data_config["image_size"],
            in_channels=data_config["in_channels"],
        )
        subsets = random_split(
            dataset,
            data_config["split_ratio"],
            generator=torch.Generator().manual_seed(seed),
        )
        return dict(zip(("train", "val", "test"), subsets))[split]

    if dataset_name == "acdc":
        from .acdc_dataset import SelfDefineDataset

        return SelfDefineDataset(
            root=data_config["root"],
            split=split,
            image_size=data_config["image_size"],
            val_ratio=data_config["val_ratio"],
            seed=seed,
            train_dir=data_config["train_dir"],
            test_dir=data_config["test_dir"],
            in_channels=data_config["in_channels"],
        )

    raise ValueError(f"不支持的数据集：{dataset_name}")
