from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import get_output_dir, load_config
from utils.task import build_criterion, predict_classes
from utils.metrics import evaluate
from models import build_model
from datasets import build_dataset


@torch.no_grad()
def save_visualizations(
    model, loader, device, data_config, visualize_dir, dataset_name, model_name
):
    model.eval()
    binary = data_config["mode"] == "binary"
    num_classes = data_config["num_classes"]
    cmap = "gray" if binary else plt.get_cmap("tab10", num_classes)
    sample_index = 1

    for images, masks in loader:
        logits = model(images.to(device))
        predictions = predict_classes(logits, data_config).cpu()

        for i in range(1):  # 每次DataLoader保存一张图片
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            if images.size(1) == 1:  # 通道为1
                axes[0].imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                axes[0].imshow(images[i].permute(1, 2, 0).numpy())
            mask = masks[i, 0] if binary else masks[i]
            prediction = predictions[i, 0] if binary else predictions[i]

            axes[1].imshow(
                mask.numpy(),
                cmap=cmap,
                vmin=0,
                vmax=num_classes - 1,
                interpolation="nearest",
            )
            axes[2].imshow(
                prediction.numpy(),
                cmap=cmap,
                vmin=0,
                vmax=num_classes - 1,
                interpolation="nearest",
            )
            for ax, title in zip(axes, ("Image", "Ground Truth", "Prediction")):
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Sample {sample_index:06d}")
            fig.tight_layout()
            fig_save_dir = visualize_dir / dataset_name / model_name
            fig_save_dir.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                fig_save_dir / f"sample_{sample_index:06d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            sample_index += 1
    return sample_index - 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    dataset_name = config["dataset"]
    data_config = config["datasets"][dataset_name]
    model_name = config["model"]["name"]

    test_dataset = build_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    checkpoint_dir = get_output_dir(config)
    model = build_model(config["model"], data_config).to(device)

    state_dict = torch.load(
        checkpoint_dir / "best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 评估整个测试集
    test_output_dir = Path("test_outputs")
    visualize_dir = test_output_dir / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)

    print("使用的模型为：", model_name)
    print("使用的数据集为：", dataset_name)
    print("使用的数据集目录为：", data_config["root"])
    print("测试集样本数：", len(test_dataset))

    criterion = build_criterion(data_config)
    results = evaluate(model, test_loader, criterion, device, data_config)

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": len(test_dataset),
        **results,
    }
    csv_path = test_output_dir / "test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    print("测试指标已保存：", csv_path)
    count = save_visualizations(
        model, test_loader, device, data_config, visualize_dir, dataset_name, model_name
    )
    print(f"已保存{count}张组合图：{visualize_dir}")


if __name__ == "__main__":
    main()
