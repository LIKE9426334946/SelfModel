import csv

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from datasets import build_dataset
from utils.task import build_criterion
from utils.config import get_output_dir, load_config
from utils.metrics import evaluate
from models import build_model
from utils.plot import plot_history


def main():

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    config = load_config()
    dataset_name = config["dataset"]
    data_config = config["datasets"][dataset_name]
    print("使用的数据集为：", dataset_name)
    print("使用的数据集目录为：", data_config["root"])

    torch.manual_seed(config["training"]["seed"])

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))

    model = build_model(config["model"], data_config).to(device)
    print(f"使用的模型为{config['model']['name']}")

    criterion = build_criterion(data_config)
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("结果保存目录：", output_dir)

    best_val_loss = float("inf")  # 无限大
    best_val_iou = 0.0
    epochs = config["training"]["epochs"]

    history = []

    # 开始训练
    for epoch in range(epochs):
        print(f"开始 Epoch {epoch+1}/{epochs}", flush=True)

        model.train()
        train_loss_sum = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(
                0
            )  # 平均每个像素的损失x图片的数量
        train_loss = train_loss_sum / len(
            train_dataset
        )  # 最后再除以训练集图片的数量，得到平均损失

        val_metrics = evaluate(model, val_loader, criterion, device, data_config)
        val_loss = val_metrics["loss"]
        val_iou = val_metrics["iou"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}",
            flush=True,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print("已保存最佳模型", flush=True)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_metrics["iou"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        with open(
            output_dir / "metrics.csv", "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    plot_history(output_dir, title=f"{config['model']['name']} | {dataset_name}")


if __name__ == "__main__":
    main()
