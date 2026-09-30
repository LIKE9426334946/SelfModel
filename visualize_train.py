from datetime import datetime
from pathlib import Path

import torch
import yaml
from matplotlib import colormaps
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.metrics import evaluate
from utils.task import build_criterion, predict_classes


@torch.no_grad()
def log_images(writer, model, images, masks, device, data_config, palette, epoch):
    """展示固定验证样本；每张对比图依次为原图、真实标签、预测标签。"""
    model.eval()
    logits = model(images.to(device))
    predictions = predict_classes(logits, data_config).cpu()

    if data_config["mode"] == "binary":
        targets = masks[:, 0].long()
        predictions = predictions[:, 0]
        probabilities = logits.sigmoid().cpu()
        probability_names = ["foreground"]
    else:
        targets = masks.long()
        probabilities = logits.softmax(dim=1).cpu()
        probability_names = [f"class_{i}" for i in range(data_config["num_classes"])]

    # 灰度原图转成三通道，便于和标签的彩色显示拼在一起。
    originals = images.repeat(1, 3, 1, 1) if images.size(1) == 1 else images
    target_colors = palette[targets].permute(0, 3, 1, 2)
    prediction_colors = palette[predictions].permute(0, 3, 1, 2)

    for i in range(images.size(0)):
        comparison = torch.cat(
            [originals[i], target_colors[i], prediction_colors[i]], dim=2
        )
        writer.add_image(
            f"Images/sample_{i + 1}/Input_GT_Prediction", comparison, epoch
        )

    # 概率图越亮，代表模型认为该像素属于对应类别的概率越高。
    for i, name in enumerate(probability_names):
        writer.add_images(f"Probability/{name}", probabilities[:, i : i + 1], epoch)


def main():
    config = load_config()
    dataset_name = config["dataset"]
    data_config = config["datasets"][dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 从验证集均匀选择至多四张样本，之后每轮都观察相同的图片
    indices = torch.linspace(0, len(val_dataset) - 1, steps=min(4, len(val_dataset)))
    sample_indices = indices.long().tolist()
    sample_images, sample_masks = zip(*(val_dataset[i] for i in sample_indices))
    sample_images = torch.stack(sample_images)
    sample_masks = torch.stack(sample_masks)

    # 背景固定为黑色，二分类前景为白色，多分类使用不同颜色
    num_classes = data_config["num_classes"]
    cmap = colormaps["tab20"].resampled(num_classes)
    palette = torch.tensor(
        [cmap(i)[:3] for i in range(num_classes)], dtype=torch.float32
    )
    palette[0] = 0
    if data_config["mode"] == "binary":
        palette[1] = 1

    model = build_model(config["model"], data_config).to(device)
    criterion = build_criterion(data_config)
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # 每次运行创建独立目录，便于在 TensorBoard 中比较不同实验。
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = Path("runs") / dataset_name / config["model"]["name"] / run_name

    with SummaryWriter(log_dir=str(log_dir), flush_secs=10) as writer:
        config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        writer.add_text("Config", f"```yaml\n{config_text}\n```", 0)
        writer.add_text(
            "Image_guide",
            "Left to right: input | ground truth | prediction. "
            "Inputs use the dataset's resized [0, 1] images. "
            "Step 0 shows predictions before training. "
            f"Validation sample indices: {sample_indices}.",
            0,
        )

        log_images(
            writer, model, sample_images, sample_masks, device, data_config, palette, 0
        )
        writer.flush()
        global_step = 0

        for epoch in range(1, config["training"]["epochs"] + 1):
            print(f"开始 Epoch {epoch}/{config["training"]["epochs"]}")
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

                loss_value = loss.item()
                train_loss_sum += loss_value * images.size(0)
                global_step += 1
                writer.add_scalar("Loss/batch", loss_value, global_step)

            train_loss = train_loss_sum / len(train_dataset)
            val_metrics = evaluate(model, val_loader, criterion, device, data_config)

            # 同一张图中比较每轮训练 Loss 和验证 Loss。
            writer.add_scalars(
                "Loss/epoch", {"train": train_loss, "val": val_metrics["loss"]}, epoch
            )
            for name in ("iou", "f1", "precision", "recall"):
                writer.add_scalar(f"Metrics/val_{name}", val_metrics[name], epoch)
            writer.add_scalar(
                "Training/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

            log_images(
                writer,
                model,
                sample_images,
                sample_masks,
                device,
                data_config,
                palette,
                epoch,
            )
            writer.flush()


if __name__ == "__main__":
    main()

# 命令行执行代码
# tensorboard --logdir=runs
