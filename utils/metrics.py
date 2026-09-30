import torch
import segmentation_models_pytorch as smp

from .task import predict_classes


@torch.no_grad()
def evaluate(model, loader, criterion, device, data_config):
    model.eval()
    mode = data_config["mode"]
    metric_config = data_config["metrics"]

    loss_sum = 0.0
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        loss_sum += loss.item() * images.size(0)

        predictions = predict_classes(logits, data_config)
        stats_kwargs = {"mode": mode}
        if mode == "multiclass":
            stats_kwargs["num_classes"] = data_config["num_classes"]

        stats = smp.metrics.get_stats(predictions, masks.long(), **stats_kwargs)
        if mode == "multiclass" and not metric_config["include_background"]:
            stats = tuple(value[:, 1:] for value in stats)

        tp, fp, fn, tn = stats

        tp_list.append(tp.cpu())
        fp_list.append(fp.cpu())
        fn_list.append(fn.cpu())
        tn_list.append(tn.cpu())

    tp = torch.cat(tp_list, dim=0)
    fp = torch.cat(fp_list, dim=0)
    fn = torch.cat(fn_list, dim=0)
    tn = torch.cat(tn_list, dim=0)

    options = {
        "reduction": metric_config["reduction"],
        "zero_division": metric_config["zero_division"],
    }
    return {
        "loss": loss_sum / len(loader.dataset),
        "iou": smp.metrics.iou_score(tp, fp, fn, tn, **options).item(),
        "f1": smp.metrics.f1_score(tp, fp, fn, tn, **options).item(),
        "precision": smp.metrics.precision(tp, fp, fn, tn, **options).item(),
        "recall": smp.metrics.recall(tp, fp, fn, tn, **options).item(),
    }
