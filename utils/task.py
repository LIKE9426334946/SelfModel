import torch.nn as nn


def build_criterion(data_config):
    loss_classes = {
        "binary": nn.BCEWithLogitsLoss,
        "multiclass": nn.CrossEntropyLoss,
    }
    return loss_classes[data_config["mode"]]()  # 别忘了，这里有个小括号


def predict_classes(logits, data_config):
    """binary 返回[B,1,H,W], multiclass 返回[B,H,W]"""
    if data_config["mode"] == "binary":
        return (logits.sigmoid() >= data_config["threshold"]).long()
    return logits.argmax(dim=1)
