# description
# 使用SMP库提供的UNet++网络结构，编码器为ResNet34，有预训练参数
#

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model07(nn.Module):
    def __init__(self, out_channels=1, in_channels=3):
        super().__init__()

        self.unet = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.unet(x)

if __name__ == "__main__":
    model = Model07(out_channels=1)
    model.eval()

    x = torch.randn(4, 3, 256, 256)
    model(x)
    torch.onnx.export(
        model, x, "onnx/model07.onnx", input_names=["x"], output_names=["output"]
    )