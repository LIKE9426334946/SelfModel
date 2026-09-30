# description
# 使用SMP库提供的UNet网络结构，编码器为ResNet18，无预训练参数
#

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model02(nn.Module):
    def __init__(self, out_channels=1, in_channels=3):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.unet(x)


if __name__ == "__main__":
    model = Model02(out_channels=1)
    model.eval()

    x = torch.randn(4, 3, 256, 256)
    model(x)
    torch.onnx.export(
        model, x, "onnx/model02.onnx", input_names=["x"], output_names=["output"]
    )
