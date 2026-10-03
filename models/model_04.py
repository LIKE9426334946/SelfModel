"""
自己写的UNet网络+ResNet18编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def residual(in_channels, out_channels, kernels, stride, pad):
    return nn.Conv2d(in_channels, out_channels, kernels, stride, pad)


def block1(in_channels, out_channels, kernels, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernels, stride=stride, padding=pad),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernels, stride=stride, padding=pad),
    )


def block2(in_channels, out_channels, kernels, stride=2, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernels, stride=stride, padding=pad),
        nn.ReLU(),
        nn.Conv2d(
            out_channels, out_channels, kernels, stride=int(stride / 2), padding=pad
        ),
    )


def upsample(in_channels, out_channels, kernels=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernels, stride, pad),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernels, stride, pad),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
    )


def final(in_channels, out_channels, kernels=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernels, stride, pad),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernels, stride, pad),
        nn.ReLU(),
    )


class Model03(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, 3)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x10 = self.max_pool(x1)
        x11 = block1(64, 64, 3)(x10)
        x12 = self.relu(x11 + x10)
        x13 = block1(64, 64, 3)(x12)
        #
        x2 = self.relu(x12 + x13)
        x20 = residual(64, 128, 1, stride=2, pad=0)(x2)
        x21 = block2(64, 128, 3, stride=2, pad=1)(x2)
        x22 = self.relu(x20 + x21)
        x23 = block1(128, 128, 3)(x22)
        #
        x3 = self.relu(x22 + x23)
        x30 = residual(128, 256, 1, stride=2, pad=0)(x3)
        x31 = block2(128, 256, 3, stride=2, pad=1)(x3)
        x32 = self.relu(x30 + x31)
        x33 = block1(256, 256, 3)(x32)
        #
        x4 = self.relu(x32 + x33)
        x40 = residual(256, 512, 1, stride=2, pad=0)(x4)
        x41 = block2(256, 512, 3, stride=2, pad=1)(x4)
        x42 = self.relu(x40 + x41)
        x43 = block1(512, 512, 3)(x42)
        #
        x5 = self.relu(x42 + x43)
        #
        i4 = nn.Upsample(scale_factor=2, mode="nearest")(x5)
        up4 = torch.cat([x4, i4], dim=1)
        #
        i3 = upsample(768, 256, 3)(up4)
        up3 = torch.cat([x3, i3], dim=1)
        #
        i2 = upsample(384, 128, 3)(up3)
        up2 = torch.cat([x2, i2], dim=1)
        #
        i1 = upsample(192, 64, 3)(up2)
        up1 = torch.cat([x1, i1], dim=1)
        #
        i0 = upsample(128, 32)(up1)

        y = final(32, 16, 3)(i0)
        return self.conv2(y)


if __name__ == "__main__":
    model = Model03().eval()

    input = torch.rand((16, 3, 256, 256))
    output = model(input)

    torch.onnx.export(
        model, (input,), "onnx/self-unet1.onnx", input_names=["input"], dynamo=True
    )
