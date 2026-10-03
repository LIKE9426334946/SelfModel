"""
简单的图像分割网络
没有激活函数，
"""

import torch
import torch.nn as nn


class Model01(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=2, padding=1
            ),  # 4x64x128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x128x64x64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4x256x32x32
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 4x512x16x16
            nn.MaxPool2d((2, 2), stride=2),  # 4x512x8x8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=2),  # 4x256x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=2),  # 4x128x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=2),  # 4x64x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=2),  # 4x32x128x128
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=(2, 2), stride=2
            ),  # 4x1x256x256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = Model01().eval()
    x = torch.randn(4, 3, 256, 256)
    model(x)
    torch.onnx.export(
        model,
        x,
        "onnx/model01.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
