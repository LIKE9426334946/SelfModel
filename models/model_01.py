# description
# 一个小型的编码器-解码器，用来测试
import torch
import torch.nn as nn


class Model01(nn.Module):
    def __init__(self, out_channels=1, in_channels=3):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            #
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = Model01(out_channels=1)
    model.eval()

    x = torch.randn(4, 3, 256, 256)
    model(x)
    torch.onnx.export(
        model, x, "onnx/model01.onnx", input_names=["x"], output_names=["output"]
    )
