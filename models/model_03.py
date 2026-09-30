# description
# 一个简单Transformer编码器+上采样分割解码器的网络

import torch
import torch.nn as nn


class Model03(nn.Module):
    def __init__(self, out_channels=1, in_channels=3, image_size=(256, 256)):
        super().__init__()

        patch_size = 16
        d_model = 256
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.path_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))  # ?
        nn.init.normal_(self.pos_embed, std=0.02)  # ?

        # 一个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4, norm=nn.LayerNorm(d_model)
        )
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, kernel_size=2, stride=2),
            nn.GELU(),
            #
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            #
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.GELU(),
            #
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.GELU(),
            #
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.path_embed(x)
        batch_size, channels, height, width = x.shape

        #
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed

        x = self.encoder(x)

        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        logits = self.decoder(x)
        return logits


if __name__ == "__main__":
    model = Model03(out_channels=1)
    model.eval()

    x = torch.randn(4, 3, 256, 256)
    model(x)
    torch.onnx.export(
        model, x, "onnx/model03.onnx", input_names=["x"], output_names=["output"]
    )
