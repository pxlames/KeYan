import torch
import torch.nn as nn

from .utils import init_weights


class DownConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, padding, pool=True):
        super().__init__()
        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, padding, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode="bilinear", scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        diff_y = bridge.shape[-2] - up.shape[-2]
        diff_x = bridge.shape[-1] - up.shape[-1]
        if diff_y != 0 or diff_x != 0:
            pad_left = max(diff_x // 2, 0)
            pad_right = max(diff_x - pad_left, 0)
            pad_top = max(diff_y // 2, 0)
            pad_bottom = max(diff_y - pad_top, 0)
            if pad_left or pad_right or pad_top or pad_bottom:
                up = nn.functional.pad(up, [pad_left, pad_right, pad_top, pad_bottom])

            if up.shape[-2] > bridge.shape[-2] or up.shape[-1] > bridge.shape[-1]:
                up = up[:, :, :bridge.shape[-2], :bridge.shape[-1]]

        out = torch.cat([up, bridge], dim=1)
        return self.conv_block(out)
