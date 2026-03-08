import torch.nn as nn

from .blocks import DownConvBlock, UpConvBlock


class Unet(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, apply_last_layer=True, padding=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.apply_last_layer = apply_last_layer

        self.contracting_path = nn.ModuleList()
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]
            self.contracting_path.append(
                DownConvBlock(input_dim, output_dim, self.padding, pool=(i != 0))
            )

        self.upsampling_path = nn.ModuleList()
        for i in range(len(self.num_filters) - 2, -1, -1):
            input_dim = output_dim + self.num_filters[i]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, self.padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output_dim, num_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        if self.apply_last_layer:
            x = self.last_layer(x)
        return x

