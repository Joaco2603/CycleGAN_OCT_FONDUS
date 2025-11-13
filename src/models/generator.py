import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, features: int = 64, num_blocks: int = 9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, 7, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        ]
        mult = 1
        for _ in range(2):
            layers.extend([
                nn.Conv2d(features * mult, features * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(features * mult * 2),
                nn.ReLU(inplace=True),
            ])
            mult *= 2
        for _ in range(num_blocks):
            layers.append(ResidualBlock(features * mult))
        for _ in range(2):
            layers.extend([
                nn.ConvTranspose2d(features * mult, features * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(features * mult // 2),
                nn.ReLU(inplace=True),
            ])
            mult //= 2
        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, in_channels, 7),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
