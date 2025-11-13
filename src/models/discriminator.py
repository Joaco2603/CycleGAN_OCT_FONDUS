import torch.nn as nn


def _block(in_channels: int, out_channels: int, normalize: bool) -> nn.Sequential:
    layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        layers = [_block(in_channels, features, normalize=False)]
        current = features
        for mult in [2, 4, 8]:
            layers.append(_block(current, features * mult, normalize=True))
            current = features * mult
        layers.append(nn.Conv2d(current, 1, 4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
