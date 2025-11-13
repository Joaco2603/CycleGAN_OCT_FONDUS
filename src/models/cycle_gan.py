from __future__ import annotations

import torch

from .discriminator import PatchDiscriminator
from .generator import ResNetGenerator


class CycleGAN(torch.nn.Module):
    def __init__(self, channels: int = 3, features: int = 64, blocks: int = 9):
        super().__init__()
        self.gen_f2o = ResNetGenerator(channels, features, blocks)
        self.gen_o2f = ResNetGenerator(channels, features, blocks)
        self.disc_f = PatchDiscriminator(channels, features)
        self.disc_o = PatchDiscriminator(channels, features)

    def generators(self):
        return [self.gen_f2o, self.gen_o2f]

    def discriminators(self):
        return [self.disc_f, self.disc_o]

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        for module in self.generators() + self.discriminators():
            module.to(*args, **kwargs)
        return self
