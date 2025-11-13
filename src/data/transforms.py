from torchvision import transforms as T

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


def build_transforms(image_size: int, train: bool = True, augment: bool = True) -> T.Compose:
    ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
    if train and augment:
        ops.extend([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.05)])
    ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
    return T.Compose(ops)
