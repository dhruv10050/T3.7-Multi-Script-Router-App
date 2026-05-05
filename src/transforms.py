from torchvision import transforms

from config import IMG_SIZE, NORM_MEAN, NORM_STD


def get_transforms(mode: str, img_size: int = IMG_SIZE) -> transforms.Compose:
    # PIL-based geometric augmentations must precede ToTensor because
    # torchvision's spatial transforms operate on PIL images, not tensors.
    aug_pil = [
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
    ] if mode == "train" else []

    # RandomErasing operates on tensors and must therefore follow ToTensor.
    aug_tensor = [
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ] if mode == "train" else []

    return transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.Resize((img_size, img_size))]
        + aug_pil
        + [transforms.ToTensor(),
           transforms.Normalize(NORM_MEAN, NORM_STD)]
        + aug_tensor
    )
