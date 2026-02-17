import cv2
import numpy as np
import torch

# reference: https://github.com/facebookresearch/dinov3
GND_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
GND_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
AER_IMAGE_MEAN = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1)
AER_IMAGE_STD = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1)


def read_image_tensor(path: str, resize: int | tuple[int, int] | None = None, device: str = "cpu") -> torch.Tensor:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")

    if resize is not None:
        # resize: (int) -> (H, W) or (H, W) -> (H, W)
        resize = (resize, resize) if isinstance(resize, int) else resize
        image = cv2.resize(image, resize[::-1], interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().mul_(255).byte()

    if image.ndim == 3:
        image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    elif image.ndim == 4:
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

    return image.numpy()


def image_to_tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    if image.ndim == 3:
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    elif image.ndim == 4:
        image = image.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

    return torch.from_numpy(image).to(device, torch.float32).div_(255)


def normalize_image_tensor(x: torch.Tensor, image_type: str = "ground") -> torch.Tensor:
    if image_type == "ground":
        return (x - GND_IMAGE_MEAN.to(x.device, x.dtype)) / GND_IMAGE_STD.to(x.device, x.dtype)
    elif image_type == "aerial":
        return (x - AER_IMAGE_MEAN.to(x.device, x.dtype)) / AER_IMAGE_STD.to(x.device, x.dtype)
    else:
        raise ValueError(f"Unknown image type: {image_type}")


def denormalize_image_tensor(x: torch.Tensor, image_type: str = "ground") -> torch.Tensor:
    if image_type == "ground":
        return (x * GND_IMAGE_STD.to(x.device, x.dtype)) + GND_IMAGE_MEAN.to(x.device, x.dtype)
    elif image_type == "aerial":
        return (x * AER_IMAGE_STD.to(x.device, x.dtype)) + AER_IMAGE_MEAN.to(x.device, x.dtype)
    else:
        raise ValueError(f"Unknown image type: {image_type}")
