from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from .constants import VALID_EXTENSIONS


def get_image_list(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    image_files = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    image_files.sort()

    if not image_files:
        raise FileNotFoundError(f"No images found in: {folder_path}")

    return image_files


def load_image(image_path, image_size, device):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device), original_size


def denormalize_tensor(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(image * std + mean, 0, 1)


def blend_with_content_by_mask(result_image, content_path, mask_path, output_size):
    content = Image.open(content_path).convert("RGB")
    if output_size is not None:
        content = content.resize(output_size, Image.LANCZOS)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(result_image.size, Image.LANCZOS)
    return Image.composite(result_image, content, mask)


def save_tensor_image(tensor, save_path, output_size=None, mask_path=None, content_path=None):
    image = transforms.ToPILImage()(denormalize_tensor(tensor))

    if output_size is not None:
        image = image.resize(output_size, Image.LANCZOS)

    if mask_path is not None and content_path is not None:
        image = blend_with_content_by_mask(image, content_path, mask_path, output_size)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def tensor_to_pil(tensor, output_size=None):
    image = transforms.ToPILImage()(denormalize_tensor(tensor))
    if output_size is not None:
        image = image.resize(output_size, Image.LANCZOS)
    return image
