import math
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter


def _gray_tensor(image_path, size=None, edge=False):
    image = Image.open(image_path).convert("L")
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    if edge:
        image = image.filter(ImageFilter.FIND_EDGES)
    return transforms.ToTensor()(image).view(-1)


def _ssim_from_tensors(tensor_a, tensor_b):
    mu_a = tensor_a.mean()
    mu_b = tensor_b.mean()
    var_a = tensor_a.var(unbiased=False)
    var_b = tensor_b.var(unbiased=False)
    cov_ab = ((tensor_a - mu_a) * (tensor_b - mu_b)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((numerator / denominator).item())


def calculate_ssim(image_a_path, image_b_path):
    image_a = Image.open(image_a_path).convert("L")
    tensor_a = transforms.ToTensor()(image_a).view(-1)
    tensor_b = _gray_tensor(image_b_path, size=image_a.size)
    return _ssim_from_tensors(tensor_a, tensor_b)


def calculate_edge_similarity(image_a_path, image_b_path):
    image_a = Image.open(image_a_path).convert("L")
    tensor_a = _gray_tensor(image_a_path, edge=True)
    tensor_b = _gray_tensor(image_b_path, size=image_a.size, edge=True)
    return _ssim_from_tensors(tensor_a, tensor_b)


def calculate_psnr(image_a_path, image_b_path):
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB").resize(image_a.size, Image.LANCZOS)
    tensor_a = transforms.ToTensor()(image_a)
    tensor_b = transforms.ToTensor()(image_b)
    mse = torch.mean((tensor_a - tensor_b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_color_similarity(image_a_path, image_b_path, bins=64):
    image_a = Image.open(image_a_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    image_b = Image.open(image_b_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    tensor_a = transforms.ToTensor()(image_a)
    tensor_b = transforms.ToTensor()(image_b)

    similarities = []
    for channel in range(3):
        hist_a = torch.histc(tensor_a[channel], bins=bins, min=0.0, max=1.0)
        hist_b = torch.histc(tensor_b[channel], bins=bins, min=0.0, max=1.0)
        hist_a = hist_a / hist_a.sum().clamp_min(1e-12)
        hist_b = hist_b / hist_b.sum().clamp_min(1e-12)
        intersection = torch.minimum(hist_a, hist_b).sum().item()
        similarities.append(intersection)

    return float(sum(similarities) / len(similarities))


def calculate_style_color_similarity(style_paths, result_path):
    scores = [calculate_color_similarity(style_path, result_path) for style_path in style_paths]
    return float(sum(scores) / len(scores)) if scores else 0.0


def format_metric(value, digits=6):
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def calculate_result_metrics(content_path, style_paths, result_path):
    content_path = Path(content_path)
    style_paths = [Path(p) for p in style_paths]
    result_path = Path(result_path)

    return {
        "ssim_content_result": calculate_ssim(content_path, result_path),
        "psnr_content_result": calculate_psnr(content_path, result_path),
        "color_similarity_content_result": calculate_color_similarity(content_path, result_path),
        "color_similarity_style_result": calculate_style_color_similarity(style_paths, result_path),
        "edge_similarity_content_result": calculate_edge_similarity(content_path, result_path),
    }
