import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_loss_curve(total_losses, content_losses, style_losses, tv_losses, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    steps = list(range(1, len(total_losses) + 1))
    all_values = total_losses + content_losses + style_losses + tv_losses
    positive_values = [value for value in all_values if value > 0]
    epsilon = min(positive_values) * 0.5 if positive_values else 1e-8

    def log_safe(values):
        return [value if value > 0 else epsilon for value in values]

    plt.figure(figsize=(9, 6))
    plt.plot(steps, log_safe(total_losses), marker="o", markersize=3, label="Total Loss")
    plt.plot(steps, log_safe(content_losses), marker="o", markersize=3, label="Content Loss")
    plt.plot(steps, log_safe(style_losses), marker="o", markersize=3, label="Style Loss")
    plt.plot(steps, log_safe(tv_losses), marker="o", markersize=3, label="TV Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.yscale("log")
    if len(steps) == 1:
        plt.xlim(0.5, 1.5)
    else:
        plt.xlim(1, len(steps))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_comparison_image(content_path, style_paths, result_path, save_path):
    style_paths = [Path(p) for p in style_paths]
    panels = [("Content Image", Path(content_path))]
    panels.extend([(f"Style Image {i + 1}", style_path) for i, style_path in enumerate(style_paths)])
    panels.append(("Result Image", Path(result_path)))

    target_size = (300, 300)
    label_height = 48
    file_height = 40
    canvas_width = target_size[0] * len(panels)
    canvas_height = target_size[1] + label_height + file_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (title, path) in enumerate(panels):
        image = Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)
        x = index * target_size[0]
        canvas.paste(image, (x, label_height))
        draw.text((x + 12, 16), title, fill="black")
        draw.text((x + 12, label_height + target_size[1] + 12), path.name[:38], fill="black")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
