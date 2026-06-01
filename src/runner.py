import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from .constants import (
    CONTENT_LAYERS,
    STYLE_LAYER_WEIGHTS,
    STYLE_LAYERS,
)
from .io_utils import get_image_list, load_image, save_tensor_image
from .losses import gram_matrix, total_variation_loss
from .metrics import calculate_result_metrics, format_metric
from .parsing import parse_blend_weights, parse_csv_floats, parse_csv_ints, parse_csv_paths
from .reports import append_summary_csv
from .visualization import create_comparison_image, plot_loss_curve


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def style_signature(style_paths):
    return "+".join(Path(p).stem for p in style_paths)


def make_experiment_name(content_path, style_paths, args):
    name = (
        f"{Path(content_path).stem}__{style_signature(style_paths)}"
        f"__size{args.image_size}_steps{args.num_steps}"
        f"_cw{args.content_weight:g}_sw{args.style_weight:g}"
    )
    if args.mask is not None:
        name += "__local"
    return name


def get_style_grams(style_paths, style_weights, extractor, image_size, device):
    accumulated = {layer: None for layer in STYLE_LAYERS}

    for style_path, blend_weight in zip(style_paths, style_weights):
        style_image, _ = load_image(style_path, image_size, device)
        style_features = extractor(style_image)

        for layer in STYLE_LAYERS:
            gram = gram_matrix(style_features[layer]) * blend_weight
            accumulated[layer] = gram if accumulated[layer] is None else accumulated[layer] + gram

    return accumulated


def run_one_experiment(content_path, style_paths, args, extractor, summary_path, device=None):
    device = device or get_device()
    content_path = Path(content_path)
    style_paths = [Path(p) for p in style_paths]

    if not content_path.exists():
        raise FileNotFoundError(f"Content image not found: {content_path}")
    for style_path in style_paths:
        if not style_path.exists():
            raise FileNotFoundError(f"Style image not found: {style_path}")

    style_weights = parse_blend_weights(args.style_blend_weights, len(style_paths))
    experiment_name = make_experiment_name(content_path, style_paths, args)
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    result_path = experiment_dir / "result.jpg"
    loss_curve_path = experiment_dir / "loss_curve.png"
    comparison_path = experiment_dir / "comparison.jpg"

    print("=" * 90)
    print(f"Experiment: {experiment_name}")
    print(f"Content: {content_path}")
    print(f"Styles: {', '.join(str(p) for p in style_paths)}")
    if args.mask is not None:
        print(f"Local transfer mask: {args.mask}")
    print(f"Output: {experiment_dir}")
    print("=" * 90)

    start_time = time.time()

    content_image, original_size = load_image(content_path, args.image_size, device)
    content_features = extractor(content_image)
    style_grams = get_style_grams(style_paths, style_weights, extractor, args.image_size, device)

    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_image], lr=args.learning_rate)

    total_losses = []
    content_losses = []
    style_losses = []
    tv_losses = []

    for step in tqdm(range(1, args.num_steps + 1)):
        optimizer.zero_grad()
        generated_features = extractor(generated_image)

        content_loss = 0.0
        for layer in CONTENT_LAYERS:
            content_loss += torch.mean((generated_features[layer] - content_features[layer]) ** 2)

        style_loss = 0.0
        for layer in STYLE_LAYERS:
            generated_gram = gram_matrix(generated_features[layer])
            layer_style_loss = torch.mean((generated_gram - style_grams[layer]) ** 2)
            style_loss += STYLE_LAYER_WEIGHTS[layer] * layer_style_loss

        tv_loss = total_variation_loss(generated_image)
        total_loss = (
            args.content_weight * content_loss
            + args.style_weight * style_loss
            + args.tv_weight * tv_loss
        )

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated_image.clamp_(-3, 3)

        total_losses.append(total_loss.item())
        content_losses.append(content_loss.item())
        style_losses.append(style_loss.item())
        tv_losses.append(tv_loss.item())

        if args.save_every > 0 and step % args.save_every == 0:
            save_tensor_image(
                generated_image,
                experiment_dir / f"result_step_{step}.jpg",
                output_size=original_size,
                mask_path=args.mask,
                content_path=content_path,
            )

        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"Step [{step}/{args.num_steps}] "
                f"Content={content_loss.item():.6f} "
                f"Style={style_loss.item():.6f} "
                f"TV={tv_loss.item():.6f} "
                f"Total={total_loss.item():.6f}"
            )

    save_tensor_image(
        generated_image,
        result_path,
        output_size=original_size,
        mask_path=args.mask,
        content_path=content_path,
    )
    plot_loss_curve(total_losses, content_losses, style_losses, tv_losses, loss_curve_path)
    create_comparison_image(content_path, style_paths, result_path, comparison_path)

    runtime_seconds = time.time() - start_time
    metrics = calculate_result_metrics(content_path, style_paths, result_path)

    append_summary_csv(summary_path, [
        experiment_name,
        args.mode,
        content_path.name,
        ";".join(p.name for p in style_paths),
        str(device),
        args.image_size,
        args.num_steps,
        args.content_weight,
        args.style_weight,
        args.tv_weight,
        ";".join(f"{w:.4f}" for w in style_weights),
        str(args.mask) if args.mask is not None else "",
        f"{content_losses[-1]:.8f}",
        f"{style_losses[-1]:.8f}",
        f"{tv_losses[-1]:.8f}",
        f"{total_losses[-1]:.8f}",
        f"{runtime_seconds:.2f}",
        format_metric(metrics["ssim_content_result"]),
        format_metric(metrics["psnr_content_result"]),
        format_metric(metrics["color_similarity_content_result"]),
        format_metric(metrics["color_similarity_style_result"]),
        format_metric(metrics["edge_similarity_content_result"]),
        str(result_path),
        str(loss_curve_path),
        str(comparison_path),
    ])

    print("Experiment finished")
    print(f"Result: {result_path}")
    print(f"Loss curve: {loss_curve_path}")
    print(f"Comparison: {comparison_path}")
    print(
        f"Runtime: {runtime_seconds:.2f}s, "
        f"SSIM: {metrics['ssim_content_result']:.6f}, "
        f"PSNR: {metrics['psnr_content_result']:.4f}"
    )
    return {
        "experiment_name": experiment_name,
        "result_path": result_path,
        "loss_curve_path": loss_curve_path,
        "comparison_path": comparison_path,
        "metrics": metrics,
    }


def build_experiment_specs(args):
    specs = []

    if args.mode == "single":
        if args.content is None or args.style is None:
            raise ValueError("single mode requires --content and --style")
        specs.append((Path(args.content), [Path(args.style)]))

    elif args.mode == "same_content_diff_styles":
        if args.fixed_content is None:
            raise ValueError("same_content_diff_styles requires --fixed_content")
        for style_path in get_image_list(args.style_dir):
            specs.append((Path(args.fixed_content), [style_path]))

    elif args.mode == "same_style_diff_contents":
        if args.fixed_style is None:
            raise ValueError("same_style_diff_contents requires --fixed_style")
        for content_path in get_image_list(args.content_dir):
            specs.append((content_path, [Path(args.fixed_style)]))

    elif args.mode == "all":
        for content_path in get_image_list(args.content_dir):
            for style_path in get_image_list(args.style_dir):
                specs.append((content_path, [style_path]))

    elif args.mode == "multi_style":
        if args.content is None or args.styles is None:
            raise ValueError("multi_style mode requires --content and --styles")
        specs.append((Path(args.content), parse_csv_paths(args.styles)))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return specs


def expand_parameter_sweep(args):
    content_weights = parse_csv_floats(args.content_weights) if args.content_weights else [args.content_weight]
    style_weights = parse_csv_floats(args.style_weights) if args.style_weights else [args.style_weight]
    num_steps_list = parse_csv_ints(args.num_steps_list) if args.num_steps_list else [args.num_steps]

    variants = []
    for content_weight in content_weights:
        for style_weight in style_weights:
            for num_steps in num_steps_list:
                variant = deepcopy(args)
                variant.content_weight = content_weight
                variant.style_weight = style_weight
                variant.num_steps = num_steps
                variants.append(variant)

    return variants
