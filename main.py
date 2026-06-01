import argparse
from pathlib import Path

from src.config_runner import load_config_runs
from src.constants import (
    DEFAULT_CONTENT_DIR,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STYLE_DIR,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_TV_WEIGHT,
)
from src.model import VGGFeatureExtractor
from src.reports import ensure_summary_csv, generate_html_report, write_system_report
from src.runner import build_experiment_specs, expand_parameter_sweep, get_device, run_one_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Neural Style Transfer with VGG19")

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "same_content_diff_styles", "same_style_diff_contents", "all", "multi_style"],
        help="Experiment mode",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file for multiple experiments")
    parser.add_argument("--content", type=str, default=None, help="Content image for single/multi_style mode")
    parser.add_argument("--style", type=str, default=None, help="Style image for single mode")
    parser.add_argument("--styles", type=str, default=None, help="Comma-separated style images for multi_style mode")
    parser.add_argument("--style_blend_weights", type=str, default=None, help="Comma-separated fusion weights")

    parser.add_argument("--fixed_content", type=str, default=None, help="Fixed content image for all styles")
    parser.add_argument("--fixed_style", type=str, default=None, help="Fixed style image for all contents")
    parser.add_argument("--content_dir", type=str, default=DEFAULT_CONTENT_DIR, help="Content image folder")
    parser.add_argument("--style_dir", type=str, default=DEFAULT_STYLE_DIR, help="Style image folder")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--mask", type=str, default=None, help="Optional grayscale mask for local transfer")

    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Training image short side")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS, help="Optimization steps")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--content_weight", type=float, default=DEFAULT_CONTENT_WEIGHT, help="Content loss weight")
    parser.add_argument("--style_weight", type=float, default=DEFAULT_STYLE_WEIGHT, help="Style loss weight")
    parser.add_argument("--tv_weight", type=float, default=DEFAULT_TV_WEIGHT, help="Total variation loss weight")

    parser.add_argument("--content_weights", type=str, default=None, help="Comma-separated content weights")
    parser.add_argument("--style_weights", type=str, default=None, help="Comma-separated style weights")
    parser.add_argument("--num_steps_list", type=str, default=None, help="Comma-separated optimization step counts")

    parser.add_argument("--save_every", type=int, default=50, help="Save intermediate image every N steps; 0 disables")
    parser.add_argument("--log_every", type=int, default=20, help="Print loss every N steps; 0 disables")
    parser.add_argument("--write_report", action="store_true", help="Write system flow and module design report")
    parser.add_argument("--html_report", action="store_true", help="Generate an HTML experiment report after running")

    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.write_report:
        write_system_report(args.output_dir)

    base_runs = load_config_runs(args.config, args) if args.config else [args]
    experiment_args = []
    for run_args in base_runs:
        experiment_args.extend(expand_parameter_sweep(run_args))

    device = get_device()
    print(f"Device: {device}")
    extractor = VGGFeatureExtractor(device).to(device)

    total_specs = 0
    summary_paths = []
    for variant_args in experiment_args:
        Path(variant_args.output_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(variant_args.output_dir) / "summary.csv"
        ensure_summary_csv(summary_path)
        if summary_path not in summary_paths:
            summary_paths.append(summary_path)

        specs = build_experiment_specs(variant_args)
        total_specs += len(specs)
        for content_path, style_paths in specs:
            run_one_experiment(content_path, style_paths, variant_args, extractor, summary_path, device)

    print(f"All experiments finished: {total_specs}")
    for summary_path in summary_paths:
        print(f"Summary CSV: {summary_path}")

    if args.html_report:
        for summary_path in summary_paths:
            report_path = generate_html_report(summary_path)
            print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
