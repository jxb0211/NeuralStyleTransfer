from copy import deepcopy
from pathlib import Path


def _load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Config mode requires PyYAML. Install dependencies from requirements.txt.") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _set_if_present(args, data, key):
    if key in data:
        setattr(args, key, data[key])


def load_config_runs(config_path, base_args):
    config_path = Path(config_path)
    data = _load_yaml(config_path)
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in config: {config_path}")

    runs = []
    scalar_keys = [
        "mode",
        "content",
        "style",
        "styles",
        "style_blend_weights",
        "fixed_content",
        "fixed_style",
        "content_dir",
        "style_dir",
        "output_dir",
        "mask",
        "image_size",
        "num_steps",
        "learning_rate",
        "content_weight",
        "style_weight",
        "tv_weight",
        "content_weights",
        "style_weights",
        "num_steps_list",
        "save_every",
        "log_every",
    ]

    for experiment in experiments:
        run_args = deepcopy(base_args)
        for key in scalar_keys:
            _set_if_present(run_args, defaults, key)
            _set_if_present(run_args, experiment, key)

        if isinstance(getattr(run_args, "styles", None), list):
            run_args.styles = ",".join(str(item) for item in run_args.styles)
        if isinstance(getattr(run_args, "style_blend_weights", None), list):
            run_args.style_blend_weights = ",".join(str(item) for item in run_args.style_blend_weights)
        runs.append(run_args)

    return runs
