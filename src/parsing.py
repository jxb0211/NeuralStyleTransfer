from pathlib import Path


def parse_csv_floats(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_paths(text):
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_blend_weights(text, count):
    if text is None:
        return [1.0 / count] * count

    weights = parse_csv_floats(text)
    if len(weights) != count:
        raise ValueError("--style_blend_weights must match the number of style images")

    total = sum(weights)
    if total <= 0:
        raise ValueError("--style_blend_weights must sum to a positive value")

    return [w / total for w in weights]
