import csv
import html
import time
from pathlib import Path

from .constants import SUMMARY_COLUMNS


def ensure_summary_csv(summary_path):
    summary_path = Path(summary_path)
    if summary_path.exists():
        with open(summary_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)

        if old_header == SUMMARY_COLUMNS:
            return

        backup_path = summary_path.with_name(
            f"{summary_path.stem}_legacy_{time.strftime('%Y%m%d_%H%M%S')}{summary_path.suffix}"
        )
        summary_path.rename(backup_path)
        print(f"Existing summary schema backed up: {backup_path}")

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)


def append_summary_csv(summary_path, row):
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_summary(summary_path):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return []

    with open(summary_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _safe_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "inf":
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return default


def _rel_path(path, base_dir):
    if not path:
        return ""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(base_dir).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def generate_html_report(summary_path, report_path=None, max_items=100):
    summary_path = Path(summary_path)
    rows = read_summary(summary_path)
    if report_path is None:
        report_path = summary_path.with_name("report.html")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(rows, key=lambda row: _safe_float(row, "final_total_loss"))
    total_runtime = sum(_safe_float(row, "runtime_seconds") for row in rows)
    avg_ssim = sum(_safe_float(row, "ssim_content_result") for row in rows) / len(rows) if rows else 0.0

    cards = []
    for index, row in enumerate(rows[:max_items], start=1):
        comparison_src = html.escape(_rel_path(row.get("comparison_path", ""), report_path.parent))
        loss_src = html.escape(_rel_path(row.get("loss_curve_path", ""), report_path.parent))
        title = html.escape(row.get("experiment_name", f"experiment-{index}"))
        metrics = (
            f"SSIM {html.escape(row.get('ssim_content_result', ''))} | "
            f"PSNR {html.escape(row.get('psnr_content_result', ''))} | "
            f"Style color {html.escape(row.get('color_similarity_style_result', ''))}"
        )
        cards.append(
            f"""
            <article class="card">
              <h2>{index}. {title}</h2>
              <p>{metrics}</p>
              <p>Runtime {html.escape(row.get('runtime_seconds', ''))}s | Steps {html.escape(row.get('num_steps', ''))}</p>
              <div class="images">
                {f'<img src="{comparison_src}" alt="comparison">' if comparison_src else ''}
                {f'<img src="{loss_src}" alt="loss curve">' if loss_src else ''}
              </div>
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Neural Style Transfer Experiment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f6f7f9; color: #1f2933; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .stat, .card {{ background: white; border: 1px solid #d8dee6; border-radius: 8px; padding: 16px; }}
    .stat strong {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    .card {{ margin-bottom: 16px; }}
    .card h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    img {{ width: 100%; border: 1px solid #d8dee6; border-radius: 6px; background: white; }}
  </style>
</head>
<body>
  <header>
    <h1>Neural Style Transfer Experiment Report</h1>
    <p>Generated from {html.escape(summary_path.name)}.</p>
  </header>
  <section class="stats">
    <div class="stat"><strong>{len(rows)}</strong>Experiments</div>
    <div class="stat"><strong>{total_runtime:.2f}s</strong>Total runtime</div>
    <div class="stat"><strong>{avg_ssim:.4f}</strong>Average SSIM</div>
  </section>
  {''.join(cards) if cards else '<p>No experiments found.</p>'}
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def write_system_report(output_dir):
    report_path = Path(output_dir) / "system_design.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """# Neural Style Transfer System Design

## System Flow

```mermaid
flowchart TD
    A[Input content images] --> C[Image loader and normalizer]
    B[Input style images] --> C
    M[Optional local mask] --> L[Local result blending]
    C --> D[VGG19 feature extractor]
    D --> E[Content feature loss]
    D --> F[Style Gram matrix loss]
    G[Generated image] --> D
    E --> H[Weighted total loss]
    F --> H
    I[TV smoothness loss] --> H
    H --> J[Adam optimization loop]
    J --> G
    G --> K[Save result image]
    K --> L
    L --> N[Comparison image]
    H --> O[Loss curves]
    K --> P[Metrics: runtime, loss, SSIM, PSNR, color similarity, edge similarity]
    N --> Q[CSV summary and experiment folders]
    O --> Q
    P --> Q
    Q --> R[HTML experiment report]
```

## Module Design

| Module | Responsibility |
|---|---|
| `src/io_utils.py` | Load content/style images, normalize tensors, save generated images. |
| `src/model.py` | Extract VGG19 content and style layer activations. |
| `src/losses.py` | Compute content loss, style Gram loss, and total variation loss. |
| `src/runner.py` | Build and run single, batch, all-pairs, multi-style, and parameter-sweep experiments. |
| `src/metrics.py` | Record SSIM, PSNR, color similarity, style color similarity, and edge similarity. |
| `src/visualization.py` | Export loss curves and content/style/result comparison images. |
| `src/reports.py` | Maintain CSV summaries and generate HTML reports. |
| `app.py` | Local Web interface for interactive style transfer. |

## Experiment Types

- Same content with different styles: `--mode same_content_diff_styles --fixed_content <path>`.
- Different contents with same style: `--mode same_style_diff_contents --fixed_style <path>`.
- Parameter comparison: add `--content_weights`, `--style_weights`, or `--num_steps_list`.
- Multi-style fusion: `--mode multi_style --styles style1.jpg,style2.jpg --style_blend_weights 0.7,0.3`.
- Local style transfer: add `--mask <grayscale-mask-path>`.
- Config-driven runs: `--config configs/experiments.yaml`.
""",
        encoding="utf-8",
    )
    print(f"System design report written: {report_path}")
    return report_path
