# Neural Style Transfer Lab

This project is a VGG19-based neural style transfer lab. It supports single image transfer, batch experiments, all content/style combinations, multi-style fusion, local mask transfer, parameter sweeps, metric tracking, and HTML experiment reports.

## Project Structure

```text
.
|-- app.py                  # Streamlit Web interface
|-- main.py                 # Command-line entry
|-- configs/
|   `-- experiments.yaml    # Example config-driven experiment suite
|-- images/
|   |-- contents/           # Content images
|   `-- styles/             # Style images
|-- output/                 # Generated results, CSV summaries, reports
`-- src/
    |-- config_runner.py    # YAML config loading
    |-- constants.py        # Shared defaults and summary schema
    |-- io_utils.py         # Image loading and saving
    |-- losses.py           # Gram matrix and total variation loss
    |-- metrics.py          # SSIM, PSNR, color and edge metrics
    |-- model.py            # VGG19 feature extractor
    |-- parsing.py          # CLI parsing helpers
    |-- reports.py          # CSV and HTML reports
    |-- runner.py           # Experiment orchestration
    `-- visualization.py    # Loss curves and comparison images
```

## Installation

```bash
pip install -r requirements.txt
```

If you already use the bundled `.venv`, activate it first.

## Command-Line Usage

Single transfer:

```bash
python main.py --mode single --content images/contents/content1.jpg --style images/styles/starry_night.jpg --html_report
```

Run one content image against every style image:

```bash
python main.py --mode same_content_diff_styles --fixed_content images/contents/content1.jpg --html_report
```

Run all content/style combinations:

```bash
python main.py --mode all --html_report
```

Multi-style fusion:

```bash
python main.py --mode multi_style --content images/contents/content1.jpg --styles images/styles/starry_night.jpg,images/styles/great_wave.jpg --style_blend_weights 0.6,0.4 --html_report
```

Parameter sweep:

```bash
python main.py --mode single --content images/contents/content1.jpg --style images/styles/starry_night.jpg --content_weights 1,3 --style_weights 500000,1000000 --num_steps_list 100,200 --html_report
```

Config-driven experiment suite:

```bash
python main.py --config configs/experiments.yaml --html_report
```

System design report:

```bash
python main.py --mode single --content images/contents/content1.jpg --style images/styles/starry_night.jpg --write_report
```

## Web Interface

```bash
streamlit run app.py
```

The Web interface lets you choose example images or upload new content/style images, tune the main transfer parameters, run the experiment, preview the comparison image and loss curve, and generate the HTML report.

## Outputs

Each experiment writes a folder under `output/`:

- `result.jpg`: final stylized image
- `result_step_*.jpg`: intermediate snapshots
- `loss_curve.png`: total/content/style/TV loss curve
- `comparison.jpg`: content, style, and result comparison

The project also writes:

- `output/summary.csv`: experiment metadata, losses, runtime, and metrics
- `output/report.html`: visual report sorted by final total loss
- `output/system_design.md`: system flow and module design when `--write_report` is used

## Metrics

The summary includes:

- `ssim_content_result`: structural similarity between content and result
- `psnr_content_result`: pixel-level content preservation
- `color_similarity_content_result`: color histogram similarity to content
- `color_similarity_style_result`: color histogram similarity to style images
- `edge_similarity_content_result`: edge-level content preservation
