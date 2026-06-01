from pathlib import Path
from types import SimpleNamespace

import streamlit as st

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
from src.io_utils import get_image_list
from src.model import VGGFeatureExtractor
from src.reports import ensure_summary_csv, generate_html_report, read_summary
from src.runner import get_device, run_one_experiment


UPLOAD_DIR = Path("web_uploads")


def save_upload(uploaded_file, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    save_path = folder / safe_name
    save_path.write_bytes(uploaded_file.getbuffer())
    return save_path


def image_options(folder):
    try:
        return get_image_list(folder)
    except FileNotFoundError:
        return []


@st.cache_resource
def load_extractor():
    device = get_device()
    return VGGFeatureExtractor(device).to(device), device


def build_args(output_dir, image_size, num_steps, learning_rate, content_weight, style_weight, tv_weight, save_every):
    return SimpleNamespace(
        mode="multi_style",
        content=None,
        style=None,
        styles=None,
        style_blend_weights=None,
        fixed_content=None,
        fixed_style=None,
        content_dir=DEFAULT_CONTENT_DIR,
        style_dir=DEFAULT_STYLE_DIR,
        output_dir=output_dir,
        mask=None,
        image_size=image_size,
        num_steps=num_steps,
        learning_rate=learning_rate,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        content_weights=None,
        style_weights=None,
        num_steps_list=None,
        save_every=save_every,
        log_every=0,
    )


st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("Neural Style Transfer Lab")

with st.sidebar:
    st.header("Inputs")
    content_source = st.radio("Content image source", ["Example", "Upload"], horizontal=True)
    style_source = st.radio("Style image source", ["Example", "Upload"], horizontal=True)

    content_path = None
    content_examples = image_options(DEFAULT_CONTENT_DIR)
    if content_source == "Example":
        content_path = st.selectbox("Content image", content_examples, format_func=lambda p: p.name)
    else:
        upload = st.file_uploader("Upload content image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload is not None:
            content_path = save_upload(upload, UPLOAD_DIR / "contents")

    style_paths = []
    style_examples = image_options(DEFAULT_STYLE_DIR)
    if style_source == "Example":
        style_paths = st.multiselect("Style images", style_examples, default=style_examples[:1], format_func=lambda p: p.name)
    else:
        uploads = st.file_uploader(
            "Upload style images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        style_paths = [save_upload(upload, UPLOAD_DIR / "styles") for upload in uploads]

    st.header("Parameters")
    image_size = st.slider("Image size", 128, 512, DEFAULT_IMAGE_SIZE, step=32)
    num_steps = st.slider("Optimization steps", 10, 500, DEFAULT_NUM_STEPS, step=10)
    learning_rate = st.number_input("Learning rate", min_value=0.001, value=DEFAULT_LEARNING_RATE, step=0.005, format="%.3f")
    content_weight = st.number_input("Content weight", min_value=0.0, value=DEFAULT_CONTENT_WEIGHT, step=0.5)
    style_weight = st.number_input("Style weight", min_value=1.0, value=float(DEFAULT_STYLE_WEIGHT), step=100000.0)
    tv_weight = st.number_input("TV weight", min_value=0.0, value=DEFAULT_TV_WEIGHT, format="%.8f")
    save_every = st.number_input("Save every N steps", min_value=0, value=50, step=10)
    output_dir = st.text_input("Output directory", DEFAULT_OUTPUT_DIR)

    run_button = st.button("Run style transfer", type="primary", use_container_width=True)

left, right = st.columns([1, 1])

with left:
    st.subheader("Selected inputs")
    if content_path:
        st.image(str(content_path), caption=f"Content: {Path(content_path).name}", width="stretch")
    if style_paths:
        st.image([str(path) for path in style_paths], caption=[Path(path).name for path in style_paths], width=180)

with right:
    st.subheader("Latest results")
    summary_path = Path(output_dir) / "summary.csv"
    rows = read_summary(summary_path)
    if rows:
        latest = rows[-1]
        comparison_path = latest.get("comparison_path")
        if comparison_path and Path(comparison_path).exists():
            st.image(comparison_path, caption=latest.get("experiment_name", "Latest comparison"), width="stretch")
        st.dataframe(rows[-10:], use_container_width=True)
    else:
        st.info("No experiments have been run in this output directory yet.")

if run_button:
    if content_path is None or not style_paths:
        st.error("Choose one content image and at least one style image.")
    else:
        args = build_args(
            output_dir=output_dir,
            image_size=image_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            save_every=int(save_every),
        )
        args.style_blend_weights = None
        summary_path = Path(output_dir) / "summary.csv"
        ensure_summary_csv(summary_path)

        with st.spinner("Running neural style transfer..."):
            extractor, device = load_extractor()
            result = run_one_experiment(content_path, style_paths, args, extractor, summary_path, device)
            report_path = generate_html_report(summary_path)

        st.success("Experiment finished.")
        st.image(str(result["comparison_path"]), caption="Comparison", width="stretch")
        st.image(str(result["loss_curve_path"]), caption="Loss curve", width="stretch")
        st.write("Metrics", result["metrics"])
        st.write(f"HTML report: {report_path}")
