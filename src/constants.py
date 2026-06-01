from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONTENT_DIR = "images/contents"
DEFAULT_STYLE_DIR = "images/styles"
DEFAULT_OUTPUT_DIR = "output"

DEFAULT_IMAGE_SIZE = 256
DEFAULT_NUM_STEPS = 100
DEFAULT_LEARNING_RATE = 0.03

DEFAULT_CONTENT_WEIGHT = 1.0
DEFAULT_STYLE_WEIGHT = 1e6
DEFAULT_TV_WEIGHT = 1e-6

CONTENT_LAYERS = ["conv4_2"]
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

STYLE_LAYER_WEIGHTS = {
    "conv1_1": 1.0,
    "conv2_1": 0.8,
    "conv3_1": 0.5,
    "conv4_1": 0.3,
    "conv5_1": 0.1,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SUMMARY_COLUMNS = [
    "experiment_name",
    "mode",
    "content_image",
    "style_images",
    "device",
    "image_size",
    "num_steps",
    "content_weight",
    "style_weight",
    "tv_weight",
    "style_blend_weights",
    "mask_path",
    "final_content_loss",
    "final_style_loss",
    "final_tv_loss",
    "final_total_loss",
    "runtime_seconds",
    "ssim_content_result",
    "psnr_content_result",
    "color_similarity_content_result",
    "color_similarity_style_result",
    "edge_similarity_content_result",
    "result_path",
    "loss_curve_path",
    "comparison_path",
]
