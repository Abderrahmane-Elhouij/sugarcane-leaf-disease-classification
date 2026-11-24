import locale

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"

STRATEGY_COLORS = {
    "feature_extraction": "#86b6ef",
    "partial_finetuning": "#2a78d6",
    "full_finetuning": "#104281",
    "from_scratch": "#898781",
}

MODEL_COLORS = {
    "efficientnetv2_s": "#2a78d6",
    "mobilenetv4_m": "#eda100",
    "convnextv2_t": "#1baf7a",
    "seresnet50": "#e34948",
    "fusion_finetuned": "#008300",
    "fusion_imagenet": "#008300",
    "late_fusion": "#6250d6",
    "scratch_cnn": "#898781",
}

MODEL_MARKERS = {
    "efficientnetv2_s": "o",
    "mobilenetv4_m": "s",
    "convnextv2_t": "^",
    "seresnet50": "D",
    "fusion_finetuned": "P",
    "fusion_imagenet": "P",
    "late_fusion": "v",
    "scratch_cnn": "X",
}

CLASS_COLORS = {
    "Healthy": "#008300",
    "Mosaic": "#2a78d6",
    "RedRot": "#e34948",
    "Rust": "#1baf7a",
    "Yellow": "#eda100",
}

CLASS_LINESTYLES = {
    "Healthy": "-",
    "Mosaic": "--",
    "RedRot": "-.",
    "Rust": ":",
    "Yellow": (0, (5, 1, 1, 1)),
}

BLUES = LinearSegmentedColormap.from_list(
    "report_blues",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def use_french_numbers():
    for name in ("fr_FR.UTF-8", "fr-FR", "French_France.1252"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
            return True
        except locale.Error:
            continue
    return False


def fr(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def use_report_style():
    plt.rcParams["axes.formatter.use_locale"] = use_french_numbers()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def save_figure(fig, name, raster=False):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if raster:
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=220)
    else:
        fig.savefig(FIGURES_DIR / f"{name}.pdf")
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160)
    plt.close(fig)


def overlay_heatmap(image, cam, alpha=0.5, colormap="inferno"):
    heat = plt.get_cmap(colormap)(cam)[..., :3]
    weight = alpha * cam[..., None] ** 0.5
    return np.clip((1.0 - weight) * image + weight * heat, 0.0, 1.0)


def image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
