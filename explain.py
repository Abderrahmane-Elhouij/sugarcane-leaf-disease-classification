import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.config import CLASSES, MODEL_LABELS, PRETRAINED_MODELS, SCRATCH_MODEL, SEED, STRATEGIES, STRATEGY_LABELS, TABLES_DIR
from src.data import fold_frames, load_images, load_splits, resize_for_eval
from src.gradcam import GradCAM
from src.models import build_model
from src.plots import INK_SECONDARY, image_axis, overlay_heatmap, save_figure, use_report_style
from src.utils import get_device, run_dir

CONFIGS = [(name, "full_finetuning") for name in PRETRAINED_MODELS] + [(SCRATCH_MODEL, "from_scratch")]


def load_trained(name, strategy, fold):
    model = build_model(name)
    model.load_state_dict(torch.load(run_dir(name, strategy, fold) / "best.pt", map_location="cpu"))
    return model.eval()


def test_set(fold):
    _, _, test_frame = fold_frames(load_splits(), fold)
    images, labels = load_images(test_frame)
    return resize_for_eval(images.float() / 255.0), labels.numpy(), test_frame


@torch.no_grad()
def predict(model, images, device):
    outputs = [torch.softmax(model(batch.to(device)).float(), dim=1).cpu() for batch in images.split(64)]
    return torch.cat(outputs).numpy()


def to_numpy_image(image):
    return image.permute(1, 2, 0).clamp(0, 1).numpy()


def add_colorbar(fig):
    axis = fig.add_axes([0.35, 0.035, 0.30, 0.012])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=axis, orientation="horizontal")
    bar.set_label("Importance Grad-CAM (normalisée)", fontsize=7, color=INK_SECONDARY)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)


def cam_grid(columns, images, labels, picks, device, name, row_titles=None):
    fig, axes = plt.subplots(len(picks), len(columns) + 1, figsize=(1.12 * (len(columns) + 1) + 0.3, 1.2 * len(picks) + 0.7))
    axes = np.atleast_2d(axes)
    batch = images[picks]
    for row, index in enumerate(picks):
        ax = axes[row, 0]
        ax.imshow(to_numpy_image(images[index]))
        image_axis(ax)
        ax.set_ylabel(row_titles[row] if row_titles else CLASSES[labels[index]], fontsize=7.5)
        if row == 0:
            ax.set_title("Image", fontsize=8)
    for col, (title, model) in enumerate(columns, start=1):
        model = model.to(device)
        cams, probs = GradCAM(model)(batch.to(device))
        for row, index in enumerate(picks):
            ax = axes[row, col]
            ax.imshow(overlay_heatmap(to_numpy_image(images[index]), cams[row]))
            image_axis(ax)
            predicted = int(probs[row].argmax())
            ax.set_xlabel(f"{CLASSES[predicted]} ({100 * probs[row].max():.0f} %)", fontsize=6.5,
                          color=INK_SECONDARY if predicted == labels[index] else "#e34948", labelpad=2)
            if row == 0:
                ax.set_title(title, fontsize=7.5)
        model.cpu()
        torch.cuda.empty_cache()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.1, wspace=0.05, hspace=0.32)
    add_colorbar(fig)
    save_figure(fig, name, raster=True)


def pick_per_class(labels, mask, rng):
    picks = []
    for k in range(len(CLASSES)):
        candidates = np.flatnonzero((labels == k) & mask)
        if len(candidates) == 0:
            candidates = np.flatnonzero(labels == k)
        picks.append(int(rng.choice(candidates)))
    return picks


def best_backbone():
    summary_path = TABLES_DIR / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fft = summary[(summary["strategy"] == "full_finetuning") & summary["model"].isin(list(PRETRAINED_MODELS))]
        if len(fft):
            return fft.sort_values("val_accuracy_mean", ascending=False).iloc[0]["model"]
    return "convnextv2_t"


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualisations")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    use_report_style()
    device = get_device()
    rng = np.random.default_rng(SEED)
    images, labels, test_frame = test_set(args.fold)

    models = {config: load_trained(*config, args.fold) for config in CONFIGS}
    probs = {config: predict(model.to(device), images, device) for config, model in models.items()}
    for model in models.values():
        model.cpu()
    all_correct = np.all([p.argmax(1) == labels for (name, _), p in probs.items() if name != SCRATCH_MODEL], axis=0)

    picks = pick_per_class(labels, all_correct, rng)
    columns = [(MODEL_LABELS[name], models[(name, strategy)]) for name, strategy in CONFIGS]
    cam_grid(columns, images, labels, picks, device, "gradcam_models")

    best = best_backbone()
    strategy_models = [(STRATEGY_LABELS[strategy], load_trained(best, strategy, args.fold)) for strategy in STRATEGIES]
    cam_grid(strategy_models, images, labels, picks, device, "gradcam_strategies")

    best_probs = probs[(best, "full_finetuning")]
    errors = np.flatnonzero(best_probs.argmax(1) != labels)
    if len(errors):
        chosen = errors[np.argsort(-best_probs[errors].max(1))][:6]
        titles = [f"Réelle : {CLASSES[labels[i]]}" for i in chosen]
        error_columns = [(f"{MODEL_LABELS[best]} (FFT)", models[(best, "full_finetuning")])]
        error_columns += [(MODEL_LABELS[name], models[(name, "full_finetuning")])
                          for name in PRETRAINED_MODELS if name != best][:2]
        cam_grid(error_columns, images, labels, list(chosen), device, "gradcam_errors", row_titles=titles)
    rows = [{"path": test_frame.loc[i, "path"], "true": CLASSES[labels[i]], "predicted": CLASSES[best_probs[i].argmax()],
             "confidence": float(best_probs[i].max())} for i in errors]
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"errors_{best}_fold{args.fold}.csv", index=False)
    print(f"Grad-CAM figures written (best backbone: {best}, {len(errors)} test errors on fold {args.fold})")


if __name__ == "__main__":
    main()
