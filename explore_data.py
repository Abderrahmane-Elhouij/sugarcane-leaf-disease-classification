import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from src.config import CLASS_NAMES_FR, CLASSES, CLEANING_REPORT, RAW_DIR, SEED, TABLES_DIR
from src.data import GpuAugment, load_images, load_splits, resize_for_eval
from src.plots import INK_SECONDARY, image_axis, save_figure, use_report_style
from src.utils import load_json


def raw_inventory():
    rows = []
    for class_name in CLASSES:
        for path in sorted((RAW_DIR / class_name).iterdir()):
            with Image.open(path) as image:
                rows.append({"class_name": class_name, "width": image.width, "height": image.height})
    return pd.DataFrame(rows)


def plot_samples(splits, per_class=3):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(per_class, len(CLASSES), figsize=(7.0, 5.4))
    for col, class_name in enumerate(CLASSES):
        sources = splits.loc[splits["class_name"] == class_name, "source"].to_numpy()
        for row, source in enumerate(rng.choice(sources, size=per_class, replace=False)):
            with Image.open(RAW_DIR / source) as image:
                cell = ImageOps.fit(image.convert("RGB"), (300, 480), Image.Resampling.LANCZOS)
            ax = axes[row, col]
            ax.imshow(cell)
            image_axis(ax)
            if row == 0:
                ax.set_title(f"{class_name}\n({CLASS_NAMES_FR[class_name]})", fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save_figure(fig, "dataset_samples", raster=True)


def plot_cleaning(report):
    raw = np.array([report["raw_counts"][c] for c in CLASSES])
    exact = raw - np.array([report["after_exact_dedup"][c] for c in CLASSES])
    final = np.array([report["final_counts"][c] for c in CLASSES])
    near = raw - exact - final
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    positions = np.arange(len(CLASSES))[::-1]
    parts = [
        (final, "Images conservées", "#2a78d6"),
        (exact, "Doublons exacts supprimés", "#eb6834"),
        (near, "Quasi-doublons supprimés", "#1baf7a"),
    ]
    left = np.zeros(len(CLASSES))
    for values, label, color in parts:
        ax.barh(positions, values, left=left, height=0.62, color=color, label=label, edgecolor="white", linewidth=1.2)
        left += values
    for y, total, kept in zip(positions, raw, final):
        ax.text(total + 6, y, f"{kept} / {total}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(positions, CLASSES)
    ax.set_xlabel("Nombre d'images")
    ax.set_xlim(0, raw.max() * 1.16)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    save_figure(fig, "class_distribution")


def plot_image_sizes(inventory):
    sizes = inventory.groupby(["width", "height"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(sizes["width"], sizes["height"], s=12 + sizes["count"] * 0.9, color="#2a78d6", alpha=0.55,
               edgecolor="white", linewidth=0.8)
    for row in sizes.nlargest(3, "count").itertuples():
        ax.annotate(f"{row.width}×{row.height} ({row.count} images)", (row.width, row.height),
                    xytext=(12, -14 if row.width > 600 else 10), textcoords="offset points",
                    fontsize=7.5, color=INK_SECONDARY)
    ax.set_xlabel("Largeur (pixels)")
    ax.set_ylabel("Hauteur (pixels)")
    ax.set_xlim(150, 1700)
    ax.set_ylim(150, 1700)
    ax.set_aspect("equal")
    save_figure(fig, "image_sizes")


def plot_augmentations(splits, n_views=6):
    picks = splits[splits["split"] == "dev"].groupby("class_name").head(1)
    picks = picks.set_index("class_name").loc[["Mosaic", "Rust"]].reset_index()
    images, _ = load_images(picks)
    images = images.float() / 255.0
    augment = GpuAugment(seed=SEED)
    views = [resize_for_eval(images)] + [augment(images) for _ in range(n_views)]
    fig, axes = plt.subplots(len(picks), n_views + 1, figsize=(7.0, 2.2))
    for col, batch in enumerate(views):
        for row in range(len(picks)):
            ax = axes[row, col]
            ax.imshow(batch[row].permute(1, 2, 0).clamp(0, 1).numpy())
            image_axis(ax)
            if row == 0:
                ax.set_title("Originale" if col == 0 else f"Augmentée {col}", fontsize=8)
            if col == 0:
                ax.set_ylabel(picks.loc[row, "class_name"], fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save_figure(fig, "augmentation_examples", raster=True)


def write_dataset_table(report):
    splits = report["splits"]
    lines = []
    for class_name in CLASSES:
        train = splits["fold_0_train"][class_name]
        val = splits["fold_0_val"][class_name]
        test = splits["test"][class_name]
        lines.append(
            f"{class_name} & {CLASS_NAMES_FR[class_name]} & {report['raw_counts'][class_name]} & "
            f"{report['final_counts'][class_name]} & {train} & {val} & {test} \\\\"
        )
    total = lambda key: sum(splits[key].values())
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{Total}} & & {report['raw_images']} & {report['final_images']} & "
        f"{total('fold_0_train')} & {total('fold_0_val')} & {total('test')} \\\\"
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "dataset_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(4)
    use_report_style()
    report = load_json(CLEANING_REPORT)
    splits = load_splits()
    plot_samples(splits)
    plot_cleaning(report)
    plot_image_sizes(raw_inventory())
    plot_augmentations(splits)
    write_dataset_table(report)
    print("Exploration figures written")


if __name__ == "__main__":
    main()
