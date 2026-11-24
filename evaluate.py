import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BENCHMARK_DIR,
    CLASSES,
    MODEL_LABELS,
    N_FOLDS,
    PRETRAINED_MODELS,
    RUNS_DIR,
    SCRATCH_MODEL,
    STRATEGIES,
    STRATEGY_LABELS,
    TABLES_DIR,
)
from src.data import fold_frames, load_splits
from src.metrics import normalized_confusion, one_vs_rest_roc, per_class_metrics
from src.plots import (
    AXIS,
    BLUES,
    CLASS_COLORS,
    CLASS_LINESTYLES,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MODEL_COLORS,
    MODEL_MARKERS,
    STRATEGY_COLORS,
    TRAIN_COLOR,
    VAL_COLOR,
    fr,
    save_figure,
    use_report_style,
)
from src.utils import load_json

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
BACKBONES = list(PRETRAINED_MODELS)
ALL_MEMBERS = "+".join(BACKBONES)
SHORT_NAMES = {"efficientnetv2_s": "E", "mobilenetv4_m": "M", "convnextv2_t": "C", "seresnet50": "S"}
CORRUPTION_LABELS = {
    "gaussian_noise": "Bruit gaussien",
    "gaussian_blur": "Flou gaussien",
    "darkening": "Assombrissement",
    "jpeg": "Compression JPEG",
    "occlusion": "Occultation",
}
CORRUPTION_UNITS = {
    "gaussian_noise": "σ",
    "gaussian_blur": "σ",
    "darkening": "facteur",
    "jpeg": "qualité",
    "occlusion": "côté",
}


def load_runs():
    rows = []
    for path in sorted(RUNS_DIR.glob("*/*/fold_*/summary.json")):
        summary = load_json(path)
        row = {"model": summary["model"], "strategy": summary["strategy"], "fold": summary["fold"], "dir": path.parent}
        for split in ("val", "test"):
            row.update({f"{split}_{key}": value for key, value in summary[split].items()})
        for key in ("total_params", "trainable_params", "train_seconds", "total_seconds",
                    "peak_memory_mb", "best_epoch", "members_seconds", "embedding_dim", "train_images"):
            row[key] = summary.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs):
    numeric = [column for column in runs.columns if column not in ("model", "strategy", "fold", "dir")]
    grouped = runs.groupby(["model", "strategy"])
    means = grouped[numeric].mean().add_suffix("_mean")
    stds = grouped[numeric].std(ddof=1).add_suffix("_std")
    counts = grouped.size().rename("folds")
    return pd.concat([means, stds, counts], axis=1).reset_index()


def row_of(summary, model, strategy):
    match = summary[(summary["model"] == model) & (summary["strategy"] == strategy)]
    return match.iloc[0] if len(match) else None


def pooled_predictions(runs, model, strategy):
    selected = runs[(runs["model"] == model) & (runs["strategy"] == strategy)].sort_values("fold")
    probs, labels = [], []
    for directory in selected["dir"]:
        data = np.load(directory / "predictions.npz")
        probs.append(data["test_probs"])
        labels.append(data["test_labels"])
    return np.concatenate(probs), np.concatenate(labels)


def histories(runs, model, strategy):
    selected = runs[(runs["model"] == model) & (runs["strategy"] == strategy)].sort_values("fold")
    return [pd.read_csv(directory / "history.csv") for directory in selected["dir"]]


def percent(mean, std, digits=1):
    return f"{fr(100 * mean, digits)} $\\pm$ {fr(100 * std, digits)}"


def plain(mean, std, digits=4):
    return f"{fr(mean, digits)} $\\pm$ {fr(std, digits)}"


def trainable_text(trainable, total):
    share = 100 * trainable / total
    count = f"{fr(trainable / 1e3, 1)}\\,k" if trainable < 1e5 else f"{fr(trainable / 1e6, 2)}\\,M"
    percent_text = fr(share, 2) if share < 1 else f"{share:.0f}"
    return f"{count} ({percent_text}\\,\\%)"


def same_display(a, b, scale=100, digits=1):
    return round(scale * a, digits) == round(scale * b, digits)


def bold_if(text, condition):
    return f"\\textbf{{{text}}}" if condition else text


def write_table(name, lines):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def minutes(seconds):
    minutes_part, seconds_part = divmod(int(round(seconds)), 60)
    return f"{minutes_part}~min~{seconds_part:02d}~s"


def main_entries(summary):
    entries = [(name, "full_finetuning") for name in BACKBONES]
    entries += [(SCRATCH_MODEL, "from_scratch"), ("fusion_finetuned", ALL_MEMBERS), ("late_fusion", ALL_MEMBERS)]
    return [(model, strategy) for model, strategy in entries if row_of(summary, model, strategy) is not None]


def pipeline_seconds(summary, model, strategy):
    row = row_of(summary, model, strategy)
    if model in ("fusion_finetuned", "late_fusion"):
        members = sum(row_of(summary, name, "full_finetuning")["total_seconds_mean"] for name in BACKBONES)
        own = 0.0 if model == "late_fusion" else row["total_seconds_mean"]
        return members + own
    return row["total_seconds_mean"]


def table_main(summary):
    entries = main_entries(summary)
    best = {metric: max(row_of(summary, m, s)[f"test_{metric}_mean"] for m, s in entries) for metric in METRICS}
    lines = []
    for model, strategy in entries:
        row = row_of(summary, model, strategy)
        cells = [MODEL_LABELS[model]]
        for metric in METRICS:
            mean, std = row[f"test_{metric}_mean"], row[f"test_{metric}_std"]
            text = plain(mean, std) if metric == "auc" else percent(mean, std)
            display = (1, 4) if metric == "auc" else (100, 1)
            cells.append(bold_if(text, same_display(mean, best[metric], *display)))
        params = row["total_params_mean"]
        if model == "fusion_finetuned":
            params = sum(row_of(summary, name, "full_finetuning")["total_params_mean"] for name in BACKBONES)
        cells.append(fr(params / 1e6, 2))
        seconds = pipeline_seconds(summary, model, strategy)
        marker = "$^{*}$" if model in ("fusion_finetuned", "late_fusion") else ""
        cells.append(minutes(seconds) + marker)
        if model == SCRATCH_MODEL:
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    write_table("results_main", lines)


def table_strategies(summary):
    lines = []
    for name in BACKBONES:
        rows = [row_of(summary, name, strategy) for strategy in STRATEGIES]
        if all(row is None for row in rows):
            continue
        best_acc = max(row["test_accuracy_mean"] for row in rows if row is not None)
        for index, (strategy, row) in enumerate(zip(STRATEGIES, rows)):
            if row is None:
                continue
            cells = [
                f"\\multirow{{3}}{{*}}{{{MODEL_LABELS[name]}}}" if index == 0 else "",
                STRATEGY_LABELS[strategy],
                trainable_text(row["trainable_params_mean"], row["total_params_mean"]),
                percent(row["val_accuracy_mean"], row["val_accuracy_std"]),
                bold_if(percent(row["test_accuracy_mean"], row["test_accuracy_std"]),
                        same_display(row["test_accuracy_mean"], best_acc)),
                percent(row["test_f1_mean"], row["test_f1_std"]),
                plain(row["test_auc_mean"], row["test_auc_std"]),
                minutes(row["total_seconds_mean"]),
                fr(row["peak_memory_mb_mean"] / 1024, 2),
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    scratch = row_of(summary, SCRATCH_MODEL, "from_scratch")
    if scratch is not None:
        lines.append(" & ".join([
            MODEL_LABELS[SCRATCH_MODEL], STRATEGY_LABELS["from_scratch"],
            trainable_text(scratch["trainable_params_mean"], scratch["total_params_mean"]),
            percent(scratch["val_accuracy_mean"], scratch["val_accuracy_std"]),
            percent(scratch["test_accuracy_mean"], scratch["test_accuracy_std"]),
            percent(scratch["test_f1_mean"], scratch["test_f1_std"]),
            plain(scratch["test_auc_mean"], scratch["test_auc_std"]),
            minutes(scratch["total_seconds_mean"]),
            fr(scratch["peak_memory_mb_mean"] / 1024, 2),
        ]) + " \\\\")
    elif lines:
        lines.pop()
    write_table("results_strategies", lines)


def table_per_class(runs, summary):
    entries = main_entries(summary)
    frames = {}
    for model, strategy in entries:
        probs, labels = pooled_predictions(runs, model, strategy)
        frames[model] = per_class_metrics(labels, probs, CLASSES)
    for metric, name, digits in (("f1", "results_per_class_f1", 1), ("recall", "results_per_class_recall", 1),
                                 ("auc", "results_per_class_auc", 4)):
        lines = []
        for k, class_name in enumerate(CLASSES):
            values = [frames[model].loc[k, metric] for model, _ in entries]
            best = max(values)
            cells = [class_name]
            for value in values:
                text = fr(100 * value, digits) if metric != "auc" else fr(value, digits)
                cells.append(bold_if(text, same_display(value, best, 1 if metric == "auc" else 100, digits)))
            lines.append(" & ".join(cells) + " \\\\")
        write_table(name, lines)
    pd.concat({MODEL_LABELS[m]: f for m, f in frames.items()}).to_csv(TABLES_DIR / "per_class_metrics.csv")


def correctness(runs, model, strategy):
    selected = runs[(runs["model"] == model) & (runs["strategy"] == strategy)].sort_values("fold")
    rows = []
    for directory in selected["dir"]:
        data = np.load(directory / "predictions.npz")
        rows.append(data["test_probs"].argmax(1) == data["test_labels"])
    return np.stack(rows).astype(float)


def bootstrap_interval(values, samples=10000, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def table_significance(runs, summary):
    entries = main_entries(summary)
    per_image = {(m, s): correctness(runs, m, s).mean(axis=0) for m, s in entries}
    singles = [key for key in per_image if key[0] in BACKBONES]
    reference = max(singles, key=lambda key: per_image[key].mean())
    lines = []
    for model, strategy in entries:
        values = per_image[(model, strategy)]
        low, high = bootstrap_interval(values)
        cells = [MODEL_LABELS[model], fr(100 * values.mean()), f"[{fr(100 * low)} ; {fr(100 * high)}]"]
        if (model, strategy) == reference:
            cells += ["référence", "--", "--"]
        else:
            difference = values - per_image[reference]
            d_low, d_high = bootstrap_interval(difference)
            significant = d_low > 0 or d_high < 0
            cells += [fr(100 * difference.mean()), f"[{fr(100 * d_low)} ; {fr(100 * d_high)}]",
                      "oui" if significant else "non"]
        lines.append(" & ".join(cells) + " \\\\")
    write_table("results_significance", lines)


def subset_label(key):
    return " + ".join(SHORT_NAMES[name] for name in key.split("+"))


def table_fusion(summary):
    subsets = [list(c) for size in range(1, len(BACKBONES) + 1) for c in itertools.combinations(BACKBONES, size)]
    columns = {}
    for source in ("fusion_imagenet", "fusion_finetuned"):
        columns[source] = {"+".join(s): row_of(summary, source, "+".join(s)) for s in subsets}
    best = {source: max([r["test_accuracy_mean"] for r in rows.values() if r is not None], default=np.nan)
            for source, rows in columns.items()}
    if all(np.isnan(value) for value in best.values()):
        return
    lines = []
    previous_size = 1
    for members in subsets:
        key = "+".join(members)
        if len(members) != previous_size:
            lines.append("\\midrule")
            previous_size = len(members)
        cells = [str(len(members)), subset_label(key)]
        for source in ("fusion_imagenet", "fusion_finetuned"):
            row = columns[source][key]
            if row is None:
                cells += ["--", "--"]
                continue
            cells.append(bold_if(percent(row["test_accuracy_mean"], row["test_accuracy_std"]),
                                 same_display(row["test_accuracy_mean"], best[source])))
            cells.append(percent(row["test_f1_mean"], row["test_f1_std"]))
        lines.append(" & ".join(cells) + " \\\\")
    late = row_of(summary, "late_fusion", ALL_MEMBERS)
    if late is not None:
        lines.append("\\midrule")
        lines.append(" & ".join([
            "4", "Vote souple (E + M + C + S)", "--", "--",
            percent(late["test_accuracy_mean"], late["test_accuracy_std"]),
            percent(late["test_f1_mean"], late["test_f1_std"]),
        ]) + " \\\\")
    write_table("results_fusion", lines)


def plot_learning_curves(runs, strategy, models, name):
    available = [m for m in models if len(histories(runs, m, strategy if m != SCRATCH_MODEL else "from_scratch"))]
    if not available:
        return
    fig, axes = plt.subplots(2, len(available), figsize=(1.3 * len(available) + 0.6, 3.7), sharex=True)
    axes = np.atleast_2d(axes).reshape(2, len(available))
    for col, model in enumerate(available):
        runs_history = histories(runs, model, strategy if model != SCRATCH_MODEL else "from_scratch")
        epochs = runs_history[0]["epoch"].to_numpy()
        for row, (metric, label, scale) in enumerate((("acc", "Accuracy (%)", 100.0), ("loss", "Perte", 1.0))):
            ax = axes[row, col]
            for split, color, legend in (("train", TRAIN_COLOR, "Entraînement"), ("val", VAL_COLOR, "Validation")):
                values = np.stack([h[f"{split}_{metric}"].to_numpy() for h in runs_history]) * scale
                mean, std = values.mean(axis=0), values.std(axis=0)
                ax.plot(epochs, mean, color=color, linewidth=1.4, label=legend)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=8.5)
                ax.set_ylim(top=100.5)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Époque")
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6, h_pad=0.5)
    save_figure(fig, name)


def plot_confusions(runs, summary):
    entries = main_entries(summary)[:6]
    cols = 3
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.45 * rows + 0.2))
    axes = np.atleast_2d(axes)
    for index, (model, strategy) in enumerate(entries):
        ax = axes.flat[index]
        probs, labels = pooled_predictions(runs, model, strategy)
        _, rates = normalized_confusion(labels, probs, len(CLASSES))
        ax.imshow(rates, cmap=BLUES, vmin=0.0, vmax=1.0)
        for i, j in itertools.product(range(len(CLASSES)), repeat=2):
            value = 100 * rates[i, j]
            text = fr(value, 1) if value >= 0.05 else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.6,
                    color="white" if rates[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
        ax.set_title(MODEL_LABELS[model], fontsize=8.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        if index % cols == 0:
            ax.set_ylabel("Classe réelle")
        if index >= len(entries) - cols:
            ax.set_xlabel("Classe prédite")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    save_figure(fig, "confusion_matrices")


def plot_rocs(runs, summary):
    entries = main_entries(summary)[:6]
    cols = 3
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.35 * rows + 0.45), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for index, (model, strategy) in enumerate(entries):
        ax = axes.flat[index]
        probs, labels = pooled_predictions(runs, model, strategy)
        curves = one_vs_rest_roc(labels, probs)
        inset = ax.inset_axes([0.40, 0.10, 0.56, 0.56])
        for k, class_name in enumerate(CLASSES):
            fpr, tpr, _ = curves[k]
            style = dict(color=CLASS_COLORS[class_name], linestyle=CLASS_LINESTYLES[class_name], linewidth=1.3)
            ax.plot(fpr, tpr, label=class_name, **style)
            inset.plot(fpr, tpr, **style)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8)
        inset.set_xlim(-0.001, 0.06)
        inset.set_ylim(0.85, 1.004)
        inset.tick_params(labelsize=6)
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
        ax.set_title(f"{MODEL_LABELS[model]}\nAUC macro = {fr(curves['macro'][2], 4)}", fontsize=8)
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0.0, 1.01)
        if index % cols == 0:
            ax.set_ylabel("Taux de vrais positifs")
        if index >= len(entries) - cols:
            ax.set_xlabel("Taux de faux positifs")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.7, h_pad=0.8)
    save_figure(fig, "roc_curves")


def plot_strategy_comparison(summary):
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    positions = np.arange(len(BACKBONES))[::-1]
    label_offsets = {"feature_extraction": 0.14, "partial_finetuning": -0.14, "full_finetuning": 0.14}
    for y, name in zip(positions, BACKBONES):
        rows = [row_of(summary, name, strategy) for strategy in STRATEGIES]
        means = [100 * r["test_accuracy_mean"] for r in rows if r is not None]
        if not means:
            continue
        ax.plot([min(means), max(means)], [y, y], color=AXIS, linewidth=1.0, zorder=1)
        for strategy, row in zip(STRATEGIES, rows):
            if row is None:
                continue
            mean, std = 100 * row["test_accuracy_mean"], 100 * row["test_accuracy_std"]
            ax.errorbar(mean, y, xerr=np.nan_to_num(std), fmt="o", markersize=6.5,
                        color=STRATEGY_COLORS[strategy], ecolor=STRATEGY_COLORS[strategy], elinewidth=1.0,
                        capsize=2, markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                        label=STRATEGY_LABELS[strategy] if y == positions[0] else None)
            offset = label_offsets[strategy]
            ax.text(mean, y + offset, fr(mean, 1), ha="center", va="bottom" if offset > 0 else "top",
                    fontsize=6.5, color=INK_SECONDARY)
    scratch = row_of(summary, SCRATCH_MODEL, "from_scratch")
    if scratch is not None:
        value = 100 * scratch["test_accuracy_mean"]
        ax.axvline(value, color=INK_MUTED, linewidth=1.0, zorder=0)
        ax.text(value, positions[0] + 0.45, f"CNN from scratch ({fr(value, 1)})", ha="center", fontsize=7,
                color=INK_SECONDARY)
    ax.set_yticks(positions, [MODEL_LABELS[n] for n in BACKBONES])
    ax.set_xlabel("Accuracy sur le test (%), moyenne et écart-type sur 5 plis")
    ax.set_ylim(-0.6, len(BACKBONES) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    save_figure(fig, "strategy_comparison")


def plot_fusion_gain(summary):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, source, title in zip(axes, ("fusion_imagenet", "fusion_finetuned"),
                                 ("Embeddings ImageNet (backbones gelés)", "Embeddings après fine-tuning complet")):
        rows = summary[summary["model"] == source].copy()
        if rows.empty:
            continue
        rows["size"] = rows["strategy"].str.count(r"\+") + 1
        values = 100 * rows["test_accuracy_mean"]
        span = values.max() - values.min()
        ax.set_ylim(values.min() - 0.15 * span, values.max() + 0.25 * span)
        best_points = []
        for size, group in rows.groupby("size"):
            jitter = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else [0.0]
            ax.scatter(size + np.asarray(jitter), 100 * group["test_accuracy_mean"], s=26, color="#86b6ef",
                       edgecolor="white", linewidth=0.6, zorder=2)
            top = group.loc[group["test_accuracy_mean"].idxmax()]
            best_points.append((size, 100 * top["test_accuracy_mean"], subset_label(top["strategy"])))
        xs, ys, labels = zip(*best_points)
        ax.plot(xs, ys, color="#104281", linewidth=1.4, marker="o", markersize=5, zorder=3)
        for x, y, label in best_points:
            ax.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5,
                        color=INK_SECONDARY)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Nombre de backbones fusionnés")
        ax.set_xlim(0.6, 4.5)
        ax.set_ylabel("Accuracy sur le test (%)")
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "fusion_gain")


def plot_efficiency(summary):
    path = BENCHMARK_DIR / "efficiency.csv"
    if not path.exists():
        return
    efficiency = pd.read_csv(path).set_index("model")
    entries = [(name, "full_finetuning") for name in BACKBONES] + [(SCRATCH_MODEL, "from_scratch"),
                                                                    ("fusion_finetuned", ALL_MEMBERS)]
    axis_ticks = {"params_m": [1, 2, 5, 10, 20, 50, 100], "cpu_latency_ms": [10, 20, 50, 100, 200, 300]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, column, label in zip(axes, ("params_m", "cpu_latency_ms"),
                                 ("Nombre de paramètres (millions)", "Latence CPU, batch de 1 (ms)")):
        label_offsets = {"fusion_finetuned": ((-8, -3), "right")}
        close_pair = sorted(("efficientnetv2_s", "seresnet50"), key=lambda name: efficiency.loc[name, column])
        label_offsets[close_pair[0]] = ((-7, 5), "right")
        label_offsets[close_pair[1]] = ((7, -10), "left")
        for model, strategy in entries:
            row = row_of(summary, model, strategy)
            if row is None or model not in efficiency.index:
                continue
            x = efficiency.loc[model, column]
            y = 100 * row["test_accuracy_mean"]
            ax.scatter(x, y, s=46, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], edgecolor="white",
                       linewidth=0.8, zorder=3)
            offset, align = label_offsets.get(model, ((7, -3), "left"))
            ax.annotate(MODEL_LABELS[model], (x, y), xytext=offset, textcoords="offset points", fontsize=7,
                        color=INK_SECONDARY, ha=align)
        ax.set_xscale("log")
        ticks = axis_ticks[column]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(ticks[0] * 0.8, ticks[-1] * 1.25)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Accuracy sur le test (%)")
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "efficiency_tradeoff")


def data_efficiency_entries(summary):
    pattern = summary["strategy"].str.extract(r"^(.+)_frac(\d+)$")
    reduced = summary.assign(base=pattern[0], percent=pd.to_numeric(pattern[1])).dropna(subset=["base"])
    entries = {}
    for (model, base), group in reduced.groupby(["model", "base"]):
        full = row_of(summary, model, base)
        points = [(int(row.percent), row.test_accuracy_mean, row.test_accuracy_std) for row in group.itertuples()]
        if full is not None:
            points.append((100, full["test_accuracy_mean"], full["test_accuracy_std"]))
        entries[(model, base)] = sorted(points)
    return entries


def plot_data_efficiency(summary, full_train_images):
    entries = data_efficiency_entries(summary)
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for (model, base), points in entries.items():
        percents, means, stds = (np.array(values, dtype=float) for values in zip(*points))
        label = MODEL_LABELS[model] if model == SCRATCH_MODEL else f"{MODEL_LABELS[model]} ({STRATEGY_LABELS[base].lower()})"
        ax.errorbar(percents, 100 * means, yerr=100 * np.nan_to_num(stds), color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4, capsize=2, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.6, label=label)
    ticks = sorted({p for points in entries.values() for p, _, _ in points})
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{p} %\n({round(p / 100 * full_train_images)})" for p in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Part des images d'apprentissage utilisées (nombre d'images)")
    ax.set_ylabel("Accuracy sur le test (%)")
    ax.legend(loc="lower right")
    save_figure(fig, "data_efficiency")


def table_data_efficiency(summary):
    entries = data_efficiency_entries(summary)
    if not entries:
        return
    lines = []
    for (model, base), points in entries.items():
        name = MODEL_LABELS[model] if model == SCRATCH_MODEL else f"{MODEL_LABELS[model]} ({STRATEGY_LABELS[base].lower()})"
        lines.append(" & ".join([name] + [percent(mean, std) for _, mean, std in points]) + " \\\\")
    write_table("results_data_efficiency", lines)


def robustness_entries():
    return [(name, "full_finetuning") for name in BACKBONES] + [(SCRATCH_MODEL, "from_scratch"),
                                                                ("fusion_finetuned", ALL_MEMBERS)]


def plot_robustness():
    path = BENCHMARK_DIR / "robustness.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    clean = frame[frame["corruption"] == "clean"]
    corruptions = list(CORRUPTION_LABELS)
    fig, axes = plt.subplots(1, len(corruptions), figsize=(7.0, 2.5), sharey=True)
    for ax, corruption in zip(axes, corruptions):
        subset = frame[frame["corruption"] == corruption]
        values = sorted(subset["value"].unique(), reverse=corruption in ("darkening", "jpeg"))
        ticks = ["0"] + [f"{v:g}".replace(".", ",") for v in values]
        for model, strategy in robustness_entries():
            base = clean[(clean["model"] == model) & (clean["strategy"] == strategy)]["accuracy"].mean()
            levels = [subset[(subset["model"] == model) & (subset["strategy"] == strategy) & (subset["level"] == level)]
                      ["accuracy"].mean() for level in (1, 2, 3)]
            ax.plot(range(4), 100 * np.array([base] + levels), color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                    markersize=4.5, linewidth=1.3, markeredgecolor="white", markeredgewidth=0.6,
                    label=MODEL_LABELS[model])
        ax.set_xticks(range(4), ticks, fontsize=7)
        ax.set_title(CORRUPTION_LABELS[corruption], fontsize=8)
        ax.set_xlabel(CORRUPTION_UNITS[corruption], fontsize=7.5)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.4)
    save_figure(fig, "robustness")


def table_robustness():
    path = BENCHMARK_DIR / "robustness.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    entries = robustness_entries() + [("late_fusion", ALL_MEMBERS)]
    lines = []
    scores = {}
    for model, strategy in entries:
        subset = frame[(frame["model"] == model) & (frame["strategy"] == strategy)]
        if subset.empty:
            continue
        clean = subset[subset["corruption"] == "clean"]["accuracy"].mean()
        per_corruption = [subset[subset["corruption"] == c]["accuracy"].mean() for c in CORRUPTION_LABELS]
        mean_corrupted = float(np.mean(per_corruption))
        scores[model] = (clean, per_corruption, mean_corrupted)
    best_mean = max(score[2] for score in scores.values())
    best_drop = min(score[0] - score[2] for score in scores.values())
    for model, (clean, per_corruption, mean_corrupted) in scores.items():
        drop = clean - mean_corrupted
        cells = [MODEL_LABELS[model], fr(100 * clean)] + [fr(100 * v) for v in per_corruption]
        cells.append(bold_if(fr(100 * mean_corrupted), same_display(mean_corrupted, best_mean)))
        cells.append(bold_if(fr(100 * drop), same_display(drop, best_drop)))
        lines.append(" & ".join(cells) + " \\\\")
    write_table("results_robustness", lines)


def table_efficiency(summary):
    path = BENCHMARK_DIR / "efficiency.csv"
    if not path.exists():
        return
    efficiency = pd.read_csv(path).set_index("model")
    lines = []
    for model in BACKBONES + [SCRATCH_MODEL, "fusion_finetuned"]:
        if model not in efficiency.index:
            continue
        row = efficiency.loc[model]
        lines.append(" & ".join([
            MODEL_LABELS[model],
            fr(row["params_m"], 2),
            fr(row["size_mb"], 1),
            fr(row["gflops"], 2),
            fr(row["gpu_latency_ms"], 2),
            f"{row['gpu_throughput']:.0f}",
            fr(row["cpu_latency_ms"], 1),
        ]) + " \\\\")
    write_table("results_efficiency", lines)


def export_csv(runs, summary):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["dir"]).to_csv(TABLES_DIR / "all_runs.csv", index=False)
    summary.to_csv(TABLES_DIR / "summary.csv", index=False)


def main():
    use_report_style()
    runs = load_runs()
    summary = summarize(runs)
    export_csv(runs, summary)
    table_main(summary)
    table_strategies(summary)
    table_per_class(runs, summary)
    table_significance(runs, summary)
    table_fusion(summary)
    table_efficiency(summary)
    table_robustness()
    for strategy in STRATEGIES:
        models = BACKBONES + ([SCRATCH_MODEL] if strategy == "full_finetuning" else [])
        plot_learning_curves(runs, strategy, models, f"learning_curves_{strategy}")
    plot_confusions(runs, summary)
    plot_rocs(runs, summary)
    plot_strategy_comparison(summary)
    plot_fusion_gain(summary)
    plot_efficiency(summary)
    plot_robustness()
    splits = load_splits()
    full_train_images = round(np.mean([len(fold_frames(splits, fold)[0]) for fold in range(N_FOLDS)]))
    plot_data_efficiency(summary, full_train_images)
    table_data_efficiency(summary)
    print(summary[["model", "strategy", "folds", "test_accuracy_mean", "test_accuracy_std", "test_f1_mean",
                   "test_auc_mean", "total_seconds_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
