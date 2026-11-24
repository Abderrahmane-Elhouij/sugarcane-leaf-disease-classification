import argparse
import time

import numpy as np
import torch

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LR_NEW_LAYERS,
    LR_PRETRAINED,
    N_FOLDS,
    PRETRAINED_MODELS,
    SCRATCH_MODEL,
    SCRATCH_STRATEGY,
    SEED,
    STRATEGIES,
    WEIGHT_DECAY,
)
from src.data import BatchLoader, fold_frames, load_splits
from src.engine import fit, predict
from src.metrics import classification_metrics
from src.models import apply_strategy, build_model
from src.utils import count_parameters, format_duration, get_device, run_dir, save_json, seed_everything


def planned_runs(models, strategies):
    for model_name in models:
        if model_name == SCRATCH_MODEL:
            yield model_name, SCRATCH_STRATEGY
            continue
        for strategy in strategies:
            yield model_name, strategy


def subsample(frame, fraction, seed):
    if fraction >= 1.0:
        return frame
    return frame.groupby("label").sample(frac=fraction, random_state=seed).reset_index(drop=True)


def train_run(model_name, strategy, fold, splits, device, epochs, fraction=1.0):
    run_name = strategy if fraction >= 1.0 else f"{strategy}_frac{round(100 * fraction)}"
    output = run_dir(model_name, run_name, fold)
    if (output / "summary.json").exists():
        print(f"[skip] {model_name} / {run_name} / fold {fold} already trained")
        return
    print(f"\n=== {model_name} | {run_name} | fold {fold} ===", flush=True)
    seed_everything(SEED + fold)
    train_frame, val_frame, test_frame = fold_frames(splits, fold)
    train_frame = subsample(train_frame, fraction, SEED + fold)
    train_loader = BatchLoader(train_frame, device, train=True, seed=SEED + fold)
    val_loader = BatchLoader(val_frame, device)
    test_loader = BatchLoader(test_frame, device)

    model = apply_strategy(build_model(model_name), strategy).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best, timing = fit(model, train_loader, val_loader, device, epochs=epochs, log=lambda line: print(line, flush=True))
    peak_memory = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0

    val_probs, val_labels = predict(model, val_loader, device)
    test_probs, test_labels = predict(model, test_loader, device)
    val_metrics = classification_metrics(val_labels, val_probs)
    test_metrics = classification_metrics(test_labels, test_probs)

    output.mkdir(parents=True, exist_ok=True)
    history.to_csv(output / "history.csv", index=False)
    np.savez_compressed(
        output / "predictions.npz",
        val_probs=val_probs,
        val_labels=val_labels,
        test_probs=test_probs,
        test_labels=test_labels,
        test_paths=np.array(test_frame["path"]),
    )
    torch.save(model.state_dict(), output / "best.pt")
    summary = {
        "model": model_name,
        "strategy": run_name,
        "fold": fold,
        "fraction": fraction,
        "train_images": len(train_frame),
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_seconds": timing["train_seconds"],
        "total_seconds": timing["total_seconds"],
        "peak_memory_mb": peak_memory,
        "batch_size": BATCH_SIZE,
        "lr_new_layers": LR_NEW_LAYERS,
        "lr_pretrained": LR_PRETRAINED,
        "weight_decay": WEIGHT_DECAY,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(summary, output / "summary.json")
    print(
        f"  best epoch {best['epoch']}  val acc {val_metrics['accuracy']:.4f}  "
        f"test acc {test_metrics['accuracy']:.4f}  f1 {test_metrics['f1']:.4f}  "
        f"auc {test_metrics['auc']:.4f}  time {format_duration(timing['total_seconds'])}",
        flush=True,
    )
    del model, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    all_models = list(PRETRAINED_MODELS) + [SCRATCH_MODEL]
    parser = argparse.ArgumentParser(description="Train the models with cross-validation")
    parser.add_argument("--models", nargs="+", default=all_models, choices=all_models)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0],
                        help="fractions of the training fold to use (data-efficiency study)")
    args = parser.parse_args()

    device = get_device()
    splits = load_splits()
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    start = time.perf_counter()
    for fold in args.folds:
        for fraction in args.fractions:
            for model_name, strategy in planned_runs(args.models, args.strategies):
                train_run(model_name, strategy, fold, splits, device, args.epochs, fraction)
    print(f"\nAll runs finished in {format_duration(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
