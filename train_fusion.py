import argparse
import itertools
import math

import numpy as np
import torch

from src.config import BATCH_SIZE, CLASSES, EPOCHS, N_FOLDS, OUTPUT_DIR, PRETRAINED_MODELS, SEED
from src.data import BatchLoader, fold_frames, load_splits
from src.engine import fit, predict
from src.metrics import classification_metrics
from src.models import FusionHead, build_model
from src.utils import format_duration, get_device, load_json, run_dir, save_json, seed_everything

FEATURES_DIR = OUTPUT_DIR / "features"
SOURCES = {"imagenet": "fusion_imagenet", "finetuned": "fusion_finetuned"}


class EmbeddingLoader:
    def __init__(self, embeddings, labels, device, train=False, batch_size=BATCH_SIZE, seed=0):
        self.embeddings = embeddings.to(device)
        self.labels = labels.to(device)
        self.train = train
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        if self.train:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        count = len(self.labels)
        order = torch.randperm(count, generator=self.generator) if self.train else torch.arange(count)
        order = order.to(self.labels.device)
        for step in range(len(self)):
            index = order[step * self.batch_size:(step + 1) * self.batch_size]
            yield self.embeddings[index], self.labels[index]


def load_member(name, source, fold):
    model = build_model(name)
    if source == "finetuned":
        checkpoint = run_dir(name, "full_finetuning", fold) / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


@torch.no_grad()
def extract(model, frame, device):
    model.eval()
    loader = BatchLoader(frame, device)
    outputs = []
    for images, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs.append(model.embed(images).float().cpu())
    return torch.cat(outputs), loader.labels


def cached_embeddings(name, source, fold, frames, device):
    path = FEATURES_DIR / source / name / f"fold_{fold}.pt"
    if path.exists():
        return torch.load(path)
    model = load_member(name, source, fold).to(device)
    cache = {}
    for split, frame in frames.items():
        embeddings, labels = extract(model, frame, device)
        cache[split] = {"embeddings": embeddings, "labels": labels}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    del model
    torch.cuda.empty_cache()
    return cache


def subset_key(members):
    return "+".join(members)


def train_head(members, source, fold, caches, device, epochs):
    output = run_dir(SOURCES[source], subset_key(members), fold)
    if (output / "summary.json").exists():
        return load_json(output / "summary.json")
    seed_everything(SEED + fold)

    def stacked(split):
        embeddings = torch.cat([caches[name][split]["embeddings"] for name in members], dim=1)
        return embeddings, caches[members[0]][split]["labels"]

    train_x, train_y = stacked("train")
    val_x, val_y = stacked("val")
    test_x, test_y = stacked("test")
    train_loader = EmbeddingLoader(train_x, train_y, device, train=True, seed=SEED + fold)
    val_loader = EmbeddingLoader(val_x, val_y, device)
    test_loader = EmbeddingLoader(test_x, test_y, device)

    dims = [caches[name]["train"]["embeddings"].shape[1] for name in members]
    head = FusionHead(dims, len(CLASSES)).to(device)
    history, best, timing = fit(head, train_loader, val_loader, device, epochs=epochs, log=lambda line: None)
    val_probs, val_labels = predict(head, val_loader, device)
    test_probs, test_labels = predict(head, test_loader, device)

    output.mkdir(parents=True, exist_ok=True)
    history.to_csv(output / "history.csv", index=False)
    np.savez_compressed(output / "predictions.npz", val_probs=val_probs, val_labels=val_labels,
                        test_probs=test_probs, test_labels=test_labels)
    torch.save(head.state_dict(), output / "head.pt")
    summary = {
        "model": SOURCES[source],
        "strategy": subset_key(members),
        "members": list(members),
        "fold": fold,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "embedding_dim": int(sum(dims)),
        "trainable_params": int(sum(p.numel() for p in head.parameters())),
        "train_seconds": timing["train_seconds"],
        "total_seconds": timing["total_seconds"],
        "val": classification_metrics(val_labels, val_probs),
        "test": classification_metrics(test_labels, test_probs),
    }
    save_json(summary, output / "summary.json")
    return summary


def late_fusion(fold, members):
    runs = [np.load(run_dir(name, "full_finetuning", fold) / "predictions.npz") for name in members]
    summaries = [load_json(run_dir(name, "full_finetuning", fold) / "summary.json") for name in members]
    val_probs = np.mean([run["val_probs"] for run in runs], axis=0)
    test_probs = np.mean([run["test_probs"] for run in runs], axis=0)
    output = run_dir("late_fusion", subset_key(members), fold)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", val_probs=val_probs, val_labels=runs[0]["val_labels"],
                        test_probs=test_probs, test_labels=runs[0]["test_labels"])
    summary = {
        "model": "late_fusion",
        "strategy": subset_key(members),
        "members": list(members),
        "fold": fold,
        "total_params": int(sum(s["total_params"] for s in summaries)),
        "trainable_params": 0,
        "members_seconds": float(sum(s["total_seconds"] for s in summaries)),
        "val": classification_metrics(runs[0]["val_labels"], val_probs),
        "test": classification_metrics(runs[0]["test_labels"], test_probs),
    }
    save_json(summary, output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Feature-level and decision-level fusion of the backbones")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = get_device()
    splits = load_splits()
    names = list(PRETRAINED_MODELS)
    subsets = [list(combo) for size in range(1, len(names) + 1) for combo in itertools.combinations(names, size)]

    for fold in args.folds:
        train_frame, val_frame, test_frame = fold_frames(splits, fold)
        frames = {"train": train_frame, "val": val_frame, "test": test_frame}
        for source in args.sources:
            caches = {name: cached_embeddings(name, source, fold, frames, device) for name in names}
            for members in subsets:
                summary = train_head(members, source, fold, caches, device, args.epochs)
                print(f"fold {fold} | {source:9s} | {subset_key(members):60s} | test acc "
                      f"{summary['test']['accuracy']:.4f} f1 {summary['test']['f1']:.4f} "
                      f"({format_duration(summary['total_seconds'])})", flush=True)
        for members in [names] + [list(c) for c in itertools.combinations(names, 2)]:
            summary = late_fusion(fold, members)
            print(f"fold {fold} | late fusion | {subset_key(members):60s} | test acc "
                  f"{summary['test']['accuracy']:.4f} f1 {summary['test']['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
