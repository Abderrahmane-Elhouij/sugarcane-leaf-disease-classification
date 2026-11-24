import argparse
import io
import math
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
from torchvision.transforms.v2 import functional as TF

from src.config import BENCHMARK_DIR, CLASSES, IMAGE_SIZE, N_FOLDS, PRETRAINED_MODELS, SCRATCH_MODEL, SEED, STRATEGIES
from src.data import fold_frames, load_images, load_splits, resize_for_eval
from src.engine import autocast
from src.models import FusionClassifier, FusionHead, build_model
from src.utils import get_device, run_dir

ALL_MEMBERS = "+".join(PRETRAINED_MODELS)


def gaussian_noise(images, sigma, generator):
    return (images + sigma * torch.randn(images.shape, generator=generator)).clamp(0, 1)


def gaussian_blur(images, sigma, generator):
    size = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(images, kernel_size=[size, size], sigma=[sigma, sigma])


def darkening(images, factor, generator):
    return (images * factor).clamp(0, 1)


def jpeg(images, quality, generator):
    compressed = []
    for image in images:
        buffer = io.BytesIO()
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            compressed.append(torch.from_numpy(np.array(decoded.convert("RGB"))).permute(2, 0, 1))
    return torch.stack(compressed).float() / 255.0


def occlusion(images, fraction, generator):
    images = images.clone()
    count, _, height, width = images.shape
    side = int(round(fraction * height))
    rows = torch.randint(0, height - side + 1, (count,), generator=generator)
    cols = torch.randint(0, width - side + 1, (count,), generator=generator)
    for i in range(count):
        images[i, :, rows[i]:rows[i] + side, cols[i]:cols[i] + side] = 0.5
    return images


CORRUPTIONS = [
    ("gaussian_noise", gaussian_noise, [0.04, 0.08, 0.12]),
    ("gaussian_blur", gaussian_blur, [1.0, 2.0, 3.0]),
    ("darkening", darkening, [0.7, 0.5, 0.3]),
    ("jpeg", jpeg, [30, 15, 5]),
    ("occlusion", occlusion, [0.25, 0.35, 0.45]),
]


def to_uint8(images):
    return (images * 255).round().clamp(0, 255).to(torch.uint8)


def corrupted_test_sets(test_frame):
    images, labels = load_images(test_frame)
    clean = resize_for_eval(images.float() / 255.0)
    sets = [("clean", 0, 0.0, to_uint8(clean))]
    for index, (name, function, levels) in enumerate(CORRUPTIONS):
        for level, value in enumerate(levels, start=1):
            generator = torch.Generator().manual_seed(SEED + 100 * index + level)
            sets.append((name, level, value, to_uint8(function(clean, value, generator))))
    return sets, labels


@torch.no_grad()
def predict_probs(model, images, device, batch_size=64):
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device).float() / 255.0
        with autocast(device):
            logits = model(batch).float()
        outputs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(outputs)


def load_single(name, strategy, fold):
    model = build_model(name)
    model.load_state_dict(torch.load(run_dir(name, strategy, fold) / "best.pt", map_location="cpu"))
    return model


def load_fusion(fold):
    members = {name: load_single(name, "full_finetuning", fold) for name in PRETRAINED_MODELS}
    head = FusionHead([member.embedding_dim for member in members.values()], len(CLASSES))
    head.load_state_dict(torch.load(run_dir("fusion_finetuned", ALL_MEMBERS, fold) / "head.pt", map_location="cpu"))
    return FusionClassifier(members, head)


def robustness(device, folds):
    splits = load_splits()
    configs = [(name, strategy) for name in PRETRAINED_MODELS for strategy in STRATEGIES]
    configs += [(SCRATCH_MODEL, "from_scratch"), ("fusion_finetuned", ALL_MEMBERS)]
    rows = []
    for fold in folds:
        _, _, test_frame = fold_frames(splits, fold)
        sets, labels = corrupted_test_sets(test_frame)
        fft_probs = {}
        for model_name, strategy in configs:
            if model_name == "fusion_finetuned":
                model = load_fusion(fold)
            else:
                model = load_single(model_name, strategy, fold)
            model = model.to(device)
            for corruption, level, value, images in sets:
                probs = predict_probs(model, images, device)
                accuracy = (probs.argmax(1) == labels).float().mean().item()
                rows.append({"model": model_name, "strategy": strategy, "fold": fold, "corruption": corruption,
                             "level": level, "value": value, "accuracy": accuracy})
                if strategy == "full_finetuning":
                    fft_probs.setdefault((corruption, level, value), []).append(probs)
            print(f"robustness fold {fold}: {model_name} / {strategy} done", flush=True)
            del model
            torch.cuda.empty_cache()
        for (corruption, level, value), probs in fft_probs.items():
            accuracy = (torch.stack(probs).mean(0).argmax(1) == labels).float().mean().item()
            rows.append({"model": "late_fusion", "strategy": ALL_MEMBERS, "fold": fold, "corruption": corruption,
                         "level": level, "value": value, "accuracy": accuracy})
    frame = pd.DataFrame(rows)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BENCHMARK_DIR / "robustness.csv", index=False)
    return frame


def count_gflops(model):
    model = model.cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    return counter.get_total_flops() / 1e9


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency_ms(model, device, batch_size, warmup, repeats):
    model = model.to(device).eval()
    images = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with autocast(device):
        for _ in range(warmup):
            model(images)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(images)
        synchronize(device)
    return (time.perf_counter() - start) / repeats * 1000.0


def efficiency(device, cpu_threads):
    entries = [(name, build_model(name)) for name in list(PRETRAINED_MODELS) + [SCRATCH_MODEL]]
    members = {name: build_model(name) for name in PRETRAINED_MODELS}
    head = FusionHead([member.embedding_dim for member in members.values()], len(CLASSES))
    entries.append(("fusion_finetuned", FusionClassifier(members, head)))
    rows = []
    torch.set_num_threads(cpu_threads)
    for name, model in entries:
        params = sum(p.numel() for p in model.parameters())
        gflops = count_gflops(model)
        cpu_latency = latency_ms(model, torch.device("cpu"), 1, warmup=5, repeats=30)
        gpu_latency = latency_ms(model, device, 1, warmup=30, repeats=200)
        gpu_batch = latency_ms(model, device, 32, warmup=10, repeats=50)
        rows.append({
            "model": name,
            "params_m": params / 1e6,
            "size_mb": params * 4 / 2 ** 20,
            "gflops": gflops,
            "gpu_latency_ms": gpu_latency,
            "gpu_throughput": 32 * 1000.0 / gpu_batch,
            "cpu_latency_ms": cpu_latency,
        })
        print(f"efficiency: {name:18s} {params / 1e6:6.2f}M  {gflops:6.2f} GFLOPs  GPU {gpu_latency:6.2f} ms  "
              f"CPU {cpu_latency:7.2f} ms  {32 * 1000.0 / gpu_batch:7.1f} img/s", flush=True)
        model.cpu()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BENCHMARK_DIR / "efficiency.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Efficiency and robustness benchmarks")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    device = get_device()
    if not args.skip_efficiency:
        efficiency(device, args.cpu_threads)
    if not args.skip_robustness:
        robustness(device, args.folds)


if __name__ == "__main__":
    main()
