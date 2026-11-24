import argparse
import hashlib
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    CACHE_SIZE,
    CLASSES,
    CLEANING_REPORT,
    DATA_DIR,
    KAGGLE_DATASET,
    N_FOLDS,
    NEAR_DUPLICATE_THRESHOLD,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    SPLIT_FILE,
    STATS_FILE,
    TEST_SIZE,
)
from src.utils import save_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download_dataset():
    archive = DATA_DIR / "sugarcane-leaf-disease-dataset.zip"
    if not archive.exists():
        url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
        print(f"Downloading {url}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zipped:
        images = [name for name in zipped.namelist() if name.lower().endswith(tuple(IMAGE_SUFFIXES))]
        zipped.extractall(RAW_DIR, members=images)
    print(f"{len(images)} images extracted to {RAW_DIR}")


def list_raw_images():
    rows = []
    for label, class_name in enumerate(CLASSES):
        for path in sorted((RAW_DIR / class_name).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({"source": path, "class_name": class_name, "label": label})
    return pd.DataFrame(rows)


def read_image_info(path):
    try:
        with Image.open(path) as image:
            image.load()
            return image.width, image.height, image.mode, True
    except OSError:
        return 0, 0, "", False


def file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def thumbnail_signature(path):
    with Image.open(path) as image:
        thumbnail = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    values = np.asarray(thumbnail, dtype=np.float32).ravel()
    values = (values - values.mean()) / (values.std() + 1e-6)
    return values / np.sqrt(values.size)


def find_near_duplicates(frame, threshold):
    signatures = np.stack([thumbnail_signature(path) for path in frame["source"]])
    similarity = signatures @ signatures.T
    keep = np.ones(len(frame), dtype=bool)
    pairs = []
    for i in range(len(frame)):
        if not keep[i]:
            continue
        for j in np.flatnonzero(similarity[i, i + 1:] >= threshold) + i + 1:
            if keep[j]:
                keep[j] = False
                pairs.append({
                    "kept": frame.at[i, "source"].name,
                    "removed": frame.at[j, "source"].name,
                    "kept_class": frame.at[i, "class_name"],
                    "removed_class": frame.at[j, "class_name"],
                    "similarity": round(float(similarity[i, j]), 4),
                })
    return keep, pairs


def assign_splits(frame):
    frame = frame.copy()
    indices = np.arange(len(frame))
    dev_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=frame["label"], random_state=SEED
    )
    frame["split"] = "dev"
    frame.loc[test_idx, "split"] = "test"
    frame["fold"] = -1
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_positions) in enumerate(folds.split(dev_idx, frame.loc[dev_idx, "label"])):
        frame.loc[dev_idx[val_positions], "fold"] = fold
    return frame


def cached_name(source):
    return source.stem.replace(" (", "_").replace(")", "").replace(" ", "_") + ".png"


def cache_images(frame):
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    paths = []
    for row in frame.itertuples():
        relative = f"{row.class_name}/{cached_name(row.source)}"
        target = PROCESSED_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row.source) as image:
            image.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.Resampling.LANCZOS).save(target)
        paths.append(relative)
    return paths


def channel_statistics(paths):
    total = np.zeros(3)
    total_squared = np.zeros(3)
    pixels = 0
    for path in paths:
        with Image.open(PROCESSED_DIR / path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        total += array.sum(axis=(0, 1))
        total_squared += (array ** 2).sum(axis=(0, 1))
        pixels += array.shape[0] * array.shape[1]
    mean = total / pixels
    std = np.sqrt(total_squared / pixels - mean ** 2)
    return mean.round(4).tolist(), std.round(4).tolist()


def class_counts(frame):
    return {name: int((frame["class_name"] == name).sum()) for name in CLASSES}


def split_counts(frame):
    counts = {"test": class_counts(frame[frame["split"] == "test"])}
    dev = frame[frame["split"] == "dev"]
    for fold in range(N_FOLDS):
        counts[f"fold_{fold}_train"] = class_counts(dev[dev["fold"] != fold])
        counts[f"fold_{fold}_val"] = class_counts(dev[dev["fold"] == fold])
    return counts


def main():
    parser = argparse.ArgumentParser(description="Clean, split and cache the sugarcane leaf dataset")
    parser.add_argument("--download", action="store_true", help="download the dataset from Kaggle first")
    args = parser.parse_args()

    if args.download or not RAW_DIR.exists():
        download_dataset()

    frame = list_raw_images()
    raw_counts = class_counts(frame)
    print(f"Raw images: {len(frame)}  {raw_counts}")

    info = [read_image_info(path) for path in frame["source"]]
    frame["width"] = [item[0] for item in info]
    frame["height"] = [item[1] for item in info]
    frame["mode"] = [item[2] for item in info]
    corrupted = frame[[not item[3] for item in info]]
    frame = frame[[item[3] for item in info]].reset_index(drop=True)
    print(f"Unreadable images: {len(corrupted)}")

    frame["md5"] = [file_md5(path) for path in frame["source"]]
    duplicated = frame.duplicated("md5", keep="first")
    duplicate_groups = int(frame[frame.duplicated("md5", keep=False)]["md5"].nunique())
    frame = frame[~duplicated].reset_index(drop=True)
    after_exact = class_counts(frame)
    print(f"Exact duplicates removed: {int(duplicated.sum())} ({duplicate_groups} groups)  {after_exact}")

    keep, near_pairs = find_near_duplicates(frame, NEAR_DUPLICATE_THRESHOLD)
    frame = frame[keep].reset_index(drop=True)
    after_near = class_counts(frame)
    cross_class = sum(pair["kept_class"] != pair["removed_class"] for pair in near_pairs)
    print(f"Near duplicates removed: {len(near_pairs)} ({cross_class} across classes)  {after_near}")

    frame = assign_splits(frame)
    frame["path"] = cache_images(frame)
    frame["source"] = [path.relative_to(RAW_DIR).as_posix() for path in frame["source"]]
    columns = ["path", "source", "class_name", "label", "width", "height", "split", "fold", "md5"]
    frame[columns].to_csv(SPLIT_FILE, index=False)

    mean, std = channel_statistics(frame.loc[frame["split"] == "dev", "path"])
    save_json({"mean": mean, "std": std, "computed_on": "dev (train + validation)"}, STATS_FILE)

    report = {
        "raw_images": int(sum(raw_counts.values())),
        "raw_counts": raw_counts,
        "corrupted_images": int(len(corrupted)),
        "exact_duplicates_removed": int(duplicated.sum()),
        "exact_duplicate_groups": duplicate_groups,
        "after_exact_dedup": after_exact,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicates_removed": len(near_pairs),
        "near_duplicates_cross_class": int(cross_class),
        "near_duplicate_pairs": near_pairs,
        "final_images": int(len(frame)),
        "final_counts": after_near,
        "splits": split_counts(frame),
        "dev_mean": mean,
        "dev_std": std,
    }
    save_json(report, CLEANING_REPORT)
    test_size = int((frame["split"] == "test").sum())
    print(f"Final dataset: {len(frame)} images, test = {test_size}, development = {len(frame) - test_size}")
    print(f"Mean = {mean}, std = {std}")


if __name__ == "__main__":
    main()
