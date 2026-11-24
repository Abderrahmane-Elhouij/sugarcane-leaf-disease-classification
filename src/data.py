import math

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, decode_image

from src.config import BATCH_SIZE, IMAGE_SIZE, PROCESSED_DIR, SPLIT_FILE


def load_splits():
    return pd.read_csv(SPLIT_FILE)


def fold_frames(splits, fold):
    dev = splits[splits["split"] == "dev"]
    train = dev[dev["fold"] != fold].reset_index(drop=True)
    val = dev[dev["fold"] == fold].reset_index(drop=True)
    test = splits[splits["split"] == "test"].reset_index(drop=True)
    return train, val, test


def load_images(frame):
    images = [decode_image(str(PROCESSED_DIR / path), mode=ImageReadMode.RGB) for path in frame["path"]]
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    return torch.stack(images), labels


def resize_for_eval(images, size=IMAGE_SIZE):
    return F.interpolate(images, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


def grayscale(images):
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


class GpuAugment:
    def __init__(self, size=IMAGE_SIZE, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), degrees=15.0,
                 brightness=0.25, contrast=0.25, saturation=0.15, seed=0):
        self.size = size
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        self.degrees = degrees
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.seed = seed
        self.generator = None

    def uniform(self, n, low, high, device):
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        return low + (high - low) * torch.rand(n, device=device, generator=self.generator)

    def geometric(self, images):
        n, device = images.size(0), images.device
        area = self.uniform(n, *self.scale, device)
        ratio = torch.exp(self.uniform(n, *self.log_ratio, device))
        width = torch.sqrt(area * ratio).clamp(max=1.0)
        height = torch.sqrt(area / ratio).clamp(max=1.0)
        shift_x = (1.0 - width) * self.uniform(n, -1.0, 1.0, device)
        shift_y = (1.0 - height) * self.uniform(n, -1.0, 1.0, device)
        angle = torch.deg2rad(self.uniform(n, -self.degrees, self.degrees, device))
        flip_x = torch.where(self.uniform(n, 0.0, 1.0, device) < 0.5, -1.0, 1.0)
        flip_y = torch.where(self.uniform(n, 0.0, 1.0, device) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.stack([
            torch.stack([cos * width * flip_x, -sin * height * flip_y, shift_x], dim=1),
            torch.stack([sin * width * flip_x, cos * height * flip_y, shift_y], dim=1),
        ], dim=1)
        grid = F.affine_grid(theta, [n, 3, self.size, self.size], align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def photometric(self, images):
        n, device = images.size(0), images.device
        shape = (n, 1, 1, 1)
        brightness = self.uniform(n, 1 - self.brightness, 1 + self.brightness, device).view(shape)
        contrast = self.uniform(n, 1 - self.contrast, 1 + self.contrast, device).view(shape)
        saturation = self.uniform(n, 1 - self.saturation, 1 + self.saturation, device).view(shape)
        images = (images * brightness).clamp(0, 1)
        mean = grayscale(images).mean(dim=(1, 2, 3), keepdim=True)
        images = ((images - mean) * contrast + mean).clamp(0, 1)
        gray = grayscale(images)
        return ((images - gray) * saturation + gray).clamp(0, 1)

    def __call__(self, images):
        return self.photometric(self.geometric(images))


class BatchLoader:
    def __init__(self, frame, device, train=False, batch_size=BATCH_SIZE, seed=0):
        self.images, self.labels = load_images(frame)
        self.paths = list(frame["path"])
        self.device = device
        self.train = train
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.augment = GpuAugment(seed=seed) if train else None

    def __len__(self):
        if self.train:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        count = len(self.labels)
        order = torch.randperm(count, generator=self.generator) if self.train else torch.arange(count)
        for step in range(len(self)):
            index = order[step * self.batch_size:(step + 1) * self.batch_size]
            images = self.images[index].to(self.device, non_blocking=True).float().div_(255.0)
            images = self.augment(images) if self.train else resize_for_eval(images)
            labels = self.labels[index].to(self.device, non_blocking=True)
            yield images, labels
