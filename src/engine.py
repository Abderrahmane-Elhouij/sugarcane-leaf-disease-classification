import math
import time

import pandas as pd
import torch
from torch import nn

from src.config import EPOCHS, LABEL_SMOOTHING, LR_NEW_LAYERS, LR_PRETRAINED, WARMUP_EPOCHS, WEIGHT_DECAY
from src.models import parameter_groups, set_train_mode


def warmup_cosine(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    set_train_mode(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        with autocast(device):
            logits = model(images)
        loss = criterion(logits.float(), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for images, labels in loader:
        with autocast(device):
            logits = model(images).float()
        total_loss += criterion(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    accuracy = (logits.argmax(1) == labels).float().mean().item()
    return total_loss / len(labels), accuracy


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        with autocast(device):
            logits = model(images).float()
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def fit(model, train_loader, val_loader, device, epochs=EPOCHS, log=print):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, LR_NEW_LAYERS, LR_PRETRAINED),
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(epochs * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch),
    )
    history = []
    best = {"epoch": 0, "val_acc": -1.0, "val_loss": float("inf")}
    best_state = None
    train_seconds = 0.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_seconds += time.perf_counter() - epoch_start
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        })
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        log(
            f"  epoch {epoch:02d}/{epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  ({history[-1]['seconds']:.1f}s)"
        )
    total_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    timing = {"train_seconds": train_seconds, "total_seconds": total_seconds}
    return pd.DataFrame(history), best, timing
