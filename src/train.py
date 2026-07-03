"""
Entrainement du CNN (transfer learning) sur APTOS 2019.

Usage :
    python -m src.train
    python -m src.train --epochs 20 --batch-size 16 --model resnet34
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from . import config
from .dataset import get_loaders, compute_class_weights
from .model import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, train):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, leave=False,
                                 desc="train" if train else "val  "):
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad()

            # AMP (mixed precision) -> indispensable sur 4 Go de VRAM
            with torch.autocast(device_type=config.DEVICE.type,
                                enabled=config.DEVICE.type == "cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return avg_loss, acc, kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()

    config.BATCH_SIZE = args.batch_size
    set_seed(config.SEED)
    print(f"Device : {config.DEVICE} | Modele : {args.model}")

    train_loader, val_loader, _ = get_loaders()

    model = build_model(args.model).to(config.DEVICE)

    # Perte ponderee car le dataset est desequilibre (beaucoup de stade 0)
    weights = compute_class_weights().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=config.DEVICE.type == "cuda")

    history = {"train_loss": [], "val_loss": [],
               "train_kappa": [], "val_kappa": []}
    best_kappa = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_kappa = run_epoch(
            model, train_loader, criterion, optimizer, scaler, train=True)
        va_loss, va_acc, va_kappa = run_epoch(
            model, val_loader, criterion, optimizer, scaler, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_kappa"].append(tr_kappa)
        history["val_kappa"].append(va_kappa)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} kappa {tr_kappa:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} kappa {va_kappa:.3f}")

        if va_kappa > best_kappa:
            best_kappa = va_kappa
            torch.save({"model_state": model.state_dict(),
                        "model_name": args.model,
                        "kappa": va_kappa}, config.MODEL_PATH)
            print(f"  -> nouveau meilleur modele sauvegarde (kappa={va_kappa:.3f})")

    with open(config.OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTermine. Meilleur kappa (val) : {best_kappa:.3f}")
    print(f"Modele sauvegarde : {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
