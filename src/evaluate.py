"""
Evaluation du modele entraine sur le jeu de TEST.

Produit :
  - accuracy
  - quadratic weighted kappa (la metrique officielle d'APTOS)
  - rapport de classification (precision / recall / f1 par classe)
  - matrice de confusion (sauvegardee en PNG)

Usage : python -m src.evaluate
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             classification_report, confusion_matrix)

from . import config
from .dataset import get_loaders
from .model import build_model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(config.DEVICE)
        out = model(imgs)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(labels), np.array(preds)


def main():
    _, _, test_loader = get_loaders()

    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = build_model(ckpt["model_name"]).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Modele charge : {ckpt['model_name']} (kappa val = {ckpt['kappa']:.3f})")

    y_true, y_pred = predict(model, test_loader)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"\n=== RESULTATS SUR LE TEST ===")
    print(f"Accuracy                : {acc:.4f}")
    print(f"Quadratic weighted kappa: {kappa:.4f}\n")
    print(classification_report(y_true, y_pred,
                                target_names=config.CLASS_NAMES, digits=3))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(config.NUM_CLASSES),
                yticklabels=range(config.NUM_CLASSES))
    plt.xlabel("Prediction")
    plt.ylabel("Verite terrain")
    plt.title(f"Matrice de confusion (acc={acc:.3f}, kappa={kappa:.3f})")
    out = config.OUTPUT_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nMatrice de confusion sauvegardee : {out}")


if __name__ == "__main__":
    main()
