"""
Grad-CAM : visualise OU le modele regarde pour prendre sa decision.

Tres utile en imagerie medicale : on superpose une carte de chaleur
sur l'image, ce qui permet a un medecin de verifier que le reseau
se concentre bien sur des lesions (et pas sur un artefact).

Usage :
    python -m src.gradcam --image data/train_images/xxxx.png
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from . import config
from .model import build_model
from .preprocess import load_and_preprocess
from .dataset import get_transforms


class GradCAM:
    """Implementation minimaliste de Grad-CAM via des hooks."""

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()

        self.model.zero_grad()
        out[0, class_idx].backward()

        # Poids = moyenne spatiale des gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_target_layer(model, model_name):
    """Derniere couche convolutive selon l'architecture."""
    if model_name.startswith("resnet"):
        return model.layer4[-1]
    if model_name.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"Couche cible inconnue pour {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="chemin de l'image")
    args = parser.parse_args()

    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = build_model(ckpt["model_name"]).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Image pretraitee -> tenseur
    img_np = load_and_preprocess(args.image, size=config.IMG_SIZE)
    tensor = get_transforms(train=False)(Image.fromarray(img_np))
    tensor = tensor.unsqueeze(0).to(config.DEVICE)

    cam_engine = GradCAM(model, get_target_layer(model, ckpt["model_name"]))
    cam, pred = cam_engine(tensor)

    # Superposition heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * img_np + 0.5 * heatmap)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(img_np); ax[0].set_title("Image pretraitee"); ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title(f"Grad-CAM\nPrediction : {config.CLASS_NAMES[pred]}")
    ax[1].axis("off")
    out = config.OUTPUT_DIR / "gradcam.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Prediction : {config.CLASS_NAMES[pred]}")
    print(f"Visualisation sauvegardee : {out}")


if __name__ == "__main__":
    main()
