"""
Utiliser le modele entraine sur une NOUVELLE image (inference).

Donne le stade predit + le niveau de confiance pour chaque stade.

Usage :
    python -m src.predict --image chemin/vers/une_image.png
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .model import build_model
from .preprocess import load_and_preprocess
from .dataset import get_transforms


@torch.no_grad()
def predict_image(image_path):
    # 1. Charger le modele entraine
    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = build_model(ckpt["model_name"]).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Pretraiter l'image (meme pipeline que l'entrainement)
    img_np = load_and_preprocess(image_path, size=config.IMG_SIZE)
    tensor = get_transforms(train=False)(Image.fromarray(img_np))
    tensor = tensor.unsqueeze(0).to(config.DEVICE)   # ajoute la dim "batch"

    # 3. Prediction -> probabilites
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())

    return pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="chemin de l'image")
    args = parser.parse_args()

    pred, probs = predict_image(args.image)

    print(f"\n=== DIAGNOSTIC PREDIT ===")
    print(f">>> {config.CLASS_NAMES[pred]}  (confiance : {probs[pred]*100:.1f}%)\n")

    print("Detail par stade :")
    for i, (name, p) in enumerate(zip(config.CLASS_NAMES, probs)):
        barre = "#" * int(p * 30)
        marque = " <--" if i == pred else ""
        print(f"  {name:25s} {p*100:5.1f}%  {barre}{marque}")


if __name__ == "__main__":
    main()
