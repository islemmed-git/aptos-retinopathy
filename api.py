"""
API (service web) pour le modele de retinopathie diabetique, avec FastAPI.

Un autre programme (appli, site, logiciel...) envoie une image de fond d'oeil
et recoit en retour un diagnostic au format JSON.

Lancement :
    .\.venv\Scripts\python.exe -m uvicorn api:app --reload
Puis :
    - documentation interactive : http://127.0.0.1:8000/docs
    - point d'acces prediction   : POST http://127.0.0.1:8000/predict
"""
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from src import config
from src.model import build_model
from src.preprocess import circle_crop, apply_clahe
from src.dataset import get_transforms

# ---------------------------------------------------------------------------
# 1. Charger le modele UNE SEULE FOIS au demarrage du serveur
# ---------------------------------------------------------------------------
ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
model = build_model(ckpt["model_name"]).to(config.DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tfm = get_transforms(train=False)

# ---------------------------------------------------------------------------
# 2. Creer l'application API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Retinopathie Diabetique",
    description="Envoie une image de fond d'oeil, recois le stade predit (0-4).",
    version="1.0",
)


@app.get("/")
def accueil():
    """Page d'info : verifie que l'API tourne."""
    return {
        "message": "API de detection de retinopathie diabetique",
        "modele": ckpt["model_name"],
        "device": str(config.DEVICE),
        "stades": config.CLASS_NAMES,
        "utilisation": "POST une image sur /predict",
    }


def _pretraiter(image_bytes):
    """bytes d'image -> tenseur pret pour le modele (meme pipeline qu'a l'entrainement)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)        # -> BGR
    if img is None:
        raise HTTPException(status_code=400,
                            detail="Fichier image invalide ou illisible.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = circle_crop(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                     interpolation=cv2.INTER_AREA)
    return tfm(Image.fromarray(img)).unsqueeze(0).to(config.DEVICE)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recoit une image de fond d'oeil, renvoie le diagnostic en JSON."""
    image_bytes = await file.read()
    tensor = _pretraiter(image_bytes)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    pred = int(probs.argmax())
    return {
        "stade": pred,
        "diagnostic": config.CLASS_NAMES[pred],
        "confiance": round(float(probs[pred]), 4),
        "probabilites": {
            config.CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(config.NUM_CLASSES)
        },
    }
