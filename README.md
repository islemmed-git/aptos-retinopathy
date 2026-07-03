# 👁️ Détection de la rétinopathie diabétique (APTOS 2019)

Classification automatique du **stade de rétinopathie diabétique** (0 à 4)
à partir d'images de **fond d'œil** (rétinographie), par *deep learning*.

<p align="center">
  <img src="assets/retino.png" alt="Fond d'œil avec lésions de rétinopathie diabétique (exsudats, hémorragies)" width="520">
  <br>
  <em>Exemple de rétinographie : exsudats (taches jaunes) et hémorragies (taches rouges sombres),
  signes caractéristiques de la rétinopathie diabétique.</em>
</p>

> Projet pédagogique — pont entre le traitement d'image classique (OpenCV)
> et l'imagerie médicale par IA (transfer learning + interprétabilité).

<p align="center">
  <img src="assets/retino.png" width="420" alt="Exemple d'image de fond d'œil (rétinographie)">
  <br>
  <em>Exemple d'image de fond d'œil : microanévrismes, exsudats et hémorragies visibles.</em>
</p>

---

## 🎯 Pipeline

```
Image fond d'œil
   │
   ├─ 1. Prétraitement OpenCV   (crop circulaire + CLAHE)
   ├─ 2. CNN pré-entraîné        (ResNet / EfficientNet, transfer learning)
   ├─ 3. Évaluation              (accuracy + quadratic weighted kappa)
   └─ 4. Grad-CAM               (carte de chaleur : où le modèle regarde)
```

<p align="center">
  <img src="assets/gradcam.png" width="720" alt="Grad-CAM : image prétraitée et carte de chaleur">
  <br>
  <em>Grad-CAM : à gauche l'image prétraitée (crop + CLAHE), à droite les zones
  qui ont motivé la prédiction du modèle. Il se concentre bien sur les lésions.</em>
</p>

Les 5 stades :
| Label | Stade |
|-------|-------|
| 0 | Pas de rétinopathie |
| 1 | Légère |
| 2 | Modérée |
| 3 | Sévère |
| 4 | Proliférante |

---

## ⚙️ Installation

> ✅ **Déjà installé sur cette machine** dans un environnement isolé (venv) :
> `D:\aptos-retinopathy\.venv` — PyTorch **2.11.0+cu128**, GPU RTX A2000 actif.
> Le projet est sur **D:** car le disque C: était plein.

Pour rejouer l'installation depuis zéro :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128   # cu128 = CUDA 12.8
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print('CUDA :', torch.cuda.is_available())"
```

> ⚠️ Disque système petit ? Redirige le temp de pip avant d'installer :
> `$env:TMP="D:\tmp"; $env:TEMP="D:\tmp"; $env:TMPDIR="D:\tmp"` + option `--no-cache-dir`.

---

## 📥 Données (APTOS 2019, gratuit via Kaggle)

1. Crée un compte sur [kaggle.com](https://www.kaggle.com)
2. **Account → Settings → API → Create New Token** → télécharge `kaggle.json`
3. Place-le dans `C:\Users\<toi>\.kaggle\kaggle.json`
4. Accepte les règles de la compétition :
   [aptos2019-blindness-detection/rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules)
5. Lance le téléchargement :
   ```powershell
   .\run.ps1 src.download_data
   ```

> Tu obtiens `data/train.csv` + `data/train_images/` (~3662 images, ~10 Go).

---

## 🚀 Utilisation

Le script **`run.ps1`** règle automatiquement le venv + les variables temp, puis
lance le module demandé. (Sans lui : `.\.venv\Scripts\python.exe -m src.xxx`.)

```powershell
# 1. (optionnel) Tester le prétraitement sur une image
.\run.ps1 src.preprocess data/train_images/000c1434d8d7.png

# 2. Entraîner le modèle
.\run.ps1 src.train --epochs 15 --model resnet34

# 3. Évaluer sur le jeu de test
.\run.ps1 src.evaluate

# 4. Visualiser avec Grad-CAM
.\run.ps1 src.gradcam --image data/train_images/000c1434d8d7.png
```

Les résultats (modèle, matrice de confusion, Grad-CAM) sont dans `outputs/`.

---

## 📊 Résultats

Modèle **ResNet34** (transfer learning), 15 époques sur le jeu APTOS 2019 :

| Métrique | Score |
|----------|-------|
| Accuracy | **0.814** |
| Quadratic Weighted Kappa | **0.889** |

<p align="center">
  <img src="assets/confusion_matrix.png" width="560" alt="Matrice de confusion">
  <br>
  <em>Matrice de confusion sur le jeu de validation. Les confusions restent
  concentrées entre stades voisins (1↔2, 2↔3), ce qui explique le kappa élevé.</em>
</p>

---

## 📁 Structure

```
aptos-retinopathy/
├── README.md
├── requirements.txt
├── data/                 # données APTOS (non versionnées)
├── outputs/              # modèles + figures générés
├── assets/               # figures du README
└── src/
    ├── config.py         # chemins, hyperparamètres, GPU
    ├── download_data.py  # téléchargement Kaggle
    ├── preprocess.py     # prétraitement OpenCV
    ├── dataset.py        # Dataset PyTorch + splits
    ├── model.py          # construction du CNN (transfer learning)
    ├── train.py          # boucle d'entraînement
    ├── evaluate.py       # métriques + matrice de confusion
    └── gradcam.py        # interprétabilité
```

---

