"""
Configuration centrale du projet.
Tous les chemins, hyperparametres et reglages materiels sont ici.
"""
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# CHEMINS
# ---------------------------------------------------------------------------
# Racine du projet (le dossier aptos-retinopathy/)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"                       # donnees brutes APTOS
TRAIN_CSV = DATA_DIR / "train.csv"             # id_code, diagnosis (0-4)

# Le dossier des images peut etre imbrique selon la source :
#   - competition officielle  -> data/train_images/
#   - miroir mariaherrerot    -> data/train_images/train_images/
# On detecte automatiquement le bon chemin.
_nested = DATA_DIR / "train_images" / "train_images"
TRAIN_IMAGES = _nested if _nested.exists() else DATA_DIR / "train_images"

OUTPUT_DIR = ROOT / "outputs"                  # modeles + figures generes
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "best_model.pth"

# ---------------------------------------------------------------------------
# MATERIEL
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DONNEES
# ---------------------------------------------------------------------------
NUM_CLASSES = 5            # retinopathie diabetique : stades 0 a 4
IMG_SIZE = 224            # taille d'entree du CNN
CLASS_NAMES = [
    "0 - Pas de retinopathie",
    "1 - Legere",
    "2 - Moderee",
    "3 - Severe",
    "4 - Proliferante",
]

# ---------------------------------------------------------------------------
# ENTRAINEMENT
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 16            # adapte aux 4 Go de la RTX A2000
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
MODEL_NAME = "resnet34"   # "resnet18" | "resnet34" | "resnet50" | "efficientnet_b0"

# Repartition train / validation / test (en fraction)
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Normalisation ImageNet (les modeles pre-entraines l'attendent)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
