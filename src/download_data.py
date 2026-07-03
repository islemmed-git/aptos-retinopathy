"""
Telechargement du dataset APTOS 2019.

On utilise un MIROIR public (dataset Kaggle classique) plutot que la
competition officielle : pas besoin d'accepter les regles ni de verifier
son telephone, seul le token API suffit.

PREREQUIS (une seule fois) :
  1. Compte Kaggle + token API.
     Nouvelle methode : enregistre ton access_token dans
        C:\\Users\\<toi>\\.kaggle\\access_token
     Ancienne methode : place kaggle.json dans  ~/.kaggle/kaggle.json
  2. Lance :  python -m src.download_data

Le dataset contient train.csv + train_images/train_images/*.png
(config.TRAIN_IMAGES detecte automatiquement le dossier imbrique).
"""
import subprocess
import sys

from . import config

# Miroir complet des donnees originales (8.6 Go)
DATASET = "mariaherrerot/aptos2019"


def main():
    config.DATA_DIR.mkdir(exist_ok=True)

    print(f"Telechargement de {DATASET} (~8.6 Go, peut prendre du temps)...")
    cmd = ["kaggle", "datasets", "download", "-d", DATASET,
           "-p", str(config.DATA_DIR), "--unzip"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nEchec du telechargement. Verifie :")
        print("  - token Kaggle bien place (~/.kaggle/access_token ou kaggle.json)")
        print("  - connexion internet")
        sys.exit(1)

    print(f"\nTermine. Donnees dans : {config.DATA_DIR}")
    print(f"Dossier images detecte : {config.TRAIN_IMAGES}")


if __name__ == "__main__":
    main()
