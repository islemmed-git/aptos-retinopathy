"""
Pretraitement des images de fond d'oeil avec OpenCV.

Les retinographies ont souvent :
  - des bordures noires inutiles autour du disque retinien
  - un eclairage/contraste variable d'une image a l'autre

On applique donc :
  1. un recadrage pour enlever le fond noir       (crop_from_gray)
  2. un recadrage circulaire centre sur la retine  (circle_crop)
  3. un rehaussement de contraste local (CLAHE)    (apply_clahe)
"""
import cv2
import numpy as np


def crop_from_gray(img, tol=7):
    """Enleve les bandes noires autour de la retine.

    On repere les pixels suffisamment clairs (> tol) et on rogne
    l'image au plus juste autour de cette zone.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:          # image entierement noire -> on ne touche pas
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords[0], coords[1], :] if img.ndim == 3 else img[coords]


def circle_crop(img):
    """Recadre l'image en cercle centre sur la retine (forme reelle de l'oeil)."""
    img = crop_from_gray(img)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(cx, cy)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return crop_from_gray(img)


def apply_clahe(img):
    """Contraste local adaptatif (CLAHE) sur le canal de luminance.

    Tres efficace en imagerie medicale : fait ressortir les details
    (vaisseaux, microanevrismes, drusens...) sans cramer l'image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_and_preprocess(path, size=224, use_clahe=True):
    """Charge une image disque -> RGB uint8 pretraitee, prete pour le CNN.

    Retourne un tableau numpy (size, size, 3) en uint8.
    """
    img = cv2.imread(str(path))                 # OpenCV lit en BGR
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB

    img = circle_crop(img)
    if use_clahe:
        img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    # Petit test visuel : python -m src.preprocess <chemin_image>
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage : python -m src.preprocess <chemin_vers_une_image>")
        sys.exit(1)

    raw = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    proc = load_and_preprocess(sys.argv[1])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(raw); ax[0].set_title("Originale"); ax[0].axis("off")
    ax[1].imshow(proc); ax[1].set_title("Pretraitee"); ax[1].axis("off")
    plt.tight_layout()
    plt.show()
