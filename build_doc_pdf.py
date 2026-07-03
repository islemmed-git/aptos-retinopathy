# -*- coding: utf-8 -*-
"""
Genere un PDF pedagogique complet expliquant le projet APTOS :
maladie, stades, modele, train/val/test, metriques, acronymes.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, HRFlowable)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Guide_APTOS_Retinopathie.pdf"

# ----- Couleurs de la charte -----
NAVY = colors.HexColor("#1b3a5b")
BLUE = colors.HexColor("#2c6fb5")
LIGHT = colors.HexColor("#eaf1f8")
ACCENT = colors.HexColor("#c0392b")
GREY = colors.HexColor("#555555")

styles = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("title", parent=styles["Title"], fontSize=26,
                            textColor=NAVY, leading=30, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=13,
                               textColor=GREY, alignment=TA_CENTER, leading=18),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16,
                         textColor=colors.white, backColor=NAVY, leading=22,
                         spaceBefore=16, spaceAfter=10, leftIndent=6,
                         borderPadding=(6, 6, 6, 6)),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                         textColor=BLUE, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5,
                           leading=15, alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", parent=styles["Normal"], fontSize=10.5,
                             leading=15, leftIndent=14, bulletIndent=4,
                             spaceAfter=3),
    "note": ParagraphStyle("note", parent=styles["Normal"], fontSize=10,
                           leading=14, textColor=NAVY, backColor=LIGHT,
                           borderPadding=(8, 8, 8, 8), spaceBefore=6,
                           spaceAfter=8),
    "small": ParagraphStyle("small", parent=styles["Normal"], fontSize=9,
                            leading=12, textColor=GREY, alignment=TA_CENTER),
    "code": ParagraphStyle("code", parent=styles["Normal"], fontName="Courier",
                           fontSize=9.5, leading=13, textColor=colors.black,
                           backColor=colors.HexColor("#f4f4f4"),
                           borderPadding=(6, 6, 6, 6), spaceAfter=8),
}

story = []


def P(txt, st="body"):
    story.append(Paragraph(txt, S[st]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))


def space(h=8):
    story.append(Spacer(1, h))


def h1(txt):
    story.append(Paragraph(txt, S["h1"]))


def h2(txt):
    story.append(Paragraph(txt, S["h2"]))


def note(txt):
    story.append(Paragraph(txt, S["note"]))


def code(txt):
    story.append(Paragraph(txt.replace(" ", "&nbsp;"), S["code"]))


def table(data, widths, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), BLUE),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(ts))
    story.append(t)
    space(8)


# Cellules de tableau en Paragraph (pour le retour a la ligne)
def cell(txt, bold=False):
    st = ParagraphStyle("c", parent=S["body"], fontSize=9, leading=12,
                        alignment=0, spaceAfter=0)
    if bold:
        st.fontName = "Helvetica-Bold"
    return Paragraph(txt, st)


# ===========================================================================
# PAGE DE TITRE
# ===========================================================================
space(60)
P("Detection de la retinopathie diabetique", "title")
P("par Deep Learning", "title")
space(10)
story.append(HRFlowable(width="60%", thickness=2, color=BLUE,
                        spaceBefore=4, spaceAfter=14, hAlign="CENTER"))
P("Guide pedagogique complet : la maladie, les stades cliniques,<br/>"
  "le fonctionnement du modele, l'entrainement et les metriques", "subtitle")
space(30)
P("Projet APTOS 2019 &nbsp;|&nbsp; ResNet34 &nbsp;|&nbsp; PyTorch", "small")
P("Document genere automatiquement", "small")
story.append(PageBreak())

# ===========================================================================
# SOMMAIRE
# ===========================================================================
h1("Sommaire")
somm = [
    "1. Vue d'ensemble du projet",
    "2. La maladie : la retinopathie diabetique",
    "3. Les 5 stades et leurs criteres cliniques",
    "4. Le pipeline du projet (vue globale)",
    "5. Le pretraitement des images (OpenCV)",
    "6. Le modele : Deep Learning et CNN",
    "7. ResNet34 et le transfer learning",
    "8. Train / Validation / Test",
    "9. Comment le modele apprend",
    "10. Les metriques d'evaluation",
    "11. Les resultats obtenus",
    "12. Grad-CAM : l'explicabilite",
    "13. Glossaire complet des acronymes",
    "14. Les commandes du projet",
]
for s in somm:
    P(s, "bullet")
story.append(PageBreak())

# ===========================================================================
# 1. VUE D'ENSEMBLE
# ===========================================================================
h1("1. Vue d'ensemble du projet")
P("Ce projet construit un systeme d'<b>intelligence artificielle</b> capable "
  "de regarder une photographie du fond de l'oeil (une <b>retinographie</b>) "
  "et de dire automatiquement a quel <b>stade de gravite</b> se trouve la "
  "retinopathie diabetique, sur une echelle de 0 (sain) a 4 (le plus grave).")
P("Le systeme apprend tout seul a partir de <b>2930 images</b> deja "
  "diagnostiquees par des medecins (le dataset public APTOS 2019). Une fois "
  "entraine, il peut diagnostiquer une nouvelle image en une fraction de "
  "seconde.")
note("<b>Idee cle :</b> on ne programme pas de regles a la main "
     "(\"si tache rouge alors...\"). Le modele <b>decouvre lui-meme</b> les "
     "signes de la maladie en observant des milliers d'exemples. C'est le "
     "principe du <b>deep learning</b>.")

# ===========================================================================
# 2. LA MALADIE
# ===========================================================================
h1("2. La maladie : la retinopathie diabetique")
P("La <b>retinopathie diabetique</b> est une complication du diabete qui "
  "abime les petits vaisseaux sanguins de la <b>retine</b> (le fond de "
  "l'oeil). C'est l'une des premieres causes de cecite chez l'adulte. Plus "
  "elle est detectee tot, mieux elle se soigne, d'ou l'interet d'un depistage "
  "automatique.")
h2("Les lesions que l'on cherche sur l'image")
table([
    [cell("Lesion", True), cell("Description", True)],
    [cell("Micro-anevrismes"), cell("Minuscules dilatations des capillaires, premiers signes (petits points rouges).")],
    [cell("Hemorragies"), cell("Fuites de sang dans la retine (taches rouges plus grandes).")],
    [cell("Exsudats"), cell("Depots de graisses/proteines (taches jaunes brillantes).")],
    [cell("AMIR / IRMA"), cell("Anomalies microvasculaires intra-retiniennes (vaisseaux anormaux).")],
    [cell("Neovascularisation"), cell("Croissance de nouveaux vaisseaux anormaux et fragiles (stade le plus grave).")],
], [4.2 * cm, 11 * cm])

# ===========================================================================
# 3. LES 5 STADES
# ===========================================================================
h1("3. Les 5 stades et leurs criteres cliniques")
P("La classification suit l'<b>echelle internationale de severite</b> de la "
  "retinopathie diabetique. Les stades 0 a 3 sont dits <b>non proliferants</b> "
  "(NPDR), le stade 4 est <b>proliferant</b> (PDR).")
table([
    [cell("Stade", True), cell("Nom", True), cell("Critere clinique", True)],
    [cell("0", True), cell("Pas de retinopathie"),
     cell("Aucune anomalie visible. Retine saine.")],
    [cell("1", True), cell("Legere (Mild NPDR)"),
     cell("<b>Micro-anevrismes uniquement.</b> Tout premier stade.")],
    [cell("2", True), cell("Moderee (Moderate NPDR)"),
     cell("Plus que de simples micro-anevrismes (hemorragies, exsudats...) mais sans atteindre le stade severe.")],
    [cell("3", True), cell("Severe (Severe NPDR)"),
     cell("<b>Regle 4-2-1</b> : hemorragies importantes dans les 4 quadrants, OU dilatation veineuse dans 2 quadrants, OU AMIR marquees dans 1 quadrant. Pas encore de neovaisseaux.")],
    [cell("4", True), cell("Proliferante (PDR)"),
     cell("<b>Neovascularisation</b> (nouveaux vaisseaux anormaux) et/ou hemorragie du vitre. Stade le plus grave, risque de cecite.")],
], [1.3 * cm, 3.6 * cm, 10.3 * cm])
note("<b>A retenir :</b> les stades sont <b>ordonnes</b> (0 &lt; 1 &lt; 2 &lt; 3 &lt; 4) "
     "et les frontieres entre stades voisins sont floues, meme pour un "
     "ophtalmologue. C'est pour cela qu'on utilise une metrique speciale, le "
     "<b>kappa quadratique</b> (chapitre 10), qui pardonne les erreurs entre "
     "stades proches.")

story.append(PageBreak())

# ===========================================================================
# 4. PIPELINE
# ===========================================================================
h1("4. Le pipeline du projet (vue globale)")
P("Voici les grandes etapes, de l'image brute au diagnostic :")
code("Image retine<br/>"
     "&nbsp;&nbsp;|<br/>"
     "&nbsp;&nbsp;v&nbsp;&nbsp; 1. PRETRAITEMENT (OpenCV : crop + CLAHE + resize)<br/>"
     "&nbsp;&nbsp;|<br/>"
     "&nbsp;&nbsp;v&nbsp;&nbsp; 2. MODELE CNN (ResNet34, transfer learning)<br/>"
     "&nbsp;&nbsp;|<br/>"
     "&nbsp;&nbsp;v&nbsp;&nbsp; 3. SORTIE : 5 probabilites (un score par stade)<br/>"
     "&nbsp;&nbsp;|<br/>"
     "&nbsp;&nbsp;v&nbsp;&nbsp; 4. DIAGNOSTIC : le stade le plus probable")
P("Les etapes 1 a 4 servent a <b>predire</b>. Pour <b>apprendre</b>, on ajoute "
  "une boucle de correction (chapitre 9).")

# ===========================================================================
# 5. PRETRAITEMENT
# ===========================================================================
h1("5. Le pretraitement des images (OpenCV)")
P("Avant d'entrer dans le modele, chaque image est nettoyee et standardisee "
  "avec la bibliotheque <b>OpenCV</b>. Quatre operations :")
table([
    [cell("Etape", True), cell("Role", True)],
    [cell("Crop circulaire"), cell("Enleve les bandes noires autour du disque retinien, recadre sur l'oeil.")],
    [cell("CLAHE"), cell("Rehaussement de contraste local : fait ressortir les vaisseaux et lesions sans cramer l'image.")],
    [cell("Redimensionnement"), cell("Toutes les images ramenees a 224x224 pixels (taille attendue par ResNet).")],
    [cell("Normalisation"), cell("Les couleurs sont recalibrees avec les statistiques d'ImageNet (moyenne/ecart-type), car le modele pre-entraine les attend.")],
], [4.2 * cm, 11 * cm])
note("<b>Pourquoi standardiser ?</b> Les photos viennent d'appareils et de "
     "conditions d'eclairage differents. En les uniformisant, on aide le "
     "modele a se concentrer sur la <b>maladie</b> plutot que sur les "
     "differences de materiel.")

# ===========================================================================
# 6. DEEP LEARNING
# ===========================================================================
h1("6. Le modele : Deep Learning et CNN")
h2("Intelligence artificielle, machine learning, deep learning")
P("Ce sont des poupees russes :")
bullets([
    "<b>IA</b> : le grand domaine (faire faire a une machine des taches \"intelligentes\").",
    "<b>Machine Learning (ML)</b> : la machine apprend a partir de donnees au lieu d'etre programmee.",
    "<b>Deep Learning (DL)</b> : du ML avec des reseaux de neurones <b>profonds</b> (beaucoup de couches).",
])
table([
    [cell("", True), cell("Machine Learning classique", True), cell("Deep Learning", True)],
    [cell("Donnees", True), cell("Tableaux de chiffres"), cell("Images, sons, textes")],
    [cell("Qui choisit les variables ?", True), cell("L'humain, a la main"), cell("Le reseau, tout seul")],
    [cell("Sur une image", True), cell("Mauvais"), cell("Excellent")],
], [3.8 * cm, 5.6 * cm, 5.6 * cm])
h2("Le CNN et la convolution")
P("Pour les images, on utilise un <b>CNN</b> (Convolutional Neural Network = "
  "reseau de neurones convolutif). Son operation cle est la "
  "<b>convolution</b> : de petits <b>filtres</b> glissent sur l'image pour "
  "detecter des motifs. En empilant les couches, le reseau detecte des motifs "
  "de plus en plus complexes :")
code("Couche 1&nbsp;&nbsp; -> bords, coins, couleurs<br/>"
     "Couche 2&nbsp;&nbsp; -> textures, courbes<br/>"
     "Couche 3&nbsp;&nbsp; -> vaisseaux, taches<br/>"
     "...<br/>"
     "Couche finale -> lesions, diagnostic (stade 0 a 4)")

# ===========================================================================
# 7. RESNET34
# ===========================================================================
h1("7. ResNet34 et le transfer learning")
P("<b>ResNet34</b> est l'architecture de CNN utilisee ici. \"34\" = le nombre "
  "de couches. \"ResNet\" = <i>Residual Network</i> (reseau residuel).")
h2("Les connexions residuelles (skip connections)")
P("L'innovation de ResNet : des <b>raccourcis</b> qui sautent par-dessus des "
  "couches. Au lieu d'apprendre toute la transformation, chaque bloc apprend "
  "juste la <b>correction a ajouter</b>. Resultat : on peut construire des "
  "reseaux tres profonds sans qu'ils \"se cassent\" a l'entrainement.")
h2("Le transfer learning")
P("On ne part pas de zero : on prend un ResNet34 <b>deja entraine</b> sur "
  "ImageNet (1,2 million d'images variees). Il sait deja reconnaitre formes et "
  "textures ; on le <b>specialise</b> ensuite sur les retines. C'est pour cela "
  "qu'on a obtenu un excellent score des la 1ere epoque.")
note("<b>Analogie :</b> plutot que d'apprendre a un enfant a voir depuis zero, "
     "on prend un adulte qui sait deja observer le monde, et on lui apprend "
     "seulement une nouvelle specialite (lire des retines).")

story.append(PageBreak())

# ===========================================================================
# 8. TRAIN / VAL / TEST
# ===========================================================================
h1("8. Train / Validation / Test")
P("Les 2930 images sont decoupees en <b>trois ensembles separes</b>. C'est "
  "fondamental pour mesurer honnetement la performance.")
table([
    [cell("Ensemble", True), cell("Part", True), cell("Nb images", True), cell("Role", True)],
    [cell("Train (entrainement)", True), cell("70%"), cell("2051"),
     cell("Le modele apprend dessus : il voit ces images et corrige ses erreurs.")],
    [cell("Validation", True), cell("15%"), cell("439"),
     cell("Controle pendant l'entrainement : verifie qu'il generalise et choisit le meilleur modele.")],
    [cell("Test", True), cell("15%"), cell("440"),
     cell("Examen final, sur des images JAMAIS vues. Mesure honnete de la vraie performance.")],
], [3.6 * cm, 1.5 * cm, 2.0 * cm, 8.3 * cm])
h2("Pourquoi separer ?")
P("Si on testait le modele sur les images d'entrainement, il pourrait avoir "
  "simplement <b>memorise</b> les reponses (comme un eleve qui apprend le "
  "corrige par coeur). Le jeu de <b>test</b> verifie qu'il a vraiment compris, "
  "sur de l'inedit.")
h2("Le surapprentissage (overfitting)")
P("Quand un modele devient excellent sur le train mais stagne (ou empire) sur "
  "la validation, il <b>surapprend</b> : il memorise au lieu de generaliser. "
  "On le repere quand l'erreur de validation remonte alors que celle "
  "d'entrainement continue de baisser. C'est pour cela qu'on sauvegarde le "
  "<b>meilleur modele selon la validation</b>, pas le dernier.")

# ===========================================================================
# 9. COMMENT LE MODELE APPREND
# ===========================================================================
h1("9. Comment le modele apprend")
P("L'apprentissage est une boucle repetee des milliers de fois. Quelques "
  "definitions d'abord :")
bullets([
    "<b>Epoque (epoch)</b> : un passage complet sur toutes les images d'entrainement. Ici : 15 epoques.",
    "<b>Batch</b> : un petit paquet d'images traitees ensemble. Ici : 16 images par batch (2051 / 16 = 129 batches par epoque).",
])
P("Pour chaque batch, 5 etapes :")
table([
    [cell("Etape", True), cell("Nom", True), cell("Ce qui se passe", True)],
    [cell("1"), cell("Forward (propagation avant)"), cell("Les images traversent le reseau, qui predit un stade.")],
    [cell("2"), cell("Loss (perte)"), cell("On mesure l'erreur : ecart entre prediction et vraie reponse.")],
    [cell("3"), cell("Backward (retropropagation)"), cell("On calcule comment chaque parametre a contribue a l'erreur.")],
    [cell("4"), cell("Step (optimiseur)"), cell("On ajuste legerement chaque parametre pour reduire l'erreur.")],
    [cell("5"), cell("Repeter"), cell("On passe au batch suivant, puis a l'epoque suivante.")],
], [1.1 * cm, 4.4 * cm, 9.9 * cm])
h2("Les ingredients techniques")
table([
    [cell("Element", True), cell("Choix dans le projet", True), cell("Role", True)],
    [cell("Fonction de perte"), cell("CrossEntropy ponderee"), cell("Mesure l'erreur ; ponderee pour gerer le desequilibre des stades.")],
    [cell("Optimiseur"), cell("AdamW"), cell("Decide comment ajuster les parametres (descente de gradient amelioree).")],
    [cell("Learning rate"), cell("0.0001"), cell("Taille des pas de correction.")],
    [cell("AMP"), cell("Mixed precision"), cell("Calculs en 16 bits : economise la memoire GPU et accelere.")],
], [3.6 * cm, 4.4 * cm, 7.4 * cm])
note("<b>Analogie de la descente :</b> l'erreur est une montagne dans le "
     "brouillard. A chaque pas, on sent la pente (le <b>gradient</b>) et on "
     "avance d'un petit pas (le <b>learning rate</b>) vers le bas, jusqu'a "
     "atteindre la vallee (erreur minimale).")

story.append(PageBreak())

# ===========================================================================
# 10. METRIQUES
# ===========================================================================
h1("10. Les metriques d'evaluation")
P("Comment juger si le modele est bon ? Plusieurs mesures complementaires :")
table([
    [cell("Metrique", True), cell("Definition", True)],
    [cell("Accuracy (exactitude)"), cell("Pourcentage de predictions exactes. Simple, mais trompeuse si les classes sont desequilibrees.")],
    [cell("Precision"), cell("Parmi les cas predits \"stade X\", combien le sont vraiment. (Eviter les fausses alertes.)")],
    [cell("Recall (rappel)"), cell("Parmi les vrais \"stade X\", combien sont retrouves. (Eviter les oublis.)")],
    [cell("F1-score"), cell("Moyenne equilibree de la precision et du rappel.")],
    [cell("Support"), cell("Nombre d'images reelles de chaque stade dans le test.")],
    [cell("Kappa quadratique"), cell("Metrique OFFICIELLE d'APTOS. Mesure l'accord en penalisant fortement les grosses erreurs et peu les erreurs entre stades voisins.")],
], [4.0 * cm, 11.2 * cm])
h2("Pourquoi le kappa quadratique ?")
P("Comme les stades sont ordonnes, confondre le stade 2 et le 3 (voisins) est "
  "peu grave, alors que confondre 0 et 4 serait dramatique. Le <b>quadratic "
  "weighted kappa</b> tient compte de cette <b>distance</b> entre stades. "
  "Echelle : 0 = hasard, 1 = parfait ; 0,8 a 0,9 = tres bon.")
h2("La matrice de confusion")
P("Elle croise la <b>verite</b> (lignes) et la <b>prediction</b> (colonnes). "
  "La diagonale = les bonnes reponses. Les cases hors diagonale = les "
  "confusions. Voici celle obtenue sur le jeu de test :")
cm_img = ROOT / "outputs" / "confusion_matrix.png"
if cm_img.exists():
    story.append(Image(str(cm_img), width=12 * cm, height=9 * cm))
    P("Matrice de confusion sur les 440 images de test.", "small")

story.append(PageBreak())

# ===========================================================================
# 11. RESULTATS
# ===========================================================================
h1("11. Les resultats obtenus")
P("Apres 15 epoques d'entrainement (ResNet34) :")
table([
    [cell("Mesure", True), cell("Valeur", True)],
    [cell("Accuracy (test)"), cell("0,814")],
    [cell("Kappa quadratique (test)"), cell("0,889  (tres bon)")],
    [cell("Kappa (validation)"), cell("0,893")],
], [7.0 * cm, 5.0 * cm])
P("Le kappa de test (0,889) est quasi identique a celui de validation "
  "(0,893) : le modele <b>generalise bien</b>, sans tricherie cachee.")
h2("Performance detaillee par stade")
table([
    [cell("Stade", True), cell("Precision", True), cell("Recall", True), cell("F1", True), cell("Support", True)],
    [cell("0 - Pas de retinopathie"), cell("0,986"), cell("0,972"), cell("0,979"), cell("215")],
    [cell("1 - Legere"), cell("0,528"), cell("0,622"), cell("0,571"), cell("45")],
    [cell("2 - Moderee"), cell("0,757"), cell("0,713"), cell("0,734"), cell("122")],
    [cell("3 - Severe"), cell("0,500"), cell("0,478"), cell("0,489"), cell("23")],
    [cell("4 - Proliferante"), cell("0,605"), cell("0,657"), cell("0,630"), cell("35")],
], [5.4 * cm, 2.5 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm])
note("<b>Lecture :</b> le stade 0 (sain) est quasi parfait. Les stades 1 et 3 "
     "sont les plus durs car ils ont peu d'exemples ET ce sont des stades "
     "intermediaires aux frontieres floues. Surtout, le modele ne fait "
     "<b>jamais</b> d'erreur grave (confondre sain et tres grave) : ses "
     "erreurs restent sur des stades voisins.")

# ===========================================================================
# 12. GRAD-CAM
# ===========================================================================
h1("12. Grad-CAM : l'explicabilite")
P("Un reseau de neurones est souvent une <b>boite noire</b>. Le <b>Grad-CAM</b> "
  "(Gradient-weighted Class Activation Mapping) l'ouvre : il produit une "
  "<b>carte de chaleur</b> montrant les zones de l'image qui ont le plus "
  "influence la decision. Rouge = zone regardee, bleu = zone ignoree.")
_gc = sorted((ROOT / "outputs").glob("gradcam*.png"))
gc_img = _gc[0] if _gc else (ROOT / "outputs" / "gradcam.png")
if gc_img.exists():
    story.append(Image(str(gc_img), width=14 * cm, height=14 * cm * 0.42))
    P("Grad-CAM sur un cas avance (predit stade 3 - severe) : le modele se "
      "concentre sur la zone centrale de la retine, la ou se trouvent les "
      "lesions.", "small")
note("<b>Pourquoi c'est important :</b> un medecin peut <b>verifier</b> que le "
     "modele regarde des zones medicalement pertinentes, et non un artefact. "
     "C'est essentiel pour la confiance en IA medicale.")

story.append(PageBreak())

# ===========================================================================
# 13. GLOSSAIRE
# ===========================================================================
h1("13. Glossaire complet des acronymes")
gloss = [
    ("IA", "Intelligence Artificielle."),
    ("ML", "Machine Learning : apprentissage automatique a partir de donnees."),
    ("DL", "Deep Learning : ML avec reseaux de neurones profonds."),
    ("CNN", "Convolutional Neural Network : reseau de neurones convolutif, specialise pour les images."),
    ("ResNet", "Residual Network : architecture CNN avec connexions residuelles (raccourcis)."),
    ("ReLU", "Rectified Linear Unit : fonction d'activation courante (garde les valeurs positives)."),
    ("APTOS", "Asia Pacific Tele-Ophthalmology Society : organisateur du dataset/competition."),
    ("NPDR", "Non-Proliferative Diabetic Retinopathy : stades 0 a 3."),
    ("PDR", "Proliferative Diabetic Retinopathy : stade 4 (avec neovaisseaux)."),
    ("DMLA", "Degenerescence Maculaire Liee a l'Age (autre maladie de la retine)."),
    ("AMIR / IRMA", "Anomalies Microvasculaires Intra-Retiniennes (Intraretinal Microvascular Abnormalities)."),
    ("CLAHE", "Contrast Limited Adaptive Histogram Equalization : rehaussement de contraste local."),
    ("RGB", "Red-Green-Blue : codage couleur d'une image (3 canaux)."),
    ("LAB", "Espace colorimetrique Luminance + 2 axes couleur (utilise par CLAHE)."),
    ("Kappa", "Quadratic Weighted Kappa : metrique d'accord penalisant selon la distance entre classes."),
    ("GPU", "Graphics Processing Unit : carte graphique, ideale pour le calcul parallele du DL."),
    ("CUDA", "Plateforme NVIDIA permettant d'utiliser le GPU pour le calcul."),
    ("VRAM", "Video RAM : memoire du GPU (ici 4 Go)."),
    ("AMP", "Automatic Mixed Precision : calculs en 16 bits pour economiser memoire et temps."),
    ("AdamW", "Optimiseur (variante d'Adam avec weight decay) : ajuste les parametres."),
    ("LR", "Learning Rate : taille des pas de correction pendant l'apprentissage."),
    ("Epoch", "Epoque : un passage complet sur toutes les donnees d'entrainement."),
    ("Batch", "Lot d'images traitees ensemble a chaque etape."),
    ("Loss", "Fonction de perte : mesure chiffree de l'erreur du modele."),
    ("Grad-CAM", "Gradient-weighted Class Activation Mapping : carte de chaleur d'explicabilite."),
    ("CSV", "Comma-Separated Values : fichier tableur texte (ici les labels)."),
    ("PNG", "Format d'image sans perte (les retinographies)."),
    ("venv", "Virtual environment : environnement Python isole pour le projet."),
    ("API", "Application Programming Interface : interface pour appeler un service par programme."),
]
rows = [[cell("Sigle", True), cell("Signification", True)]]
for a, d in gloss:
    rows.append([cell(a, True), cell(d)])
table(rows, [3.0 * cm, 12.2 * cm])

story.append(PageBreak())

# ===========================================================================
# 14. COMMANDES
# ===========================================================================
h1("14. Les commandes du projet")
P("Toutes les commandes se lancent depuis le dossier du projet, via le "
  "lanceur <font name='Courier'>run.ps1</font> qui regle l'environnement "
  "automatiquement.")
table([
    [cell("Commande", True), cell("Role", True)],
    [cell("run.ps1 src.download_data"), cell("Telecharge les donnees APTOS.")],
    [cell("run.ps1 src.preprocess &lt;image&gt;"), cell("Montre le pretraitement avant/apres.")],
    [cell("run.ps1 src.train"), cell("Entraine le modele.")],
    [cell("run.ps1 src.evaluate"), cell("Evalue sur le test (kappa + matrice).")],
    [cell("run.ps1 src.predict --image &lt;img&gt;"), cell("Predit le stade d'une image.")],
    [cell("run.ps1 src.gradcam --image &lt;img&gt;"), cell("Carte de chaleur Grad-CAM.")],
], [7.6 * cm, 7.6 * cm])
h2("Options de l'entrainement")
code("run.ps1 src.train --epochs 15 --model resnet34 --batch-size 16 --lr 0.0001")
space(20)
story.append(HRFlowable(width="100%", thickness=1, color=BLUE))
P("Fin du document. Projet APTOS - detection de la retinopathie diabetique "
  "par deep learning.", "small")


# ===========================================================================
# Pied de page (numero) + construction
# ===========================================================================
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"- {doc.page} -")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        title="Guide APTOS - Retinopathie diabetique")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF genere :", OUT)
