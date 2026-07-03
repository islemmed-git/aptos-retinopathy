# Lanceur pratique : regle l'environnement (venv + temp sur D:) puis execute une commande du projet.
#
# Exemples :
#   .\run.ps1 src.download_data
#   .\run.ps1 src.train --epochs 15 --model resnet34
#   .\run.ps1 src.evaluate
#   .\run.ps1 src.gradcam --image data/train_images/000c1434d8d7.png

$env:TMP = "D:\tmp"
$env:TEMP = "D:\tmp"
$env:TMPDIR = "D:\tmp"

$python = "D:\aptos-retinopathy\.venv\Scripts\python.exe"
Set-Location "D:\aptos-retinopathy"

& $python -m $args
