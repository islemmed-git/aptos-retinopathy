@echo off
title API Retinopathie Diabetique
cd /d D:\aptos-retinopathy

rem -- dossiers temporaires sur D: --
set TMP=D:\tmp
set TEMP=D:\tmp
set TMPDIR=D:\tmp

echo ============================================================
echo   API Retinopathie Diabetique
echo.
echo   Demarrage du serveur (chargement du modele + GPU)...
echo   Patiente ~30 secondes : la page web s'ouvrira toute seule.
echo   (laisse cette fenetre ouverte ; ferme-la pour arreter l'API)
echo ============================================================

rem -- ouvre le navigateur sur la doc apres 30 secondes (le temps que le modele charge) --
start "" cmd /c "timeout /t 30 >nul & start http://127.0.0.1:8000/docs"

rem -- lance le serveur --
".venv\Scripts\python.exe" -m uvicorn api:app --port 8000

echo.
echo Le serveur s'est arrete.
pause
