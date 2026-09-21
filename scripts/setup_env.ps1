# Rebuilds the environment from scratch. Safe to re-run.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { py -3.10 -m venv .venv }
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install -r requirements.txt
& $Py -m pip install -e vendor/monotonic_align
if (-not (Test-Path "StyleTTS2")) { git clone https://github.com/yl4579/StyleTTS2.git }
git -C StyleTTS2 checkout -- .
git -C StyleTTS2 checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529
git -C StyleTTS2 apply (Join-Path $Root "patches\styletts2.patch")
& $Py scripts\download_models.py
& $Py scripts\check_env.py
