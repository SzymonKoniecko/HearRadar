#!/usr/bin/env bash
# Cold start dla klasyfikacji dźwięku ML: jedna komenda, która przygotowuje
# wszystko potrzebne, żeby włączyć eksperymentalną klasyfikację w HearRadar.
#
#   ./ml/bootstrap.sh
#
# Wymaga Pythona 3.9+ z dostępem do internetu (pobiera YAMNet z TF-Hub oraz
# zależności z PyPI). Zajmuje kilka minut przy pierwszym uruchomieniu.
# Wynik: assets/ml/yamnet.onnx + assets/ml/category_mapping.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== HearRadar: cold start klasyfikacji ML =="

if [ ! -d .venv ]; then
    echo "-> tworzę wirtualne środowisko (.venv)"
    python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "-> instaluję zależności (ml/requirements.txt)"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

mkdir -p out

echo "-> eksportuję YAMNet do ONNX"
python export_yamnet_onnx.py

echo "-> buduję mapowanie 521 klas AudioSet -> kategorie"
python export_mapping.py

mkdir -p ../assets/ml
cp out/yamnet.onnx ../assets/ml/yamnet.onnx
cp out/category_mapping.json ../assets/ml/category_mapping.json

echo ""
echo "Gotowe. Pliki modelu:"
ls -lh ../assets/ml/

echo ""
echo "Włącz 'Klasyfikację dźwięku (ML, eksperymentalne)' w ustawieniach radaru."
echo "(Kwantyzacja INT8 jest opcjonalna — patrz ml/quantize.py, jeśli inferencja jest za wolna.)"
