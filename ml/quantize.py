#!/usr/bin/env python3
"""Kwantyzacja FP32 -> INT8 (dynamiczna, wagi) dla `out/yamnet.onnx`,
zmniejszająca koszt inferencji na CPU. OPCJONALNE — uruchom to dopiero,
jeśli po zmierzeniu czasu inferencji w aplikacji (Rust, `ml_classifier.rs`)
FP32 okaże się za wolny na docelowym CPU. `ml/bootstrap.sh` NIE wywołuje
tego skryptu automatycznie.
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

OUT_DIR = Path(__file__).parent / "out"
INPUT_PATH = OUT_DIR / "yamnet.onnx"
OUTPUT_PATH = OUT_DIR / "yamnet.int8.onnx"


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Brak {INPUT_PATH} — najpierw uruchom export_yamnet_onnx.py (albo ./bootstrap.sh).")

    quantize_dynamic(
        model_input=str(INPUT_PATH),
        model_output=str(OUTPUT_PATH),
        weight_type=QuantType.QInt8,
    )

    before = INPUT_PATH.stat().st_size / 1_000_000
    after = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"Zapisano {OUTPUT_PATH} ({before:.1f} MB -> {after:.1f} MB)")
    print("Żeby aplikacja użyła tego wariantu, skopiuj go jako assets/ml/yamnet.onnx")
    print("albo ustaw ścieżkę modelu w kodzie (patrz src/sound/ml_classifier.rs).")


if __name__ == "__main__":
    main()
