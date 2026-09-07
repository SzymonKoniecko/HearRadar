#!/usr/bin/env python3
"""Buduje `category_mapping.json`: dla każdej z 521 klas AudioSet (w
kolejności indeksów zwracanych przez YAMNet) zapisuje jej kategorię wg
`class_mapping.categorize`. Rust czyta ten plik i NIE musi znać ani
parsować nazw klas AudioSet — dostaje gotową tablicę indeks -> kategoria.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tensorflow_hub as hub

from class_mapping import all_categories, categorize

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
OUT_PATH = Path(__file__).parent / "out" / "category_mapping.json"


def main() -> None:
    print(f"Ładowanie mapy klas YAMNet z {YAMNET_HANDLE}...")
    yamnet_model = hub.load(YAMNET_HANDLE)
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")

    categories: list[str] = []
    display_names: list[str] = []
    with open(class_map_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_names.append(row["display_name"])
            categories.append(categorize(row["display_name"]))

    if len(categories) != 521:
        print(f"UWAGA: oczekiwano 521 klas AudioSet, znaleziono {len(categories)} — sprawdź wersję modelu.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Indeks w tej liście == indeks klasy w wyjściu modelu ONNX (scores[i]).
        "categories": categories,
        # Pełna lista możliwych kategorii, w stałej kolejności — Rust używa
        # tego do zbudowania domyślnej palety kolorów (RadarConfig.category_colors).
        "unique_categories": all_categories(),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    counts: dict[str, int] = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1
    print(f"Zapisano {OUT_PATH} — {len(categories)} klas AudioSet -> {len(counts)} kategorii:")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
