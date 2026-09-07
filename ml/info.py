#!/usr/bin/env python3
"""Szybka diagnostyka środowiska ML — wypisuje JSON na stdout z informacją,
czy Python/TensorFlow/onnxruntime są zainstalowane, czy model już został
wyeksportowany, ORAZ — jeśli onnxruntime i model są dostępne — faktycznie
uruchamia klasyfikację testowego sygnału i zwraca jej wynik. To nie jest
tylko "czy pliki istnieją": realnie sprawdza, że środowisko Pythona jest w
stanie coś sklasyfikować, tym samym silnikiem (onnxruntime) i tym samym
plikiem `.onnx`, którego używa potem aplikacja Rust — więc wynik stąd
powinien zgadzać się (co do kategorii) z tym, co pokaże Rust dla tego
samego sygnału (patrz `ui_python_info_test` w `src/ui/app.rs`).

Używane przez UI aplikacji Rust (przycisk "Sprawdź środowisko Pythona") jako
jedyny punkt, w którym appka w ogóle odpala proces Pythona w runtime — poza
tym Python jest używany wyłącznie offline (patrz `ml/README.md`).

Celowo NIE importuje TensorFlow na sztywno (import trwa kilka sekund i
mógłby zawiesić UI) — sprawdza dostępność przez `importlib.util` i importuje
tylko, żeby odczytać `__version__`. Sam test klasyfikacji używa
onnxruntime, nie TensorFlow — szybszy start, i to dokładnie ten sam silnik
co w aplikacji Rust.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

ML_DIR = Path(__file__).parent
MODEL_PATH = ML_DIR.parent / "assets" / "ml" / "yamnet.onnx"
MAPPING_PATH = ML_DIR.parent / "assets" / "ml" / "category_mapping.json"

# Ten sam sygnał testowy (częstotliwość, czas trwania, częstotliwość
# próbkowania) generuje po swojej stronie Rust (`ui_python_info_test`) —
# dzięki temu dwa niezależne wyniki (Python/onnxruntime i Rust/ort) da się
# bezpośrednio porównać w UI.
TEST_FREQ_HZ = 1000.0
TEST_SAMPLE_RATE = 16_000
TEST_AMPLITUDE = 0.5


def package_version(name: str) -> tuple[bool, str | None]:
    """Zwraca (czy zainstalowany, wersja albo None)."""
    if importlib.util.find_spec(name) is None:
        return False, None
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "?")
    except Exception:
        # Pakiet jest "widoczny" (find_spec go znalazł), ale import się nie
        # udał (np. niekompletna instalacja) — traktujemy jak brak.
        return False, None


def generate_test_waveform() -> list[float]:
    """Ton sinusoidalny 1 kHz, 1 s @ 16 kHz — bez numpy, żeby ten skrypt nie
    zależał od niczego poza biblioteką standardową poza samym onnxruntime."""
    return [
        TEST_AMPLITUDE * math.sin(2.0 * math.pi * TEST_FREQ_HZ * (i / TEST_SAMPLE_RATE))
        for i in range(TEST_SAMPLE_RATE)
    ]


def run_self_test(categories: list[str]) -> dict:
    """Faktycznie klasyfikuje testowy sygnał przez onnxruntime na
    wyeksportowanym modelu — nie tylko sprawdza, że pliki istnieją, tylko że
    cały łańcuch (onnxruntime + model + mapowanie) daje sensowny wynik.
    Uśrednia wyjście po klatkach czasowych i bierze najsilniejszą kategorię —
    dokładnie tak samo jak `SoundClassifier::classify` w Rust
    (`src/sound/ml_classifier.rs`), żeby wyniki dało się porównać 1:1."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    waveform = np.array(generate_test_waveform(), dtype=np.float32)

    scores = session.run(None, {input_name: waveform})[0]
    num_classes = len(categories)
    if num_classes == 0 or len(scores[0]) != num_classes:
        raise ValueError(
            f"liczba klas w wyjściu modelu ({len(scores[0]) if len(scores) else 0}) "
            f"nie zgadza się z category_mapping.json ({num_classes})"
        )

    # Uśrednij po klatkach czasowych, potem znajdź najsilniejszą kategorię
    # (max spośród klas AudioSet należących do niej) — patrz classify() w Rust.
    avg_per_class = [sum(frame[i] for frame in scores) / len(scores) for i in range(num_classes)]
    best_per_category: dict[str, float] = {}
    for class_idx, score in enumerate(avg_per_class):
        category = categories[class_idx]
        if score > best_per_category.get(category, float("-inf")):
            best_per_category[category] = score

    top_category, top_confidence = max(best_per_category.items(), key=lambda kv: kv[1])
    return {"category": top_category, "confidence": top_confidence}


def main() -> None:
    tf_installed, tf_version = package_version("tensorflow")
    ort_installed, ort_version = package_version("onnxruntime")

    model_exists = MODEL_PATH.exists()
    model_size_mb = round(MODEL_PATH.stat().st_size / 1_000_000, 1) if model_exists else None

    mapping_exists = MAPPING_PATH.exists()
    category_count = None
    categories: list[str] = []
    if mapping_exists:
        try:
            mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
            categories = mapping.get("categories", [])
            category_count = len(mapping.get("unique_categories", []))
        except (json.JSONDecodeError, OSError):
            mapping_exists = False

    test_category = None
    test_confidence = None
    test_error = None
    if ort_installed and model_exists and mapping_exists:
        try:
            result = run_self_test(categories)
            test_category = result["category"]
            test_confidence = result["confidence"]
        except Exception as e:  # noqa: BLE001 — to tylko diagnostyka, nie ma się wysypać
            test_error = str(e)

    info = {
        "python_version": platform.python_version(),
        "tensorflow_installed": tf_installed,
        "tensorflow_version": tf_version,
        "onnxruntime_installed": ort_installed,
        "onnxruntime_version": ort_version,
        "test_category": test_category,
        "test_confidence": test_confidence,
        "test_error": test_error,
        "model_exists": model_exists,
        "model_size_mb": model_size_mb,
        "mapping_exists": mapping_exists,
        "category_count": category_count,
    }

    json.dump(info, sys.stdout)


if __name__ == "__main__":
    main()
