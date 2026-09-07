#!/usr/bin/env python3
"""Eksportuje YAMNet (TF-Hub) do ONNX i WERYFIKUJE eksport, porównując top-5
wykrytych klas między oryginalnym modelem TensorFlow a wyeksportowanym
grafem ONNX na tym samym syntetycznym sygnale. Bez tej weryfikacji łatwo
przeoczyć błąd eksportu (np. złą oś czasu wejścia) i dowiedzieć się o nim
dopiero jako "dziwne wyniki klasyfikacji" w aplikacji.

Uruchamiane przez `ml/bootstrap.sh` — nie uruchamiaj bezpośrednio, chyba że
wiesz co robisz (potrzebuje aktywnego venv z ml/requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
OUT_DIR = Path(__file__).parent / "out"
OUT_PATH = OUT_DIR / "yamnet.onnx"
OPSET = 13


def build_scores_fn():
    """Ładuje YAMNet i zwraca tf.function zwracającą TYLKO `scores`
    (521 klas per klatka ~0.96s) — embeddingi i spektrogram nie są nam
    potrzebne w aplikacji, więc pomijamy je już na etapie eksportu."""
    yamnet_model = hub.load(YAMNET_HANDLE)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
    def yamnet_scores(waveform):
        scores, _embeddings, _spectrogram = yamnet_model(waveform)
        return scores

    return yamnet_scores


def freeze(yamnet_scores):
    """Zamraża WSZYSTKIE zmienne/zasoby przechwycone z modelu TF-Hub w stałe
    wpisane wprost w graf. Bez tego `tf2onnx` zostawia dziesiątki dodatkowych,
    wymaganych wejść (odpowiadających wagom modelu skopiowanym z SavedModel
    TF-Hub) zamiast jednego wejścia `waveform` — bo `hub.load()` zwraca
    obiekt, którego zmienne nie są automatycznie zamrażane przy zwykłym
    `tf2onnx.convert.from_function`."""
    concrete_func = yamnet_scores.get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    return frozen_func


def verify(frozen_func, onnx_path: Path) -> None:
    """Porównuje top-5 klas (uśrednionych po klatkach czasowych) między
    zamrożonym grafem TF a ONNX na tym samym losowym sygnale — musi się
    zgadzać, inaczej eksport jest zepsuty."""
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    waveform = rng.uniform(-1.0, 1.0, size=16_000).astype(np.float32)  # 1 s @ 16 kHz

    tf_scores = frozen_func(tf.constant(waveform))[0].numpy()
    tf_top5 = set(np.argsort(tf_scores.mean(axis=0))[-5:].tolist())

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_scores = session.run(None, {input_name: waveform})[0]
    onnx_top5 = set(np.argsort(onnx_scores.mean(axis=0))[-5:].tolist())

    if tf_top5 != onnx_top5:
        print(
            f"BŁĄD: eksport ONNX niezgodny z TF.\n  TF top-5 indeksów:   {sorted(tf_top5)}\n  ONNX top-5 indeksów: {sorted(onnx_top5)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"OK: TF i ONNX zgadzają się co do top-5 klas: {sorted(tf_top5)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ładowanie YAMNet z {YAMNET_HANDLE} (pierwsze uruchomienie pobiera model, może chwilę potrwać)...")
    yamnet_scores = build_scores_fn()

    print("Zamrażanie grafu (wagi modelu -> stałe)...")
    frozen_func = freeze(yamnet_scores)
    input_name = frozen_func.inputs[0].name
    output_name = frozen_func.outputs[0].name

    print(f"Eksport do ONNX (opset {OPSET})...")
    tf2onnx.convert.from_graph_def(
        frozen_func.graph.as_graph_def(),
        input_names=[input_name],
        output_names=[output_name],
        opset=OPSET,
        output_path=str(OUT_PATH),
    )
    print(f"Zapisano {OUT_PATH}")

    print("Weryfikacja eksportu (TF vs ONNX)...")
    verify(frozen_func, OUT_PATH)


if __name__ == "__main__":
    main()
