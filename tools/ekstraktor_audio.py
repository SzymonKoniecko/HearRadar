"""Wyciąga ścieżkę audio z nagrania NVIDIA (ShadowPlay / NVIDIA App) do pliku WAV.

Nagranie z NVIDIA często ma kilka ścieżek audio (np. dźwięk gry i mikrofon), więc najpierw
wypisz je opcją --lista, a potem wybierz numer opcją --sciezka. Numeruje się od 0.

Przykłady:
  python tools/ekstraktor_audio.py nagranie.mp4 --lista
  python tools/ekstraktor_audio.py nagranie.mp4
  python tools/ekstraktor_audio.py nagranie.mp4 --sciezka 1 --float --out gra.wav
  python tools/ekstraktor_audio.py nagranie.mp4 --kanaly 2 --czestotliwosc 48000

Wymaga ffmpeg i ffprobe w PATH.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def audio_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout).get("streams", [])


def describe(index: int, stream: dict) -> str:
    tags = stream.get("tags", {})
    title = tags.get("title") or tags.get("handler_name") or ""
    codec = stream.get("codec_name", "?")
    channels = stream.get("channels", "?")
    rate = stream.get("sample_rate", "?")
    return f"ścieżka {index}: {codec}, {channels} kan., {rate} Hz {title}".rstrip()


def extract(
    source: Path,
    out: Path,
    track: int,
    float_format: bool,
    rate: int | None,
    channels: int | None,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        f"0:a:{track}",
        "-vn",
    ]
    if rate:
        command += ["-ar", str(rate)]
    if channels:
        command += ["-ac", str(channels)]
    command += ["-c:a", "pcm_f32le" if float_format else "pcm_s16le", str(out)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wyciąga audio z nagrania NVIDIA do WAV.")
    parser.add_argument("nagranie", type=Path, help="plik MP4 z NVIDIA")
    parser.add_argument("--lista", action="store_true", help="tylko wypisz ścieżki audio")
    parser.add_argument("--sciezka", type=int, default=0, help="numer ścieżki audio (od 0)")
    parser.add_argument("--out", type=Path, help="plik WAV (domyślnie obok nagrania)")
    parser.add_argument("--float", action="store_true", help="zapisz jako 32-bit float zamiast 16-bit")
    parser.add_argument("--czestotliwosc", type=int, help="częstotliwość próbkowania, np. 48000")
    parser.add_argument("--kanaly", type=int, help="liczba kanałów wyjściowych, np. 2 (miks w dół)")
    parser.add_argument("--nadpisz", action="store_true", help="nadpisz istniejący plik WAV")
    args = parser.parse_args()

    source = args.nagranie
    if not source.is_file():
        print(f"Błąd: brak pliku {source}", file=sys.stderr)
        return 2

    streams = audio_streams(source)
    if not streams:
        print(f"Błąd: {source} nie ma ścieżki audio", file=sys.stderr)
        return 2

    if args.lista:
        for index, stream in enumerate(streams):
            print(describe(index, stream))
        return 0

    if not 0 <= args.sciezka < len(streams):
        print(f"Błąd: --sciezka {args.sciezka} poza zakresem 0..{len(streams) - 1}", file=sys.stderr)
        return 2

    out = args.out or source.with_suffix(".wav")
    if out.exists() and not args.nadpisz:
        print(f"Błąd: {out} już istnieje (dodaj --nadpisz)", file=sys.stderr)
        return 2

    print(f"Wybrano {describe(args.sciezka, streams[args.sciezka])}")
    if out.exists():
        out.unlink()
    extract(source, out, args.sciezka, args.float, args.czestotliwosc, args.kanaly)
    print(f"Zapisano {out} ({out.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
