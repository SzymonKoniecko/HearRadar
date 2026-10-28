"""Scala audio z radaru z wideo z NVIDIA bez przekodowania obrazu.

Wyrównanie: w obu nagraniach szukany jest sygnał synchronizacji, czyli pół sekundy tonu 1 kHz
z `radar --sygnal`. Przesunięcie to różnica czasów tego tonu. Wideo jest kopiowane (`-c copy`),
oryginalne AAC z NVIDIA też idzie bez zmian, a WAV radaru trafia jako PCM 16-bit (bez kompresji).

Przykłady:
  python tools/scal_nagrania.py --wideo nagranie.mp4 --audio recordings/sesja/audio.wav
  python tools/scal_nagrania.py --wideo nagranie.mp4 --audio audio.wav --przesuniecie 1.7 --dry-run

Wymaga ffmpeg w PATH.
"""

import argparse
import array
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

TONE_HZ = 1000.0
BLOCK_S = 0.01
MIN_TONE_S = 0.3
PRESENCE_MIN = 0.25
ENERGY_MIN = 1e-3
SEARCH_S = 15.0


def read_mono(path: Path, seconds: float) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as w:
        channels, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        if width != 2:
            raise SystemExit(f"Błąd: {path} nie jest 16-bitowy")
        frames = min(w.getnframes(), int(seconds * rate))
        raw = w.readframes(frames)
    samples = array.array("h")
    samples.frombytes(raw)
    if channels == 2:
        mono = [(a + b) / 65536.0 for a, b in zip(samples[0::2], samples[1::2])]
    else:
        mono = [v / 32768.0 for v in samples[::channels]]
    return mono, rate


def tone_ratio(block: list[float], rate: int) -> float:
    """Ile energii bloku siedzi w 1 kHz (ok. 0,5 dla czystego tonu, blisko zera dla szumu)."""
    n = len(block)
    k = round(TONE_HZ * n / rate)
    coeff = 2.0 * math.cos(2.0 * math.pi * k / n)
    s1 = s2 = 0.0
    for x in block:
        s1, s2 = x + coeff * s1 - s2, s1
    power = (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n * n)
    mean_sq = sum(x * x for x in block) / n
    if mean_sq < ENERGY_MIN:
        return 0.0
    return power / mean_sq


def refine_onset(mono: list[float], start: int, block: int) -> int:
    """Dokładny początek: pierwsza próbka ponad 10% szczytu tuż przy znalezionym bloku."""
    peak = max((abs(x) for x in mono[start : start + 5 * block]), default=0.0) or 1.0
    threshold = 0.1 * peak
    for i in range(max(0, start - block), min(len(mono), start + 2 * block)):
        if abs(mono[i]) > threshold:
            return i
    return start


def find_tone_onset(path: Path) -> float:
    mono, rate = read_mono(path, SEARCH_S)
    block = max(1, round(BLOCK_S * rate))
    need = math.ceil(MIN_TONE_S / BLOCK_S)
    run = 0
    for b in range(len(mono) // block):
        ratio = tone_ratio(mono[b * block : (b + 1) * block], rate)
        run = run + 1 if ratio > PRESENCE_MIN else 0
        if run >= need:
            start = (b - need + 1) * block
            return refine_onset(mono, start, block) / rate
    raise SystemExit(f"Błąd: w {path} nie ma sygnału synchronizacji w pierwszych {SEARCH_S:.0f} s")


def extract_nvidia_audio(video: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )


def build_command(video: Path, audio: Path, out: Path, offset: float, overwrite: bool) -> list[str]:
    # Dodatni offset: WAV radaru zaczyna się później w czasie wideo, więc go opóźniamy.
    # Ujemny: część początku WAV radaru jest przed wideo, więc ją przycinamy.
    if offset >= 0:
        audio_input = ["-itsoffset", f"{offset:.6f}", "-i", str(audio)]
    else:
        audio_input = ["-ss", f"{-offset:.6f}", "-i", str(audio)]
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        *audio_input,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a:0",
        "pcm_s16le",
        "-c:a:1",
        "copy",
        "-metadata:s:a:0",
        "title=Radar (PCM 16-bit)",
        "-metadata:s:a:1",
        "title=NVIDIA (oryginał)",
        str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scala WAV radaru z wideo z NVIDIA bez utraty jakości.")
    parser.add_argument("--wideo", required=True, type=Path, help="plik MP4 z NVIDIA")
    parser.add_argument("--audio", required=True, type=Path, help="audio.wav z radaru")
    parser.add_argument("--wyjscie", type=Path, help="plik MKV (domyślnie obok wideo)")
    parser.add_argument(
        "--przesuniecie",
        type=float,
        help="ręcznie: moment początku WAV radaru w czasie wideo (zamiast szukania tonu)",
    )
    parser.add_argument("--dry-run", action="store_true", help="tylko wypisz polecenie ffmpeg")
    parser.add_argument("--nadpisz", action="store_true", help="nadpisz istniejący plik wyjściowy")
    args = parser.parse_args()

    for path in (args.wideo, args.audio):
        if not path.is_file():
            print(f"Błąd: brak pliku {path}", file=sys.stderr)
            return 2
    out = args.wyjscie or args.wideo.with_name(f"{args.wideo.stem}_zlaczone.mkv")
    if out.exists() and not args.nadpisz and not args.dry_run:
        print(f"Błąd: {out} już istnieje (dodaj --nadpisz)", file=sys.stderr)
        return 2

    try:
        if args.przesuniecie is None:
            with tempfile.TemporaryDirectory() as tmp:
                nvidia_wav = Path(tmp) / "nvidia.wav"
                extract_nvidia_audio(args.wideo, nvidia_wav)
                t_nvidia = find_tone_onset(nvidia_wav)
            t_radar = find_tone_onset(args.audio)
            offset = t_nvidia - t_radar
            print(f"Ton w NVIDIA: {t_nvidia:.3f} s, ton w radarze: {t_radar:.3f} s")
            print(f"Przesunięcie WAV radaru w czasie wideo: {offset:+.3f} s")
        else:
            offset = args.przesuniecie
            print(f"Przesunięcie ręczne: {offset:+.3f} s")

        command = build_command(args.wideo, args.audio, out, offset, args.nadpisz)
        if args.dry_run:
            print(" ".join(f'"{part}"' if " " in part else part for part in command))
            return 0
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as err:
        print(f"Błąd ffmpeg (kod {err.returncode})", file=sys.stderr)
        return 2
    print(f"Zapisano {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
