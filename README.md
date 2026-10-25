# HearRadar

Radar dźwięku dla gry: na tarczy pokazuje, skąd nadchodzi dźwięk i jak jest głośny.
Dźwięk przechwytywany jest z wirtualnego kabla (VB-Audio Virtual Cable), więc aplikacja
nie czyta pamięci gry.

Prace nad rozpoznawaniem klas dźwięku (ML) i azymutem z różnicy czasu między uszami (ITD)
są w toku. Szczegóły: `src/analyzer/HearAnalytics-ML/` (`ML_RADAR_ANALYSIS.md`,
`PLAN_DZIALAN.md`).

---

## Wymagania

- Windows (przechwytywanie przez WASAPI).
- VB-Audio Virtual Cable.
- Ustawienie urządzenia `CABLE Input`: Ustawienia dźwięku → Urządzenia odtwarzania →
  CABLE Input → Właściwości → Zaawansowane → format **8 kanałów, 48 000 Hz**.
- Gra wysyła dźwięk na urządzenie `CABLE Input` (jako domyślne wyjście).
- Rust, edycja 2024.

---

## Uruchamianie

```bash
cargo run                      # okno radaru (GUI)
cargo run -- <komenda> <opcje> # tryb bez GUI
```

Albo zbudowaną binarkę: `target\debug\radar.exe <komenda> <opcje>`.

Ustawienia okna zapisują się w `config/radar_config.json`.

> **Uwaga:** nieznane argumenty są ignorowane. Literówka w nazwie komendy (np. `--recrod`)
> uruchomi więc okno GUI zamiast komendy.

---

## Komendy

| Komenda | Do czego | Domyślne |
|---|---|---|
| `radar` (bez argumentów) | okno radaru | — |
| `--listen-cable [--seconds N]` | RMS kanałów urządzenia „Cable In 16” | 10 s |
| `--channel-test <urządzenie>` | sprawdza, który indeks kanału gra | — |
| `--monitor <urządzenie> [--seconds N]` | podgląd na żywo: RMS kanałów i aktywne indeksy | 60 s |
| `--recordSample <nazwa sesji> [--seconds N] [opcje]` | nagrywa dźwięk (kanały 0 i 1) do WAV z metadanymi i znacznikami, do etykietowania; z `--seconds` kończy się sam | `CABLE Input`, tryb `stereo`, bez limitu czasu |
| `--recordVideo <nazwa sesji> [opcje]` | to samo co `--recordSample` plus obraz do `video.mp4` (przez ffmpeg) | jak wyżej, `ffmpeg` z PATH, cały pulpit, 30 fps |
| `--classify-clips <folder> [--onset S]` | przypisuje każdemu klipowi klasę A–D i warstwę (strzały, kroki, reszta) tą samą drogą co radar; do porównania z kolumną `klasa_auto` z Pythona | `--onset 0.2` |

---

### `--listen-cable [--seconds N]`

Nasłuchuje urządzenia, którego nazwa zawiera „cable in 16”, i co 0,5 s wypisuje RMS
wszystkich kanałów.

| Parametr | Opis | Domyślnie |
|---|---|---|
| `--seconds N` | czas nasłuchu w sekundach | 10 |

Ograniczenie: nazwa urządzenia jest na stałe w kodzie i nie da się jej zmienić. Do
VB-Cable ustawionego na 8 kanałów użyj `--monitor`.

```bash
radar --listen-cable --seconds 30
```

Wyjście: `t=1.5s ch=8 rms=[0.0000, 0.2120, ...]`.

---

### `--channel-test <urządzenie>`

Sprawdza, który indeks kanału urządzenia faktycznie dostaje sygnał. Dla każdego indeksu
gra przez 1 s sinus 1 kHz (amplituda 0,3), potem 0,4 s ciszy. Jednocześnie nasłuchuje
pętli zwrotnej tego samego urządzenia.

| Parametr | Opis |
|---|---|
| `<urządzenie>` | fragment nazwy urządzenia wyjściowego, bez rozróżniania wielkości liter. Bierze pierwsze pasujące. |

Liczba kanałów to ustawienie urządzenia w Windows, nie opcja programu.

Wyjście, po jednej linii na kanał, i podsumowanie:

```
[0]  FL  → pętla zwrotna: kanał 0 ✓
...
[7]  SR  → pętla zwrotna: kanał 7 ✓

Zgodnych z pętlą zwrotną: 8/8
```

Nazwy przy kanałach (FL, FR, FC, LFE, BL, BR, SL, SR) to oczekiwana kolejność wg maski
WASAPI. Test nie sprawdza jej słuchem, więc trzeba to potwierdzić na prawdziwych
głośnikach albo w grze.

```bash
radar --channel-test "CABLE Input"
```

---

### `--monitor <urządzenie> [--seconds N]`

Podgląd na żywo: co 250 ms wypisuje średni RMS każdego kanału pętli zwrotnej i listę
aktywnych kanałów (RMS ≥ 0,002). Służy do sprawdzenia, które indeksy niesie gra.

| Parametr | Opis | Domyślnie |
|---|---|---|
| `<urządzenie>` | fragment nazwy urządzenia wyjściowego | — |
| `--seconds N` | czas podglądu w sekundach | 60 |

```bash
radar --monitor "CABLE Input" --seconds 120
```

Wyjście: `t= 2.4s  [0.054 0.052 0.000 0.000 0.000 0.000 0.000 0.000]  aktywne: [0, 1]`.

---

### `--recordSample <nazwa sesji> [opcje]`

Nagrywa dwa pierwsze kanały urządzenia (indeks 0 i 1) do pliku WAV. Obok zapisuje metadane
i znaczniki czasu.

| Parametr | Opis | Domyślnie |
|---|---|---|
| `<nazwa sesji>` | tylko litery ASCII, cyfry, `-` i `_`. Istniejąca nazwa jest odrzucana. | — |
| `--seconds N` | czas nagrywania w sekundach; po upływie sesja kończy się sama (300 = 5 minut). Bez opcji nagrywa do `q`. | brak limitu |
| `--device <urządzenie>` | urządzenie, z którego nagrywamy | `CABLE Input` |
| `--game <tekst>` | nazwa gry (metadane) | brak |
| `--scene <tekst>` | mapa lub scena (metadane) | brak |
| `--mode <tekst>` | tryb dźwięku w grze (metadane) | `stereo` |
| `--output <tekst>` | wyjście, np. `słuchawki`, `głośniki` (metadane) | brak |
| `--spatial <tekst>` | efekt przestrzenny, np. `none`, `windows`, `game` (metadane) | brak |
| `--game-volume <tekst>` | głośność w grze, np. `70%` (metadane) | brak |
| `--windows-volume <tekst>` | głośność w Windows (metadane) | brak |

Podczas nagrywania:

- wpisz tekst i naciśnij Enter, żeby dodać znacznik (zapisywany z czasem i numerem klatki),
- wpisz `q` i naciśnij Enter, żeby zakończyć.

Z `--seconds N` nagrywanie kończy się samo po N sekundach. Znaczniki i `q` działają też wtedy,
więc można zakończyć wcześniej. Etykiety można też dopisać po nagraniu, bo plik `markers.csv`
to zwykły CSV.

```bash
radar --recordSample sesja_5min --seconds 300 --game "Gra X"
radar --recordSample s1_zachod --game "Gra X" --scene "Mapa 1" --output słuchawki --spatial none --game-volume 70%
```

Wyjście: katalog `recordings/<nazwa sesji>/` z plikami:

| Plik | Zawartość |
|---|---|
| `audio.wav` | kanały 0 (L) i 1 (P), 48 kHz, 16 bitów, PCM |
| `session.json` | metadane, status (`recording` / `done`), liczba klatek, czas trwania, opis kanałów |
| `markers.csv` | `t_s,frame,label`: czas, numer klatki i etykieta każdego znacznika |

Przerwanie w trakcie (Ctrl+C albo zabicie procesu) zostawia `audio.wav` odtwarzalny do
ostatniej sekundy. Status w `session.json` zostaje wtedy `recording`, a liczba klatek w nim
to 0. Prawdziwe dane są w nagłówku WAV.

---

### `--recordVideo <nazwa sesji> [opcje]`

Robi to samo co `--recordSample`: dźwięk do `audio.wav`, metadane, znaczniki, `q` i `--seconds`.
Dodatkowo nagrywa obraz do `video.mp4`. Obraz przechwytuje zewnętrzny program **ffmpeg**, który
musi być zainstalowany i dostępny w PATH albo podany przez `--ffmpeg`.

| Parametr | Opis | Domyślnie |
|---|---|---|
| `--ffmpeg <ścieżka>` | program ffmpeg | `ffmpeg` z PATH |
| `--window <tytuł>` | nagrywa tylko okno o podanym tytule zamiast całego pulpitu | cały pulpit |
| `--fps N` | klatki na sekundę wideo | 30 |

Pozostałe opcje (`--device`, `--seconds`, `--game`, `--scene`, `--mode`, `--output`, `--spatial`,
`--game-volume`, `--windows-volume`) działają tak samo jak w `--recordSample`.

```bash
radar --recordVideo sesja_wideo --seconds 300 --game "Gra X" --window "Gra X"
```

Wyjście: pliki jak w `--recordSample` plus `video.mp4`. W `session.json` pojawiają się pola:

- `video_file`: nazwa pliku wideo,
- `video_offset_s`: o ile sekund wideo wystartowało po dźwięku (dodatnie = później).

Przesunięcie to czas od pierwszej ramki audio do pierwszej klatki wideo (raport `-progress`
ffmpeg). Dokładność to ok. 0,1 s. Nagrywanie idzie z `-tune zerolatency`, żeby pierwsza klatka
nie czekała na bufor kodera.

Ograniczenia:

- gdigrab przechwytuje obraz z pulpitu. Gra w pełnoekranowym trybie wyłącznym może dać czarny
  obraz. Używaj trybu okienkowego albo bez ramki.
- Kończ przez `q`, nie Ctrl+C. Przerwanie może uszkodzić plik MP4.
- Gdy ffmpeg nie działa, komenda kończy się zanim utworzy sesję.

## Odsłuch nagrania i słupki warstw

W oknie głównym, w panelu „Źródło audio”, jest lista nagrań z `recordings/` (`audio.wav` i klipy). „▶ Odtwórz” puszcza wybrany plik przez ten sam potok analizy co żywe audio, w czasie rzeczywistym, więc radar pokazuje to, co pokazałby podczas grania. Odtwarzanie wyłącza żywe przechwytywanie. „Przechwytuj” je przywraca.

Tryb „▮▮▮ Warstwy 3+3” (w ustawieniach wyświetlania) rysuje sześć słupków, po trzy na stronę. Od środka: strzały i wybuchy, kroki, reszta. Klasa A–D wynika z poziomu i jasności zdarzenia. Mapowanie klasa → warstwa jest tymczasowe i zapisane w `src/sound/layers.rs` (`layer_for`). Trafienie pojawia się ok. 0,35 s po zdarzeniu, bo tyle trwa okno szczytu.

## Pliki tworzone przez program

| Plik | Kiedy powstaje | Uwagi |
|---|---|---|
| `config/radar_config.json` | po zmianie ustawień w oknie | jeden plik, nadpisywany |
| `config/sample_log_NNNN.csv` | po włączeniu loggera próbek w oknie | numer rośnie od najwyższego istniejącego |
| `recordings/<sesja>/audio.wav` | `--recordSample`, `--recordVideo` | nazwa pliku stała, unikalność daje folder |
| `recordings/<sesja>/session.json` | `--recordSample`, `--recordVideo` (start i koniec) | nadpisywany w tej samej sesji |
| `recordings/<sesja>/markers.csv` | `--recordSample`, `--recordVideo` | dopisywany przy każdym znaczniku |
| `recordings/<sesja>/video.mp4` | `--recordVideo` | obraz z ffmpeg |

Katalog `recordings/` jest w `.gitignore`.

---

## Testy i budowanie

```bash
cargo build
cargo test
```

Testy jednostkowe nie zmieniają `config/radar_config.json`. Testy nagrywania piszą do
katalogu tymczasowego.

---

## Znane ograniczenia

- `--listen-cable` szuka tylko urządzenia „Cable In 16”, bez opcji wyboru.
- Nazwy zastrzeżone Windows (np. `CON`, `COM1`) nie są jeszcze odrzucane przez `--recordSample` ani `--recordVideo`.
- Moduł ITD (`src/sound/binaural.rs`) nie jest podłączony do radaru. Działa tylko w testach.
- W nagraniach zakładamy, że indeks 0 to lewy kanał. To jeszcze nie jest potwierdzone.
- Dane sesji (`recordings/`) mogą zawierać dźwięk z gry. Nie udostępniaj ich.
