# Klasyfikacja dźwięku (ML) — narzędzia offline

Ten katalog **nie jest częścią aplikacji Rust** — to jednorazowe (albo
rzadko powtarzane) narzędzia Pythona, które przygotowują plik modelu
(`.onnx`) używany potem przez `src/sound/ml_classifier.rs` w czasie
działania aplikacji. Aplikacja sama w sobie nie wymaga Pythona ani
TensorFlow — tylko gotowego pliku `.onnx`.

## Szybki start

macOS/Linux:

```bash
./ml/bootstrap.sh
```

Windows (PowerShell):

```powershell
.\ml\bootstrap.ps1
```

Jeśli PowerShell odmówi uruchomienia skryptu (domyślna polityka wykonywania
blokuje nieskompilowane skrypty), uruchom raz:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

...albo jednorazowo, bez trwałej zmiany polityki:

```powershell
powershell -ExecutionPolicy Bypass -File ml\bootstrap.ps1
```

Oba skrypty robią to samo: tworzą wirtualne środowisko, instalują zależności,
eksportują [YAMNet](https://tfhub.dev/google/yamnet/1) do ONNX, budują
mapowanie 521 klas AudioSet na kategorie i kopiują gotowe pliki do
`assets/ml/` (który jest w `.gitignore` — to duże pliki binarne, nie kod).

Po zakończeniu włącz w aplikacji: **Ustawienia radaru → 🧪 Klasyfikacja
dźwięku (ML, eksperymentalne)**.

### Windows: błąd `ensurepip`/`venv` przy tworzeniu `.venv`

Jeśli `bootstrap.ps1` wywali się na etapie tworzenia `.venv` z długim
tracebackiem kończącym się gdzieś w `ensurepip`/`subprocess.communicate` —
to znany problem `python -m venv` na Windows, nie błąd w tym repo. `venv`
wewnętrznie odpala nowy proces Pythona, żeby zainstalować pip, i ten
podproces czasem pada w trakcie czytania jego wyjścia — najczęściej przez
konsolę/locale inne niż UTF-8, rzadziej przez antywirusa blokującego świeżo
odpalony `python.exe`.

`bootstrap.ps1` **próbuje to obejść automatycznie**: ustawia
`PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8` (usuwa przyczynę w większości
przypadków) i — jeśli mimo to `venv` się wywali — tworzy środowisko bez pip
(`--without-pip`) i doinstalowuje pip osobno (`ensurepip`, a jak i to
zawiedzie, `get-pip.py` pobrane bezpośrednio z pypa.io).

Jeśli to WCIĄŻ nie pomoże, zrób to ręcznie:

```powershell
cd ml
Remove-Item -Recurse -Force .venv  # jeśli częściowo się utworzył
python -m venv .venv --without-pip
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe export_yamnet_onnx.py
.\.venv\Scripts\python.exe export_mapping.py
```

...i skopiuj `out\yamnet.onnx` oraz `out\category_mapping.json` do
`..\assets\ml\`. Jeśli nawet `ensurepip --upgrade` się wywala, sprawdź, czy
antywirus/Windows Defender nie blokuje uruchamiania `python.exe` z katalogu
`.venv` (dodaj wyjątek albo tymczasowo wyłącz ochronę w czasie rzeczywistym
na czas tego jednego polecenia).

## Co robią poszczególne skrypty

| Plik | Co robi |
|---|---|
| `bootstrap.sh` / `bootstrap.ps1` | Cold start — venv, zależności, eksport modelu i mapowania, kopiowanie do `assets/ml/`. macOS/Linux vs Windows, robią dokładnie to samo. |
| `export_yamnet_onnx.py` | Ładuje YAMNet z TF-Hub, eksportuje do ONNX, **weryfikuje eksport** porównując top-5 klas między TF a ONNX na tym samym sygnale testowym (przerywa z błędem, jeśli się nie zgadza). |
| `class_mapping.py` | Statyczna tabela słów kluczowych: nazwa klasy AudioSet → jedna z ~8 kategorii ("Kroki", "Strzały/wybuchy", "Mowa", "Pojazdy"...). |
| `export_mapping.py` | Stosuje `class_mapping.py` do wszystkich 521 klas YAMNet i zapisuje `category_mapping.json` (tablica indeks → kategoria). |
| `info.py` | Diagnostyka: sprawdza instalację TF/onnxruntime i pliki modelu, a jeśli są dostępne — faktycznie klasyfikuje testowy ton i zwraca wynik. Wywoływane przez przycisk "Sprawdź środowisko Pythona" w UI aplikacji, nie ręcznie. |
| `quantize.py` | **Opcjonalny.** Kwantyzuje model FP32 → INT8, żeby przyspieszyć inferencję na CPU. Uruchom dopiero, jeśli profilowanie w aplikacji pokaże, że jest to potrzebne — `bootstrap.sh`/`bootstrap.ps1` tego nie robią automatycznie. |

## Dlaczego YAMNet, nie PANNs CNN14

YAMNet jest lżejszy, szybszy na CPU i zaprojektowany pod strumieniowanie
(natywne okno ~0.96 s). Co ważniejsze dla integracji: konwersja
waveform → mel-spektrogram jest częścią jego grafu obliczeniowego, więc po
eksporcie do ONNX aplikacja Rust karmi model surowym dźwiękiem (16 kHz mono)
i nie musi mieć własnej implementacji mel-spektrogramu.

## Dlaczego mapowanie po słowach kluczowych, nie ręczna lista 521 klas

521 klas AudioSet to dużo do ręcznego wypisania i łatwo o pomyłkę. Zamiast
tego `class_mapping.py` dopasowuje nazwy klas po słowach kluczowych — jeśli
chcesz poprawić kategoryzację konkretnej klasy, dopisz/przenieś słowo
kluczowe w `CATEGORY_KEYWORDS`, uruchom `python export_mapping.py` ponownie
(bez potrzeby re-eksportu samego modelu) i skopiuj nowy
`out/category_mapping.json` do `assets/ml/`.

## Ograniczenia — przeczytaj przed pokazaniem tego jako "gotowej" funkcji

- Nie ma w tym repo żadnego zbioru nagrań z gier z etykietami, więc **nie da
  się automatycznie zmierzyć trafności klasyfikacji**. `export_yamnet_onnx.py`
  weryfikuje tylko, że eksport ONNX jest wierny oryginalnemu modelowi TF —
  nie że model dobrze rozpoznaje dźwięki z konkretnej gry.
- Przed pokazaniem funkcji jako działającej: nagraj kilka krótkich klipów
  (kroki, strzał, cisza, muzyka) z docelowej gry, uruchom aplikację z
  włączoną klasyfikacją i ręcznie sprawdź, czy wyniki są sensowne. Zapisz
  wynik takiego sprawdzenia — to jedyna dostępna dziś forma "testu" tej
  funkcji.
- Klasy AudioSet są ogólne (np. "Explosion" nie odróżnia wybuchu granatu od
  fajerwerków) — to narzędzie orientacyjne, nie pewne źródło informacji.

## Regenerowanie modelu

Pliki w `assets/ml/` są w `.gitignore` — nie trafiają do repo. Jeśli
klonujesz repo od nowa (albo zmieniasz mapowanie kategorii), uruchom
`./ml/bootstrap.sh` (macOS/Linux) albo `.\ml\bootstrap.ps1` (Windows) ponownie.
