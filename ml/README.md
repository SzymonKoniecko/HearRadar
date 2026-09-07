# Klasyfikacja dźwięku (ML) — narzędzia offline

Ten katalog **nie jest częścią aplikacji Rust** — to jednorazowe (albo
rzadko powtarzane) narzędzia Pythona, które przygotowują plik modelu
(`.onnx`) używany potem przez `src/sound/ml_classifier.rs` w czasie
działania aplikacji. Aplikacja sama w sobie nie wymaga Pythona ani
TensorFlow — tylko gotowego pliku `.onnx`.

## Szybki start

```bash
./ml/bootstrap.sh
```

To jedna komenda: tworzy wirtualne środowisko, instaluje zależności,
eksportuje [YAMNet](https://tfhub.dev/google/yamnet/1) do ONNX, buduje
mapowanie 521 klas AudioSet na kategorie i kopiuje gotowe pliki do
`assets/ml/` (który jest w `.gitignore` — to duże pliki binarne, nie kod).

Po zakończeniu włącz w aplikacji: **Ustawienia radaru → 🧪 Klasyfikacja
dźwięku (ML, eksperymentalne)**.

## Co robią poszczególne skrypty

| Plik | Co robi |
|---|---|
| `export_yamnet_onnx.py` | Ładuje YAMNet z TF-Hub, eksportuje do ONNX, **weryfikuje eksport** porównując top-5 klas między TF a ONNX na tym samym sygnale testowym (przerywa z błędem, jeśli się nie zgadza). |
| `class_mapping.py` | Statyczna tabela słów kluczowych: nazwa klasy AudioSet → jedna z ~8 kategorii ("Kroki", "Strzały/wybuchy", "Mowa", "Pojazdy"...). |
| `export_mapping.py` | Stosuje `class_mapping.py` do wszystkich 521 klas YAMNet i zapisuje `category_mapping.json` (tablica indeks → kategoria). |
| `quantize.py` | **Opcjonalny.** Kwantyzuje model FP32 → INT8, żeby przyspieszyć inferencję na CPU. Uruchom dopiero, jeśli profilowanie w aplikacji pokaże, że jest to potrzebne — `bootstrap.sh` tego nie robi automatycznie. |

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
`./ml/bootstrap.sh` ponownie.
