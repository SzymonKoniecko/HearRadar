Wyciągnij audio z MKV z OBS do folderu sesji. Używamy 16-bit, bo tak czyta je detektor. Oryginalny MKV zostaje nietknięty:
> mkdir recordings\sesja_obs
> ffmpeg -i "nagranie_obs.mkv" -map 0:a:0 -ac 2 -ar 48000 -c:a pcm_s16le recordings\sesja_obs\audio.wav
Wykryj zdarzenia. Polecenie uruchamiasz z katalogu src\analyzer\HearAnalytics-ML:
> cd src\analyzer\HearAnalytics-ML
>python -m hearanalytics ..\..\..\recordings\sesja_obs

Powstają events.csv i katalog clips\.

Przejrzyj klipy. Odtwarza każde zdarzenie i zapisuje Twoje klasy i kierunki do events.csv:
>python -m hearanalytics review ..\..\..\recordings\sesja_obs

Opcja --only-unlabeled pokaże tylko zdarzenia bez etykiety.

Porównaj z radarem. Z katalogu repo:
>radar --classify-session recordings\sesja_obs\audio.wav

Kolumna klasa_auto w events.csv pokazuje klasę detektora, a Twoja klasa jest w klasa.