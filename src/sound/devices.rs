use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sound::direction::{self, analyze_channels};
use crate::sound::sample_logger;
use crate::sound::spectrum::{self, ColorRule};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RadarConfig {
    /// Pozycja X środka radaru jako ułamek szerokości okna (0.0 = lewo, 1.0 = prawo).
    pub anchor_x: f32,
    /// Pozycja Y środka radaru jako ułamek wysokości okna (0.0 = góra, 1.0 = dół).
    pub anchor_y: f32,
    /// Promień radaru w pikselach (50 .. 300).
    pub radius: f32,
    /// Przezroczystość tła radaru (0.0 = pełna przezroczystość, 1.0 = pełne krycie).
    pub bg_alpha: f32,
    /// Ogólny mnożnik przezroczystości linii i kropek (0.0 .. 1.0).
    pub fg_alpha: f32,
    /// Tryb z etykietami (PRZÓD/TYŁ/LEWO/PRAWO + legenda) zamiast minimalnego.
    pub show_labels: bool,
    /// Tryb wysokiego kontrastu — paleta monochromatyczna zamiast kolorowej.
    pub high_contrast: bool,
    /// Czas zanikania pojedynczej kropki w sekundach (1.0 .. 3.0) — efekt
    /// "echa" ma być widoczny co najmniej 2-3 s, nie tylko przez ułamek sekundy.
    pub decay_secs: f32,
    /// Konfigurowalne reguły "zakres częstotliwości → kolor" — użytkownik
    /// może mieć ich dowolnie wiele (dodawać/usuwać/edytować w ustawieniach),
    /// zamiast sztywnych 3 pasm. Patrz [`crate::sound::spectrum::ColorRule`].
    pub color_rules: Vec<ColorRule>,
    /// Gdy `true`, overlay jest przezroczysty dla kliknięć/scrolla — myszka i
    /// klawiatura trafiają wprost do gry pod spodem. Wyłącz tymczasowo, żeby
    /// przeciągnąć radar w inne miejsce. `#[serde(default...)]` na `true`,
    /// żeby wczytanie starszego pliku konfiguracji (sprzed tego pola) nie
    /// włączyło niechcący trybu blokującego kliknięcia.
    #[serde(default = "default_overlay_click_through")]
    pub overlay_click_through: bool,
    /// Wzmocnienie sygnału mono wysyłanego do słuchawek przez `mono_router`,
    /// w decybelach (0.0 = bez zmian, ujemne = ciszej, dodatnie = głośniej).
    /// Odpowiednik `INTENSITY_SCALE` z `sound::direction` — tam sztywny
    /// mnożnik kompensujący z natury niskie RMS na potrzeby WYŚWIETLANIA na
    /// radarze, tu analogiczny, ale regulowany przez użytkownika mnożnik na
    /// samym WYJŚCIU dźwięku (patrz `mono_router::db_to_linear`).
    #[serde(default)]
    pub mono_boost_db: f32,
    /// Czułość radaru — mnożnik RMS każdego kanału przed przeliczeniem na
    /// odległość/rozmiar kropki od środka tarczy (patrz
    /// `radar_egui::RadarState::update`). Wyżej = ciche dźwięki wyglądają na
    /// "silniejsze"/dalsze, niżej = trzeba głośniej, żeby kropka dobiła do
    /// krawędzi. Domyślna wartość to dawna stała `direction::INTENSITY_SCALE`
    /// (patrz historia — dźwięki ledwo ruszały radarem, zanim ją
    /// wprowadzono); teraz user może ją dostroić suwakiem zamiast na sztywno.
    #[serde(default = "default_radar_sensitivity")]
    pub radar_sensitivity: f32,
    /// Drugi, opcjonalny tryb wizualizacji kierunku dźwięku, OBOK tarczy
    /// radaru (nie zamiast niej): poświata na krawędzi ekranu najbliższej
    /// kierunkowi każdego aktywnego wskaźnika (patrz
    /// `ui::radar_egui::draw_edge_glow_indicators`) — peryferyjna podpowiedź
    /// w stylu "screen edge indicator" ze strzelanek, żeby kierunek dźwięku
    /// był czytelny nawet kątem oka, bez patrzenia wprost na tarczę w rogu
    /// ekranu. Domyślnie wyłączone (`#[serde(default)]`), żeby nie zmieniać
    /// wyglądu dla obecnych użytkowników bez ich zgody — wymaga też, żeby
    /// overlay rozciągnął się na cały ekran zamiast małego kwadratu wokół
    /// tarczy (patrz `HearRadarApp::ui_overlay_contents`).
    #[serde(default)]
    pub edge_glow_enabled: bool,
    /// Mnożnik grubości WSZYSTKICH linii rysowanych na radarze (okręgi skali,
    /// krzyż kierunków, sektory 7.1, obrys kropek/wskaźników) — `1.0`
    /// odtwarza dawne, sztywne szerokości linii z `radar_egui::draw_radar`.
    /// `#[serde(default = ...)]` na `1.0` (nie na `0.0`, jak dałby zwykły
    /// `Default`), żeby wczytanie starszego pliku konfiguracji (sprzed tego
    /// pola) nie wyzerowało linii do niewidocznych.
    #[serde(default = "default_line_thickness")]
    pub line_thickness: f32,
    /// Rysowanie kropek kanałów (ślad energii każdego głośnika, z zanikiem
    /// w czasie) — NIEZALEŻNE od niezależnych wskaźników kierunku (onsety,
    /// patrz `ui::radar_egui::DirectionMarker`), które rysują się zawsze.
    /// Wyłączenie zostawia samą tarczę + markery, bez "szumu" tła kropek.
    /// `#[serde(default = ...)]` na `true`, żeby wczytanie starszego pliku
    /// konfiguracji (sprzed tego pola) zachowało dotychczasowy wygląd.
    #[serde(default = "default_show_channel_dots")]
    pub show_channel_dots: bool,
    /// Styl efektu (patrz [`MonoCueStyle`]/`mono_router::apply_cue`) nakładany
    /// na składową POCHODZĄCĄ z LEWEGO kanału źródłowego, ZAWSZE — niezależnie
    /// od tego, które ucho (`MonoEar::Left`/`Right`) faktycznie gra. Razem z
    /// [`Self::right_cue_style`] (osobny styl dla prawej składowej) pozwala w
    /// JEDNYM, fizycznym uchu odróżnić po samym brzmieniu, czy dźwięk
    /// pochodził z lewego czy prawego kanału — bez tego (dawne zachowanie,
    /// jedno pole `mono_cue_style`) tylko strona PRZECIWNA do granego ucha
    /// dostawała efekt, a własna strona zawsze grała surowo, co dawało tylko
    /// dwie możliwe tekstury (surowo / jeden efekt) zamiast dwóch NIEZALEŻNYCH.
    /// Domyślnie `Off` (surowo) — zachowuje dawne zachowanie trybu `Left`
    /// (własna, lewa strona zawsze grała bez zmian), patrz `mono_router::fill_output_f32`.
    #[serde(default = "default_left_cue_style")]
    pub left_cue_style: MonoCueStyle,
    /// Jak [`Self::left_cue_style`], ale dla składowej z PRAWEGO kanału
    /// źródłowego. Domyślnie [`MonoCueStyle::Buzz`] (dawny domyślny wariant
    /// enuma) — zachowuje dawne zachowanie trybu `Left` (strona przeciwna,
    /// czyli prawa, dostawała `Buzz`). Uwaga: w trybie `MonoEar::Right`
    /// domyślne wartości są ODWROTNE względem dawnego zachowania (dawniej to
    /// LEWA strona była tam "przeciwna" i dostawała efekt) — kto korzysta z
    /// ucha Prawego, powinien zamienić te dwa style miejscami w ustawieniach.
    #[serde(default = "default_right_cue_style")]
    pub right_cue_style: MonoCueStyle,
    /// Włącza odejmowanie energii wspólnej dla wszystkich kanałów kierunkowych
    /// przed sprawdzeniem onsetu (patrz [`crate::sound::direction::CONTRAST_AMOUNT`]
    /// i doc-komentarz [`crate::sound::direction::detect_channel_activity`]) —
    /// tłumi dźwięk obecny mniej więcej RÓWNO na wszystkich kanałach naraz
    /// (otoczenie, muzyka w tle), dzięki czemu wyraźnie JEDNOSTRONNE zdarzenie
    /// na tle takiego szumu łatwiej przebija próg onsetu, nawet gdy samo w
    /// sobie nie jest dość głośne względem WŁASNEJ historii. Działa tylko
    /// przy >2 kanałach kierunkowych (5.1/7.1) — przy zwykłym stereo L≈R to
    /// normalny zapis wyśrodkowanego dźwięku (dialog), nie oznaka otoczenia,
    /// a przy bardzo cichych sygnałach blisko progu ciszy odejmowanie
    /// mogłoby zgasić dźwięk, który bez kontrastu jeszcze by przebił próg.
    /// Domyślnie wyłączone (`#[serde(default)]`) — nowa, opcjonalna poprawka,
    /// włączana checkboxem w ustawieniach, żeby dało się ją porównać na
    /// żywym audio i wyłączyć, gdyby gdzieś zaszkodziła.
    #[serde(default)]
    pub direction_contrast_enabled: bool,
}

fn default_overlay_click_through() -> bool {
    true
}

fn default_show_channel_dots() -> bool {
    true
}

fn default_radar_sensitivity() -> f32 {
    crate::sound::direction::INTENSITY_SCALE
}

fn default_line_thickness() -> f32 {
    1.0
}

fn default_left_cue_style() -> MonoCueStyle {
    MonoCueStyle::Off
}

fn default_right_cue_style() -> MonoCueStyle {
    MonoCueStyle::Buzz
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            anchor_x: 1.0,
            anchor_y: 1.0,
            radius:   125.0,
            bg_alpha: 0.82,
            fg_alpha: 1.0,
            show_labels: false,
            high_contrast: false,
            decay_secs: 2.5,
            color_rules: spectrum::default_color_rules(),
            overlay_click_through: true,
            mono_boost_db: 0.0,
            radar_sensitivity: default_radar_sensitivity(),
            edge_glow_enabled: false,
            line_thickness: default_line_thickness(),
            show_channel_dots: default_show_channel_dots(),
            left_cue_style: default_left_cue_style(),
            right_cue_style: default_right_cue_style(),
            direction_contrast_enabled: false,
        }
    }
}

/// Wybór ucha do odtwarzania mono.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonoEar {
    /// Mono routing wyłączony — brak strumienia wyjściowego.
    #[default]
    Off,
    /// Mono sygnał odtwarzany tylko na lewym kanale.
    Left,
    /// Mono sygnał odtwarzany tylko na prawym kanale.
    Right,
    /// Mono sygnał odtwarzany na obu kanałach.
    Both,
}

impl MonoEar {
    pub const ALL: &'static [Self] = &[Self::Off, Self::Left, Self::Right, Self::Both];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off   => "Off",
            Self::Left  => "← Lewe ucho",
            Self::Right => "→ Prawe ucho",
            Self::Both  => "↔ Oba",
        }
    }
}

/// Sposób oznaczenia w słuchawce dźwięku z PRZECIWNEJ strony niż aktualnie
/// grane ucho w trybie jednousznym (`MonoEar::Left`/`Right`) — bez tego cała
/// informacja L/R ginęłaby, bo obie strony grałyby identycznie w jednym
/// kanale. Każdy wariant to inny, tani obliczeniowo efekt DSP w czasie
/// rzeczywistym (patrz `mono_router::apply_cue`), do wyboru wg tego, co komu
/// łatwiej rozróżnić na ucho.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MonoCueStyle {
    /// Strona przeciwna gra niezmieniona, tak samo jak swoja — brak wskazówki
    /// (zachowanie sprzed dodania tej funkcji).
    Off,
    /// Ring modulation ~45 Hz — wyraźne, niskie "brzęczenie".
    #[default]
    Buzz,
    /// Ring modulation ~220 Hz — ten sam mechanizm co `Buzz`, ale szybciej,
    /// więc brzmi wyżej/"piskliwiej".
    Squeaky,
    /// Rytmiczne wyciszanie/odciszanie (~8 Hz) zamiast płynnej fali —
    /// dźwięk "mruga" zamiast brzęczeć.
    Pulse,
    /// Filtr dolnoprzepustowy — strona przeciwna brzmi stłumiona, "zza
    /// ściany", bez zmiany rytmu głośności (w odróżnieniu od pozostałych
    /// wariantów, które modulują amplitudę).
    Muffled,
    /// Podbicie wysokich tonów (dopełnienie filtra z `Muffled`) — strona
    /// przeciwna brzmi jaśniej/cieńciej, bez zmiany rytmu głośności.
    Bright,
    /// Mocniejszy filtr dolnoprzepustowy niż `Muffled` (niższe odcięcie) —
    /// strona przeciwna brzmi wyraźnie ciemniej/głucho.
    Dark,
    /// Redukcja efektywnej częstotliwości próbkowania (sample-and-hold) +
    /// kwantyzacja do garści poziomów — cyfrowy, chropowaty "chrzęst",
    /// jakościowo inny niż filtry (Muffled/Dark/Bright) czy modulacja
    /// amplitudy (Buzz/Squeaky/Pulse). Patrz `mono_router::bitcrush_step`.
    Bitcrush,
    /// PRAWDZIWA modulacja pierścieniowa: mnożenie sygnału przez nośną falę
    /// sinusoidalną BEZPOŚREDNIO (nie jako obwiednię jak `Buzz`/`Squeaky`,
    /// które mimo nazwy w kodzie są technicznie tremolo) — daje inharmoniczne
    /// wstęgi boczne, metaliczne/"kosmiczne" brzmienie. Patrz
    /// `mono_router::ring_mod_step`.
    RingMod,
    /// Modulowane, krótkie opóźnienie (kilka ms, oscylujące wolno) zamiast
    /// filtra czy modulacji głośności — moduluje WYSOKOŚĆ dźwięku, dając
    /// efekt "pływania"/wibrata. Patrz `mono_router::vibrato_step`.
    Vibrato,
    /// Wyraźne, pojedyncze powtórzenie (echo) z opóźnieniem ~120ms i lekkim
    /// sprzężeniem zwrotnym — wrażenie odległości/przestrzeni zamiast barwy
    /// czy rytmu głośności. Patrz `mono_router::echo_step`.
    Echo,
    /// Rytmiczny wzorzec "stuk-stuk-...-cisza" (dwa krótkie stuknięcia,
    /// dłuższa przerwa) zamiast prostego bramkowania 0/1 jak `Pulse` —
    /// badania nad earconami wskazują, że rozpoznawalność wzorców
    /// rytmicznych jest silniejsza niż samej wysokości dźwięku. Patrz
    /// `mono_router::double_tap_step`.
    DoubleTap,
}

impl MonoCueStyle {
    pub const ALL: &'static [Self] = &[
        Self::Off,
        Self::Buzz,
        Self::Squeaky,
        Self::Pulse,
        Self::Muffled,
        Self::Bright,
        Self::Dark,
        Self::Bitcrush,
        Self::RingMod,
        Self::Vibrato,
        Self::Echo,
        Self::DoubleTap,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Wyłączone",
            Self::Buzz => "Brzęczenie",
            Self::Squeaky => "Bardziej piskliwy",
            Self::Pulse => "Pulsowanie",
            Self::Muffled => "Stłumiony",
            Self::Bright => "Jasny",
            Self::Dark => "Ciemny",
            Self::Bitcrush => "Chrzęst cyfrowy",
            Self::RingMod => "Metaliczny",
            Self::Vibrato => "Wibrato",
            Self::Echo => "Echo",
            Self::DoubleTap => "Podwójne stuknięcie",
        }
    }
}

/// Pojedyncze, niezależne zdarzenie dźwiękowe (onset) wykryte na jednym
/// kanale w ostatniej ramce analizy — patrz [`crate::sound::direction::detect_onsets`].
/// W przeciwieństwie do dawnego, sumarycznego `angle`/`intensity` powyżej
/// (jeden wektor dla całej ramki, który dwa przeciwstawne źródła o podobnej
/// energii potrafi wzajemnie znieść), tu każde zdarzenie niesie WŁASNY,
/// płynny kąt i WŁASNY kolor. Nie niesie znacznika czasu — `spawned_at`
/// nadaje dopiero `ui::radar_egui::RadarState::update`, która ma dostęp do
/// zegara klatki UI (`egui::InputState::time`), spójnego z resztą wskaźników
/// na radarze; wątek audio nie powinien stemplować czasu na własną rękę.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionEvent {
    pub angle_deg: f32,
    pub intensity: f32,
    pub color_rgb: (u8, u8, u8),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioAnalysis {
    pub peak_l: f32,
    pub peak_r: f32,
    /// Poziom RMS każdego kanału z ostatniej ramki analizy (patrz
    /// [`crate::sound::direction::analyze_channels`]) — jedna wartość na
    /// kanał, w kolejności zgodnej z układem głośnikowym danego strumienia.
    pub channel_rms: Vec<f32>,
    /// Liczba kanałów aktywnego strumienia (2, 6, 8, ...), decyduje o
    /// interpretacji `channel_rms` przez [`crate::sound::direction::channel_layout`].
    pub channel_count: usize,
    /// Energia w każdej regule kolorowania (`radar_config.color_rules`) dla
    /// każdego kanału z ostatniej ramki — `channel_rule_energies[kanał][reguła]`.
    /// Patrz [`crate::sound::spectrum`]. Używane do kolorowania kropek radaru
    /// wg "typu" dźwięku, niezależnie od kierunku (kąta) kanału.
    pub channel_rule_energies: Vec<Vec<f32>>,
    /// Niezależne zdarzenia kierunku (onsety) wykryte w ostatniej ramce —
    /// patrz [`DirectionEvent`]. Zasila `RadarState::update`, które zamienia
    /// każde w osobny, gasnący wskaźnik na radarze.
    pub active_direction_events: Vec<DirectionEvent>,
    /// Indeks scenariusza testowego: 0 = live audio, 1..N = kolejne testy.
    /// Ustawiany przez UI Dioxus, odczytywany przez egui.
    pub test_scenario: u8,
    /// Konfiguracja wyglądu i pozycji radaru egui.
    pub radar_config: RadarConfig,
    /// Wybór ucha do mono routingu. Ustawiany przez UI, czytany przez mono_router.
    pub mono_ear: MonoEar,
}

/// Pojemność bufora kołowego próbek mono — ~340ms przy 48kHz, zaokrąglone
/// w górę do potęgi dwójki. To TWARDY limit pamięci (nigdy nie rośnie ponad
/// to), NIE docelowe opóźnienie — patrz [`RING_TRIM_THRESHOLD_SAMPLES`] niżej,
/// który utrzymuje faktyczną głębokość bufora dużo niżej na co dzień.
const MONO_RING_CAPACITY: usize = 16_384;

/// Docelowa głębokość bufora kołowego w próbkach, do której [`SharedAudioAnalysis::push_stereo_pan`]
/// aktywnie przycina, gdy przekroczy [`RING_TRIM_THRESHOLD_SAMPLES`] — margines
/// na drobne (rzędu pojedynczej ramki audio) wahania tempa callbacków, nie na
/// trwały dryf zegarów. ~50ms przy 48kHz.
const TARGET_RING_LATENCY_SAMPLES: usize = 2_400;

/// Próg, po przekroczeniu którego bufor jest od razu przycinany z powrotem do
/// [`TARGET_RING_LATENCY_SAMPLES`], zamiast czekać, aż urośnie do
/// [`MONO_RING_CAPACITY`]. Wejście i wyjście to dwa NIEZALEŻNE zegary
/// sprzętowe (patrz doc-komentarz [`SharedAudioAnalysis::mono_ring`]) — gdy
/// wejście choćby ułamek procenta wyprzedza tempem wyjście, bufor bez tego
/// progu monotonicznie rośnie i UTYKA na twardym limicie (~340ms), więc
/// KAŻDA odtworzona próbka jest odtąd trwale spóźniona o tyle. Przycinanie
/// dużo wcześniej, przy przekroczeniu tego niższego progu, utrzymuje
/// opóźnienie blisko `TARGET_RING_LATENCY_SAMPLES` bez względu na kierunek
/// dryfu. ~150ms przy 48kHz — trzykrotność celu, żeby nie przycinać przy
/// zwykłym, chwilowym jitterze.
const RING_TRIM_THRESHOLD_SAMPLES: usize = TARGET_RING_LATENCY_SAMPLES * 3;

/// Licznik do przerzedzenia diagnostycznego logu głębokości `mono_ring` w
/// [`SharedAudioAnalysis::push_stereo_pan`] — bez tego logowalibyśmy co
/// callback (co ~10ms), zalewając `debug.log`.
static PUSH_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Licznik do przerzedzenia logów `[ch]`/`[analysis]` w [`update_analysis`] —
/// wcześniej logowane BEZWARUNKOWO co callback audio (~10ms, ~100x/s), co
/// realnie narosło do dziesiątek tysięcy linii w kilka minut i było powodem,
/// dla którego `debug_log::init()` zostało tymczasowo wyłączone w `main.rs`.
/// Ta sama częstotliwość co [`PUSH_LOG_COUNTER`] (~co 2s przy 480 klatkach/48kHz).
static ANALYSIS_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Co który callback logować [`ANALYSIS_LOG_COUNTER`]/[`PUSH_LOG_COUNTER`] —
/// ~2s przy 480 klatkach/48kHz (480/48000 * 200 ≈ 2s).
const DIAGNOSTIC_LOG_EVERY_N_CALLBACKS: u64 = 200;

#[derive(Debug, Clone)]
pub struct SharedAudioAnalysis {
    inner: Arc<Mutex<AudioAnalysis>>,
    /// Bufor kołowy próbek "ważonego mono" — KAŻDA para to `(left_weight,
    /// right_weight)`, czyli ile energii danej ramki pochodzi z kanałów
    /// lewej, a ile z prawej strony (patrz [`downmix_to_stereo_pan`]),
    /// zamiast jednego płaskiego uśrednienia wszystkich kanałów. Dzięki temu
    /// `mono_router` może w trybie jednousznym oznaczyć dźwięk z przeciwnej
    /// strony efektem (ring modulation), zamiast bezpowrotnie tracić
    /// informację L/R już na etapie miksu. Zasilany przez TEN SAM strumień
    /// wejściowy, który karmi radar (`update_analysis`), i czytany przez
    /// `mono_router` przy odtwarzaniu. Wcześniej `mono_router` otwierał
    /// WŁASNY, niezależny strumień wejściowy na nazwie urządzenia z tej samej
    /// listy co radar — na Windows ta lista zawiera nazwy urządzeń
    /// WYJŚCIOWYCH (do przechwytywania pętlą zwrotną WASAPI, patrz
    /// `fetch_output_devices`), a `mono_router` szukał ich tylko wśród
    /// `host.input_devices()`, więc lookup zawsze się nie powodził i routing
    /// w ogóle nie startował (błąd widoczny tylko w konsoli, niewidoczny w UI).
    mono_ring: Arc<Mutex<VecDeque<(f32, f32)>>>,
    /// Ostatnia próbka realnie zdjęta z `mono_ring` przez [`Self::read_stereo_pan`]
    /// — trzymana, żeby przy podkręceniu bufora (patrz tam) powtórzyć ją
    /// zamiast wstawiać twarde zero.
    last_pan: Arc<Mutex<(f32, f32)>>,
    /// Stan średniej kroczącej per kanał dla [`crate::sound::direction::detect_onsets`],
    /// który musi przetrwać między kolejnymi wywołaniami `update_analysis`
    /// (jeden callback cpal na ramkę) — stąd trzymany tu, obok `mono_ring`,
    /// zamiast jako zmienna lokalna.
    onset_ema: Arc<Mutex<Vec<f32>>>,
    /// Jak [`Self::onset_ema`], ale per kanał×pasmo kolorowania — zasila
    /// sprawdzanie per-pasmowe w [`direction::detect_channel_activity`]
    /// (patrz doc-komentarz tam), dzięki któremu cichy dźwięk w jednym paśmie
    /// generuje onset, nawet gdy energia szerokopasmowa kanału go nie widzi.
    band_ema: Arc<Mutex<Vec<Vec<f32>>>>,
    /// Logger próbek klasyfikacji koloru do ręcznej analizy configu — patrz
    /// [`crate::sound::sample_logger`]. Wyłączony domyślnie; włączany
    /// checkboxem w UI (`ui::app`).
    sample_logger: Arc<sample_logger::SampleLogger>,
}

impl PartialEq for SharedAudioAnalysis {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Default for SharedAudioAnalysis {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AudioAnalysis::default())),
            mono_ring: Arc::new(Mutex::new(VecDeque::with_capacity(MONO_RING_CAPACITY))),
            last_pan: Arc::new(Mutex::new((0.0, 0.0))),
            onset_ema: Arc::new(Mutex::new(Vec::new())),
            band_ema: Arc::new(Mutex::new(Vec::new())),
            sample_logger: Arc::new(sample_logger::SampleLogger::new()),
        }
    }
}

impl SharedAudioAnalysis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, AudioAnalysis>> {
        self.inner.lock()
    }

    /// Dopisuje próbki "ważonego mono" (`(left_weight, right_weight)`, patrz
    /// [`downmix_to_stereo_pan`]) do bufora kołowego czytanego przez strumień
    /// wyjściowy `mono_router`. Najstarsze próbki są odrzucane, gdy bufor
    /// jest pełny — routing ma pokazywać "teraz", nie gonić zaległości.
    ///
    /// Dodatkowo: gdy bufor urośnie ponad [`RING_TRIM_THRESHOLD_SAMPLES`]
    /// (bo wejście trwale wyprzedza tempem wyjście — patrz doc-komentarz
    /// tam), od razu przycina go z powrotem do [`TARGET_RING_LATENCY_SAMPLES`]
    /// zamiast czekać na twardy limit [`MONO_RING_CAPACITY`] — bez tego
    /// opóźnienie odtwarzanego dźwięku trwale rosło do ~340ms i tam zostawało.
    pub(crate) fn push_stereo_pan(&self, samples: &[(f32, f32)]) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            for &s in samples {
                if ring.len() >= MONO_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(s);
            }
            if ring.len() > RING_TRIM_THRESHOLD_SAMPLES {
                let excess = ring.len() - TARGET_RING_LATENCY_SAMPLES;
                for _ in 0..excess {
                    ring.pop_front();
                }
                crate::dlog!(
                    "[mono_ring] przycięto {excess} próbek (wejście trwale wyprzedzało wyjście) -> {} (~{:.0}ms @48kHz)",
                    ring.len(),
                    ring.len() as f32 / 48.0
                );
            }
            let n = PUSH_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
            if n % DIAGNOSTIC_LOG_EVERY_N_CALLBACKS == 0 {
                crate::dlog!("[mono_ring] depth={} (~{:.0}ms @48kHz)", ring.len(), ring.len() as f32 / 48.0);
            }
        }
    }

    /// Czyta dokładnie `count` próbek "ważonego mono". Wejście i wyjście to
    /// dwa NIEZALEŻNE zegary sprzętowe (osobne urządzenia audio) — nawet przy
    /// tej samej nominalnej częstotliwości próbkowania powoli się rozjeżdżają,
    /// więc bufor kołowy co jakiś czas na chwilę wysycha (podkręcanie), zanim
    /// dogoni go kolejna porcja z wejścia. Wcześniej brakujące miejsce
    /// wypełnialiśmy twardym zerem — nagła cisza-w-środku-sygnału słyszalna
    /// jako pojedyncza ścinka/trzask co parę sekund. Zamiast tego powtarzamy
    /// (`last_pan`) ostatnią realnie odczytaną próbkę, lekko wygaszając ją co
    /// klatkę — jedna-dwie powtórzone próbki są niesłyszalne (zero-order
    /// hold), a przy dłuższym zaniku (przechwytywanie faktycznie stoi) i tak
    /// płynnie spada do ciszy zamiast zapętlać się w nieskończoność.
    pub(crate) fn read_stereo_pan(&self, count: usize) -> Vec<(f32, f32)> {
        const HOLD_DECAY: f32 = 0.999;
        let mut out = Vec::with_capacity(count);
        let mut last = self.last_pan.lock().map(|g| *g).unwrap_or((0.0, 0.0));
        if let Ok(mut ring) = self.mono_ring.lock() {
            for _ in 0..count {
                let sample = match ring.pop_front() {
                    Some(s) => {
                        last = s;
                        s
                    }
                    None => {
                        last = (last.0 * HOLD_DECAY, last.1 * HOLD_DECAY);
                        last
                    }
                };
                out.push(sample);
            }
        } else {
            out.resize(count, (0.0, 0.0));
        }
        if let Ok(mut guard) = self.last_pan.lock() {
            *guard = last;
        }
        out
    }

    /// Czy logger próbek (`sample_logger::SampleLogger`) jest aktualnie
    /// włączony — odczytywane przez checkbox w UI, żeby odzwierciedlić stan
    /// nawet po tym, jak zapis pliku faktycznie się nie powiódł.
    pub fn sample_logging_enabled(&self) -> bool {
        self.sample_logger.is_enabled()
    }

    /// Włącza/wyłącza logger próbek. Włączenie ZAWSZE zaczyna nowy plik —
    /// patrz [`sample_logger::SampleLogger::set_enabled`].
    pub fn set_sample_logging_enabled(&self, enabled: bool) {
        self.sample_logger.set_enabled(enabled);
    }
}

#[derive(Debug, Clone)]
pub struct SoundDevice {
    pub id: usize,
    pub name: String,
    pub is_input: bool,
    pub is_default: bool,
    pub configs: Vec<DeviceConfig>,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub channels: u16,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub sample_format: cpal::SampleFormat,
    pub buffer_size: cpal::SupportedBufferSize,
}

impl SoundDevice {
    pub fn get_device_direction(&self) -> &'static str {
        if self.is_input { "Input" } else { "Output" }
    }

    pub fn get_default_status(&self) -> &'static str {
        if self.is_default { "Yes" } else { "No" }
    }
}

/// Na macOS zwraca urządzenia wejściowe (np. BlackHole) zamiast wyjściowych —
/// cpal na CoreAudio nie obsługuje loopback na urządzeniach wyjściowych.
/// Na Windows/Linux zwraca urządzenia wyjściowe (WASAPI loopback działa natywnie).
pub fn fetch_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();

    #[cfg(target_os = "macos")]
    {
        let default_name = host
            .default_input_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        for (index, device) in host.input_devices()?.enumerate() {
            let Ok(name) = device.name() else { continue };
            let is_default = default_name.as_deref() == Some(&name);

            let mut configs = Vec::new();
            if let Ok(supported) = device.supported_input_configs() {
                for c in supported {
                    configs.push(DeviceConfig {
                        channels: c.channels(),
                        min_sample_rate: c.min_sample_rate().0,
                        max_sample_rate: c.max_sample_rate().0,
                        sample_format: c.sample_format(),
                        buffer_size: *c.buffer_size(),
                    });
                }
            }

            devices.push(SoundDevice {
                id: index,
                name,
                is_input: true,
                is_default,
                configs,
            });
        }
        return Ok(devices);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let default_name = host
            .default_output_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        for (index, device) in host.output_devices()?.enumerate() {
            let Ok(name) = device.name() else { continue };
            let is_default = default_name.as_deref() == Some(&name);

            let mut configs = Vec::new();
            if let Ok(supported) = device.supported_output_configs() {
                for c in supported {
                    configs.push(DeviceConfig {
                        channels: c.channels(),
                        min_sample_rate: c.min_sample_rate().0,
                        max_sample_rate: c.max_sample_rate().0,
                        sample_format: c.sample_format(),
                        buffer_size: *c.buffer_size(),
                    });
                }
            }

            devices.push(SoundDevice {
                id: index,
                name,
                is_input: false,
                is_default,
                configs,
            });
        }
        return Ok(devices);
    }
}

/// Zwraca fizyczne urządzenia wyjściowe (słuchawki, głośniki) na wszystkich platformach.
/// Używane do wyboru urządzenia docelowego dla mono routera.
pub fn fetch_physical_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());

    let mut devices = Vec::new();
    for (index, device) in host.output_devices()?.enumerate() {
        let Ok(name) = device.name() else { continue };
        let is_default = default_name.as_deref() == Some(&name);

        let mut configs = Vec::new();
        if let Ok(supported) = device.supported_output_configs() {
            for c in supported {
                configs.push(DeviceConfig {
                    channels: c.channels(),
                    min_sample_rate: c.min_sample_rate().0,
                    max_sample_rate: c.max_sample_rate().0,
                    sample_format: c.sample_format(),
                    buffer_size: *c.buffer_size(),
                });
            }
        }

        devices.push(SoundDevice {
            id: index,
            name,
            is_input: false,
            is_default,
            configs,
        });
    }
    Ok(devices)
}

/// Preferowany rozmiar bufora audio w klatkach — mniejszy bufor oznacza
/// mniejsze opóźnienie między dźwiękiem w grze a reakcją radaru/cue w
/// słuchawkach (patrz `crate::sound::mono_router`, który używa tej samej
/// funkcji dla strumienia wyjściowego), kosztem nieco większego ryzyka
/// przycięć (underrun) na wolniejszym sprzęcie — wejście i wyjście to dwa
/// NIEZALEŻNE zegary sprzętowe (patrz doc-komentarz `SharedAudioAnalysis::mono_ring`),
/// więc im mniejszy bufor, tym mniej zapasu na ich chwilowe rozjechanie.
///
/// UWAGA: pierwotnie `256` (~5,3 ms przy 48 kHz) — po zgłoszeniu słyszalnych
/// "przycięć" w dźwięku przekierowanym przez `mono_router` podniesione do
/// `480` (~10 ms), bezpieczniejszej, powszechnie używanej wartości dla trybu
/// "niskie opóźnienie" na zwykłym WASAPI w trybie współdzielonym (bez
/// wyłączności na urządzenie) — wciąż wyraźnie poniżej typowego domyślnego
/// bufora systemowego, ale z większym zapasem na wahania harmonogramowania OS.
pub(crate) const PREFERRED_BUFFER_FRAMES: u32 = 480;

/// Wybiera stały, mały rozmiar bufora ([`PREFERRED_BUFFER_FRAMES`]), jeśli
/// urządzenie faktycznie deklaruje obsługiwany zakres — część sterowników
/// (typowo WASAPI w trybie współdzielonym) zgłasza `Unknown` i wtedy
/// zostajemy przy `BufferSize::Default` wybieranym przez system, żeby nie
/// zażądać wartości spoza tego, co urządzenie faktycznie obsługuje.
pub(crate) fn low_latency_buffer_size(supported: &cpal::SupportedBufferSize) -> cpal::BufferSize {
    match *supported {
        cpal::SupportedBufferSize::Range { min, max } => {
            cpal::BufferSize::Fixed(PREFERRED_BUFFER_FRAMES.clamp(min, max))
        }
        cpal::SupportedBufferSize::Unknown => cpal::BufferSize::Default,
    }
}

pub fn fetch_audio_from_device(
    target_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    // Na macOS szukamy w urządzeniach wejściowych (BlackHole itp.)
    // Na Windows szukamy najpierw w wyjściowych (WASAPI loopback), potem wejściowych
    #[cfg(target_os = "macos")]
    let (target_device, is_output) = {
        if let Some(dev) = host
            .input_devices()?
            .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
        {
            (dev, false)
        } else {
            anyhow::bail!("Device not found: {}", target_device_name);
        }
    };

    #[cfg(not(target_os = "macos"))]
    let (target_device, is_output) = {
        if let Some(dev) = host
            .output_devices()?
            .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
        {
            (dev, true)
        } else if let Some(dev) = host
            .input_devices()?
            .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
        {
            (dev, false)
        } else {
            anyhow::bail!("Device not found: {}", target_device_name);
        }
    };

    let default_config = if is_output {
        target_device
            .default_output_config()
            .or_else(|_| target_device.default_input_config())?
    } else {
        target_device
            .default_input_config()
            .or_else(|_| target_device.default_output_config())?
    };

    let mut config: StreamConfig = default_config.clone().into();
    config.buffer_size = low_latency_buffer_size(default_config.buffer_size());
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;
    let err_fn = |e| crate::dlog!("Stream error: {e}");

    crate::dlog!(
        "[audio] stream start: device={:?} format={:?} rate={}Hz channels={}",
        target_device_name,
        default_config.sample_format(),
        config.sample_rate.0,
        channels
    );

    let call_counter = Arc::new(AtomicU64::new(0));

    let stream = match default_config.sample_format() {
        SampleFormat::F32 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    if n % 200 == 0 {
                        let peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        crate::dlog!("[audio] cb #{n} samples={} peak={peak:.4}", data.len());
                    }
                    update_analysis(data, channels, sample_rate, &a);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    let f: Vec<f32> = data.iter().map(|&s| direction::i16_to_f32(s)).collect();
                    if n % 200 == 0 {
                        let peak = f.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        crate::dlog!("[audio] cb #{n} samples={} peak={peak:.4}", f.len());
                    }
                    update_analysis(&f, channels, sample_rate, &a);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    let f: Vec<f32> = data.iter().map(|&s| direction::u16_to_f32(s)).collect();
                    if n % 200 == 0 {
                        let peak = f.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        crate::dlog!("[audio] cb #{n} samples={} peak={peak:.4}", f.len());
                    }
                    update_analysis(&f, channels, sample_rate, &a);
                },
                err_fn,
                None,
            )?
        }
        fmt => anyhow::bail!("Unsupported sample format: {fmt}"),
    };

    stream.play()?;
    Ok(stream)
}

fn update_analysis(samples: &[f32], channels: usize, sample_rate: f32, analysis: &SharedAudioAnalysis) {
    let stats = analyze_channels(samples, channels);

    // Reguły kolorowania i czułość mogą się zmienić w dowolnym momencie z
    // panelu ustawień (UI), więc czytamy je świeżo przy każdej ramce zamiast
    // zakładać stałe wartości.
    let (rules, sensitivity, contrast_enabled) = analysis
        .lock()
        .map(|g| {
            (
                g.radar_config.color_rules.clone(),
                g.radar_config.radar_sensitivity,
                g.radar_config.direction_contrast_enabled,
            )
        })
        .unwrap_or_else(|_| (spectrum::default_color_rules(), default_radar_sensitivity(), false));
    let rule_energies = spectrum::analyze_channel_rule_energies(samples, channels, sample_rate, &rules);

    // Niezależne zdarzenia kierunku: KAŻDY nagły skok energii na KAŻDYM
    // kanale osobno, zamiast jednego sumarycznego wektora, który przy dwóch
    // przeciwstawnych źródłach o podobnej energii potrafi się wyzerować
    // (patrz doc-komentarz `direction::detect_onsets`). Ten sam suwak
    // "Czułość radaru", który już reguluje jak duże/daleko są kropki kanałów,
    // reguluje też PRÓG detekcji onsetów — bez tego, przy cichszych źródłach
    // (patrz log diagnostyczny), detekcja odpalała się raz na starcie i
    // prawie nigdy więcej, niezależnie od tego, jak wysoko user podniósłby
    // suwak.
    // Stan KAŻDEGO kanału z kierunkiem (nie tylko tych, które przebiły próg
    // onsetu) — `direction_events` niżej filtruje z tego tylko onsety
    // (identycznie jak dawne `detect_onsets`), a logger próbek (niżej) widzi
    // WSZYSTKIE, żeby dało się złapać dźwięk, który był słyszalny, ale nigdy
    // nie wygenerował markera (patrz doc-komentarz `direction::ChannelActivity`).
    let channel_activity: Vec<direction::ChannelActivity> = {
        let mut ema_guard = analysis.onset_ema.lock().unwrap_or_else(|e| e.into_inner());
        let mut band_ema_guard = analysis.band_ema.lock().unwrap_or_else(|e| e.into_inner());
        direction::detect_channel_activity(
            &stats.rms,
            &rule_energies,
            &mut ema_guard,
            &mut band_ema_guard,
            sensitivity,
            contrast_enabled,
        )
    };

    let direction_events: Vec<DirectionEvent> = channel_activity
        .iter()
        .filter(|a| a.is_onset)
        .map(|onset| DirectionEvent {
            angle_deg: onset.angle_deg,
            intensity: (onset.energy * sensitivity).clamp(0.0, 1.0),
            color_rgb: spectrum::resolve_color(
                &rules,
                rule_energies.get(onset.channel_index).map(Vec::as_slice).unwrap_or(&[]),
            ),
        })
        .collect();

    // Logger próbek do ręcznego dostrajania `color_rules` — patrz
    // `sample_logger`. Sprawdzamy `is_enabled()` najpierw, żeby przy
    // wyłączonym loggerze nie liczyć nic dodatkowego co ramkę audio.
    // Próg 0.01 to ten sam floor co przy spawnowaniu `ChannelDot`
    // (patrz `ui::radar_egui::RadarState::update`) — poniżej niego kanał
    // uznajemy za ciszę, niewartą logowania.
    if analysis.sample_logger.is_enabled() {
        let unix_ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        for a in &channel_activity {
            if a.scaled_energy < 0.01 {
                continue;
            }
            let energies = rule_energies.get(a.channel_index).map(Vec::as_slice).unwrap_or(&[]);
            let rule_index = spectrum::dominant_rule_index(energies);
            let rule_label = rule_index.and_then(|i| rules.get(i)).and_then(|r| r.label.clone());
            analysis.sample_logger.log_sample(sample_logger::SampleRow {
                unix_ts_ms,
                channel_index: a.channel_index,
                rms: a.energy,
                scaled_energy: a.scaled_energy,
                angle_deg: a.angle_deg,
                sensitivity,
                marker_spawned: a.is_onset,
                rule_index,
                rule_label,
                color: spectrum::resolve_color(&rules, energies),
            });
        }
    }

    // Przerzedzone do ~co 2s (patrz [`ANALYSIS_LOG_COUNTER`]) — bezwarunkowe
    // logowanie co callback (~100x/s) narosło kiedyś do dziesiątek tysięcy
    // linii w kilka minut.
    let analysis_n = ANALYSIS_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
    if analysis_n % DIAGNOSTIC_LOG_EVERY_N_CALLBACKS == 0 {
        let labels = ["L", "R", "C", "LFE", "RL", "RR", "SL", "SR"];
        let ch_str: Vec<String> = stats.rms.iter().enumerate().map(|(i, rms)| {
            let label = labels.get(i).copied().unwrap_or("?");
            let pct = (rms.clamp(0.0, 1.0) * 100.0).round() as u32;
            format!("{label}:{pct}")
        }).collect();
        crate::dlog!("[ch] {} | onsets={}", ch_str.join(" "), direction_events.len());
        crate::dlog!("[analysis] rms={:?}", &stats.rms);
    }
    // Pozycja (kąt na tarczy) i intensywność każdego wskaźnika kierunku —
    // dokładnie to, co dostaje `RadarState::update` i zamienia na osobny,
    // gasnący marker na radarze (patrz `ui::radar_egui::DirectionMarker`).
    // Logowane tylko gdy coś faktycznie wykryto, żeby nie zalewać pliku
    // pustymi liniami przy każdej ramce ciszy.
    for event in &direction_events {
        crate::dlog!(
            "[onset] angle={:.1}° intensity={:.2} color=rgb({},{},{})",
            event.angle_deg,
            event.intensity,
            event.color_rgb.0,
            event.color_rgb.1,
            event.color_rgb.2
        );
    }
    if let Ok(mut lock) = analysis.lock() {
        lock.peak_l = stats.peak.first().copied().unwrap_or(0.0);
        lock.peak_r = stats.peak.get(1).copied().unwrap_or(0.0);
        lock.channel_rms = stats.rms;
        lock.channel_count = channels;
        lock.channel_rule_energies = rule_energies;
        lock.active_direction_events = direction_events;
    }

    // Miks do "ważonego mono" na sam koniec — dopiero tu, dla ewentualnego
    // odtwarzania przez `mono_router`. Radar powyżej już dostał pełne dane
    // per-kanał, więc to nie zastępuje ich, tylko dokłada osobny strumień do
    // wyjścia.
    analysis.push_stereo_pan(&downmix_to_stereo_pan(samples, channels));
}

/// Rozdziela zinterlacowane próbki na dwa "ważone" miksy mono: `(left,
/// right)`, gdzie każdy to udział energii kanałów danej strony w tej ramce
/// (patrz [`channel_pan_weights`]) — w przeciwieństwie do dawnego, płaskiego
/// uśrednienia WSZYSTKICH kanałów, tu informacja L/R przeżywa aż do
/// `mono_router`, który (w trybie jednousznym) może nią oznaczyć dźwięk z
/// przeciwnej strony efektem (ring modulation), zamiast go bezpowrotnie
/// zlewać z resztą.
fn downmix_to_stereo_pan(interleaved: &[f32], channels: usize) -> Vec<(f32, f32)> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        // Mono źródło: brak informacji o stronie, więc dzielimy po równo.
        return interleaved.iter().map(|&s| (s * 0.5, s * 0.5)).collect();
    }
    let layout = direction::channel_layout(channels);
    let weights: Vec<(f32, f32)> = (0..channels)
        .map(|i| layout.get(i).copied().map(channel_pan_weights).unwrap_or((0.5, 0.5)))
        .collect();
    interleaved
        .chunks_exact(channels)
        .map(|frame| {
            let mut left = 0.0f32;
            let mut right = 0.0f32;
            for (i, &s) in frame.iter().enumerate() {
                let (wl, wr) = weights[i];
                left += s * wl;
                right += s * wr;
            }
            (left / channels as f32, right / channels as f32)
        })
        .collect()
}

/// Waga `(lewa, prawa)` pojedynczego kanału przy miksowaniu do "ważonego
/// mono": pełne `1.0` po swojej stronie dla kanałów jednoznacznie bocznych,
/// po połowie dla środka/LFE (brak wyraźnej strony, więc trafiają w obie
/// bez efektu ring modulation w `mono_router`).
fn channel_pan_weights(role: direction::ChannelRole) -> (f32, f32) {
    use direction::ChannelRole::*;
    match role {
        FrontLeft | RearLeft | SideLeft => (1.0, 0.0),
        FrontRight | RearRight | SideRight => (0.0, 1.0),
        Center | Lfe => (0.5, 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_cue_style_all_lists_every_variant_with_a_unique_non_empty_label() {
        // 7 dawnych + 5 nowych (Bitcrush/RingMod/Vibrato/Echo/DoubleTap) —
        // regresja przeciw "dodałem wariant do enuma, zapomniałem dopisać go
        // do ALL" (wtedy nie pojawiłby się w ogóle w pickerze UI).
        assert_eq!(MonoCueStyle::ALL.len(), 12, "spodziewano się dokładnie 12 stylów w ALL");
        let labels: Vec<&str> = MonoCueStyle::ALL.iter().map(|s| s.label()).collect();
        assert!(labels.iter().all(|l| !l.is_empty()), "żaden styl nie powinien mieć pustej etykiety: {labels:?}");
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(labels[i], labels[j], "etykiety stylów muszą być unikalne (żeby dało się je odróżnić w UI)");
            }
        }
    }

    #[test]
    fn radar_config_round_trips_new_cue_styles_through_json() {
        // Regresja: nowe warianty enuma (Bitcrush/RingMod/Vibrato/Echo/
        // DoubleTap) muszą się poprawnie (de)serializować jako pola
        // `left_cue_style`/`right_cue_style` configu, tak samo jak dawne.
        for &style in MonoCueStyle::ALL {
            let cfg = RadarConfig { left_cue_style: style, right_cue_style: style, ..RadarConfig::default() };
            let json = serde_json::to_string(&cfg).expect("serializacja");
            let back: RadarConfig = serde_json::from_str(&json).expect("deserializacja");
            assert_eq!(cfg, back, "round-trip JSON powinien zachować styl {style:?}");
        }
    }

    #[test]
    fn test_device_helpers() {
        let dev = SoundDevice {
            id: 0,
            name: "Test".into(),
            is_input: true,
            is_default: false,
            configs: vec![],
        };
        assert_eq!(dev.get_device_direction(), "Input");
        assert_eq!(dev.get_default_status(), "No");
    }

    #[test]
    fn downmix_to_stereo_pan_keeps_left_and_right_separate_for_plain_stereo() {
        // L=1.0, R=0.5 przez 2 ramki.
        let interleaved = [1.0, 0.5, 1.0, 0.5];
        let pan = downmix_to_stereo_pan(&interleaved, 2);
        assert_eq!(pan, vec![(0.5, 0.25), (0.5, 0.25)]);
    }

    #[test]
    fn downmix_to_stereo_pan_sums_back_to_the_old_flat_mono_average() {
        // (left + right) musi się zgadzać z dawnym płaskim `sum(channels)/channels`
        // dla KAŻDEGO układu kanałów, bo `mono_router::MonoEar::Both` na tym polega.
        let interleaved = [1.0, 0.5, 0.2, -0.4, -0.1, 0.3];
        for &channels in &[1usize, 2, 6] {
            let pan = downmix_to_stereo_pan(&interleaved[..channels.min(interleaved.len())], channels);
            if let Some(&(l, r)) = pan.first() {
                let expected: f32 = interleaved[..channels].iter().sum::<f32>() / channels as f32;
                assert!((l + r - expected).abs() < 1e-6, "channels={channels}");
            }
        }
    }

    #[test]
    fn downmix_to_stereo_pan_center_and_lfe_split_evenly() {
        // 5.1: FrontLeft, FrontRight, Center, Lfe, RearLeft, RearRight — tylko Center głośny.
        let interleaved = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let pan = downmix_to_stereo_pan(&interleaved, 6);
        let (l, r) = pan[0];
        assert!((l - r).abs() < 1e-6, "Center powinien trafić po równo w obie strony: {pan:?}");
        assert!(l > 0.0);
    }

    #[test]
    fn read_stereo_pan_holds_last_sample_on_underrun_instead_of_hard_zero() {
        // Regresja dla "co parę sekund pojedyncza ścinka dźwięku": gdy wejście
        // i wyjście (dwa niezależne zegary audio) rozjadą się na tyle, że
        // bufor na chwilę wyschnie, odczyt NIE może wstawić twardego zera —
        // to słychać jako trzask. Ma powtórzyć (z lekkim wygaszeniem) ostatnią
        // realną próbkę.
        let analysis = SharedAudioAnalysis::new();
        analysis.push_stereo_pan(&[(0.6, 0.4), (0.6, 0.4)]);
        // Prosimy o więcej próbek niż jest w buforze — reszta to "podkręcanie".
        let out = analysis.read_stereo_pan(5);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], (0.6, 0.4));
        assert_eq!(out[1], (0.6, 0.4));
        for &(l, r) in &out[2..] {
            assert!(l > 0.0 && r > 0.0, "podkręcanie ma powtarzać ostatnią próbkę, nie wstawiać ciszy: {out:?}");
            assert!(l <= 0.6 && r <= 0.4, "powtórzona próbka ma się lekko wygaszać, nie rosnąć: {out:?}");
        }
    }

    #[test]
    fn push_stereo_pan_trims_back_to_target_instead_of_growing_to_hard_capacity() {
        // Regresja dla "prawie sekundowego opóźnienia": gdy wejście trwale
        // wyprzedza tempem wyjście (nikt nie woła read_stereo_pan wcale, jak
        // tutaj — skrajny przypadek tego samego zjawiska), bufor bez
        // aktywnego przycinania rósłby aż do MONO_RING_CAPACITY (~340ms) i
        // tam utykał. Powinien zamiast tego zatrzymać się blisko
        // TARGET_RING_LATENCY_SAMPLES (~50ms), NIE przy pełnej pojemności.
        let analysis = SharedAudioAnalysis::new();
        let chunk = vec![(0.1, 0.2); 500];
        for _ in 0..50 {
            analysis.push_stereo_pan(&chunk); // 50 * 500 = 25 000 próbek wepchniętych, bez ani jednego odczytu
        }

        let ring_len = analysis.mono_ring.lock().unwrap().len();
        assert!(
            ring_len <= RING_TRIM_THRESHOLD_SAMPLES,
            "bufor nie powinien urosnąć powyżej progu przycinania, ma {ring_len}"
        );
        assert!(
            ring_len < MONO_RING_CAPACITY,
            "bufor utknął przy pełnej pojemności (~340ms) zamiast się przyciąć, ma {ring_len}"
        );
    }
}
