use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::sound::direction::{self, analyze_channels};
use crate::sound::layers::{self, LayerHit};
use crate::sound::recorder;
use crate::sound::sample_logger;
use crate::sound::spectrum::{self, ColorRule};

/// Tryb wyświetlania wizualizacji audio na overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VisualizationMode {
    /// Tylko radar ze strzałkami kierunków.
    OnlyRadar,
    /// Tylko słupki stereo L/R.
    OnlyBars,
    /// Zarówno radar jak i słupki stereo.
    Both,
    /// Sześć słupków 3+3: po trzy na stronę (strzały, kroki, reszta), patrz `sound::layers`.
    LayerBars,
    /// Nic nie wyświetlaj (overlay wyłączony).
    None,
}

impl VisualizationMode {
    pub const ALL: &'static [Self] = &[
        Self::OnlyRadar,
        Self::OnlyBars,
        Self::Both,
        Self::LayerBars,
        Self::None,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::OnlyRadar => "◉ Tylko radar",
            Self::OnlyBars => "≡ Tylko słupki L/R",
            Self::Both => "◉≡ Radar + słupki",
            Self::LayerBars => "▮▮▮ Warstwy 3+3",
            Self::None => "⊘ Nic",
        }
    }
}

impl Default for VisualizationMode {
    fn default() -> Self {
        Self::Both
    }
}

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
    /// Korekcja barwy sygnału mono na wyjściu, w decybelach: dodatnie = jaśniej
    /// (wysokie w górę, bas w dół), ujemne = ciemniej/cieplej (bas w górę,
    /// wysokie w dół), 0.0 = bez zmian. To pochylenie widma wokół
    /// `mono_router::TONE_CROSSOVER_HZ`, a NIE zmiana wysokości dźwięku (pitch).
    #[serde(default)]
    pub mono_tone_db: f32,
    /// Zmiana głośności w trybie mono z klasyfikacji radaru: strzały i wybuchy ciszej, kroki głośniej.
    /// Dodaje opóźnienie wyjścia (0,25 s), bo klasyfikacja trwa chwilę po początku dźwięku. Domyślnie wyłączona.
    #[serde(default)]
    pub mono_class_gain: bool,
    /// Wielkość zmiany klas w dB: strzały o tyle ciszej, kroki o tyle głośniej (0–12).
    #[serde(default = "default_mono_class_gain_db")]
    pub mono_class_gain_db: f32,
    /// Dynamika Scout w mono: ciche wzmacnia, głośne ścisza. Domyślnie wyłączona.
    #[serde(default)]
    pub mono_scout: bool,
    /// Siła dynamiki Scout od 0 do 1 (domyślnie 0,5).
    #[serde(default = "default_mono_scout_amount")]
    pub mono_scout_amount: f32,
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
    /// Rozróżnienie układu: 2.0 dostaje łuki L/R, 5.1/7.1 pierścień kropek
    /// w pozycjach głośników i etykietę układu. Wyłączone = dawny wygląd.
    #[serde(default)]
    pub speaker_layout_view: bool,
    /// Kolor łuku kanału lewego (2.0). Czerwień celowo poza listą wyboru —
    /// jest zarezerwowana na obrys dominującego kanału.
    #[serde(default = "default_stereo_left_color")]
    pub stereo_left_color: [u8; 3],
    /// Kolor łuku kanału prawego (2.0), patrz `stereo_left_color`.
    #[serde(default = "default_stereo_right_color")]
    pub stereo_right_color: [u8; 3],
    /// Punkt źródła dźwięku na tarczy w układzie 2.0: lewo–prawo z balansu L/R,
    /// przód–tył z widma (wysokie pasma), wysokość jako jasność punktu.
    #[serde(default)]
    pub spatial_audio_view: bool,
    /// Mnożnik grubości WSZYSTKICH linii rysowanych na radarze (okręgi skali,
    /// krzyż kierunków, sektory 7.1, obrys kropek/wskaźników) — `1.0`
    /// odtwarza dawne, sztywne szerokości linii z `radar_egui::draw_radar`.
    /// `#[serde(default = ...)]` na `1.0` (nie na `0.0`, jak dałby zwykły
    /// `Default`), żeby wczytanie starszego pliku konfiguracji (sprzed tego
    /// pola) nie wyzerowało linii do niewidocznych.
    #[serde(default = "default_line_thickness")]
    pub line_thickness: f32,
    /// Jak oznaczyć dźwięk z przeciwnej strony niż wybrane ucho w trybie
    /// jednousznym (`MonoEar::Left`/`Right`), żeby dało się po samym
    /// brzmieniu poznać, że był z drugiej strony — patrz
    /// [`MonoCueStyle`]/`mono_router::fill_output_f32`. `#[serde(default)]`
    /// korzysta z `#[default]` wariantu enuma, więc wczytanie starszego
    /// pliku configu (sprzed tego pola) włącza nową funkcję zamiast ją wyciszyć.
    #[serde(default)]
    pub mono_cue_style: MonoCueStyle,
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
    /// Tryb wyświetlania: radar, słupki L/R, oba razem, albo nic.
    #[serde(default)]
    pub visualization_mode: VisualizationMode,
    /// Promień wewnętrznego łuku zestawu 3+3 (warstwa strzałów), jako ułamek promienia radaru.
    /// `#[serde(default = ...)]`, żeby stary plik konfiguracji dał wartość domyślną.
    #[serde(default = "default_layer_arc_inner")]
    pub layer_arc_inner: f32,
    /// Odstęp promieni łuków kolejnych warstw 3+3, jako ułamek promienia radaru.
    #[serde(default = "default_layer_arc_spacing")]
    pub layer_arc_spacing: f32,
    /// Wzmocnienie wysokości łuków kroków na radarze (1.0 = bez zmian). Pozwala lepiej widzieć
    /// ciche kroki, nie zmienia tego, co zostało wykryte.
    #[serde(default = "default_layer_steps_gain")]
    pub layer_steps_gain: f32,
    /// Wzmocnienie strzałów i reszty na radarze (1.0 = bez zmian). 0.0 chowa parę.
    #[serde(default = "default_layer_gain")]
    pub layer_shots_gain: f32,
    #[serde(default = "default_layer_gain")]
    pub layer_rest_gain: f32,
    /// Kolory każdej pary 3+3 (RGB).
    #[serde(default = "default_layer_shots_color")]
    pub layer_shots_color: [u8; 3],
    #[serde(default = "default_layer_steps_color")]
    pub layer_steps_color: [u8; 3],
    #[serde(default = "default_layer_rest_color")]
    pub layer_rest_color: [u8; 3],
}

fn default_overlay_click_through() -> bool {
    true
}

fn default_mono_scout_amount() -> f32 {
    0.5
}

fn default_mono_class_gain_db() -> f32 {
    MONO_CLASS_GAIN_DB_DEFAULT
}

fn default_radar_sensitivity() -> f32 {
    crate::sound::direction::INTENSITY_SCALE
}

fn default_line_thickness() -> f32 {
    1.0
}

fn default_layer_arc_inner() -> f32 {
    0.35
}

fn default_layer_arc_spacing() -> f32 {
    0.15
}

fn default_layer_steps_gain() -> f32 {
    1.0
}

fn default_layer_gain() -> f32 {
    1.0
}

fn default_layer_shots_color() -> [u8; 3] {
    [255, 140, 40]
}

fn default_layer_steps_color() -> [u8; 3] {
    [150, 255, 90]
}

fn default_layer_rest_color() -> [u8; 3] {
    [80, 170, 255]
}

fn default_stereo_left_color() -> [u8; 3] {
    [0, 150, 255]
}

fn default_stereo_right_color() -> [u8; 3] {
    [100, 200, 50]
}

pub const STEREO_COLOR_PRESETS: &[(&str, [u8; 3])] = &[
    ("Niebieski", [0, 150, 255]),
    ("Zielony", [100, 200, 50]),
    ("Cyjan", [0, 220, 220]),
    ("Żółty", [255, 230, 60]),
    ("Fioletowy", [180, 40, 200]),
    ("Biały", [255, 255, 255]),
];

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            anchor_x: 1.0,
            anchor_y: 1.0,
            radius:   125.0,
            bg_alpha: 0.82,
            fg_alpha: 1.0,
            high_contrast: false,
            decay_secs: 2.5,
            color_rules: spectrum::default_color_rules(),
            overlay_click_through: true,
            mono_boost_db: 0.0,
            mono_tone_db: 0.0,
            mono_class_gain: false,
            mono_class_gain_db: default_mono_class_gain_db(),
            mono_scout: false,
            mono_scout_amount: default_mono_scout_amount(),
            radar_sensitivity: default_radar_sensitivity(),
            edge_glow_enabled: false,
            speaker_layout_view: false,
            stereo_left_color: default_stereo_left_color(),
            stereo_right_color: default_stereo_right_color(),
            spatial_audio_view: false,
            line_thickness: default_line_thickness(),
            mono_cue_style: MonoCueStyle::default(),
            direction_contrast_enabled: false,
            visualization_mode: VisualizationMode::default(),
            layer_arc_inner: default_layer_arc_inner(),
            layer_arc_spacing: default_layer_arc_spacing(),
            layer_steps_gain: default_layer_steps_gain(),
            layer_shots_gain: default_layer_gain(),
            layer_rest_gain: default_layer_gain(),
            layer_shots_color: default_layer_shots_color(),
            layer_steps_color: default_layer_steps_color(),
            layer_rest_color: default_layer_rest_color(),
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

/// Zdarzenie kierunku z chwilą wykrycia w callbacku audio. Kolejka trzyma je,
/// dopóki UI ich nie odbierze, więc zdarzenie nie znika, gdy UI nie zdąży przed
/// kolejnym callbackiem (callback działa co ~10 ms, klatka UI co ~16 ms).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueuedDirectionEvent {
    pub captured_at: Instant,
    pub event: DirectionEvent,
}

/// Najwięcej zdarzeń czekających w kolejce. Gdy UI stoi, odrzucane są najstarsze.
const MAX_QUEUED_DIRECTION_EVENTS: usize = 256;

/// Trafienie warstwy z chwilą decyzji w callbacku audio. Działa tak samo jak
/// [`QueuedDirectionEvent`]: kolejka trzyma je, dopóki UI ich nie odbierze.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueuedLayerHit {
    pub captured_at: Instant,
    pub hit: LayerHit,
}

/// Najwięcej trafień warstw czekających w kolejce (jak [`MAX_QUEUED_DIRECTION_EVENTS`]).
const MAX_QUEUED_LAYER_HITS: usize = 256;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioAnalysis {
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
    /// Niezależne zdarzenia kierunku (onsety) czekające na UI, z chwilą wykrycia.
    /// Callback audio dokłada do kolejki, a UI opróżnia ją co klatkę
    /// (patrz [`AudioAnalysis::drain_direction_events`]).
    pub direction_queue: VecDeque<QueuedDirectionEvent>,
    /// Trafienia warstw 3+3 (strzały/kroki/reszta, strona) czekające na UI, patrz
    /// [`crate::sound::layers`]. Opróżniane razem z `direction_queue`.
    pub layer_queue: VecDeque<QueuedLayerHit>,
    /// Indeks scenariusza testowego: 0 = live audio, 1..N = kolejne testy.
    /// Ustawiany przez UI Dioxus, odczytywany przez egui.
    pub test_scenario: u8,
    /// Konfiguracja wyglądu i pozycji radaru egui.
    pub radar_config: RadarConfig,
    /// Wybór ucha do mono routingu. Ustawiany przez UI, czytany przez mono_router.
    pub mono_ear: MonoEar,
}

impl AudioAnalysis {
    /// Dokłada zdarzenia wykryte w jednym callbacku. Wszystkie dostają tę samą
    /// chwilę `captured_at`, więc dają się rozpoznać jako jedno uderzenie.
    pub fn push_direction_events(&mut self, events: &[DirectionEvent], captured_at: Instant) {
        for event in events {
            if self.direction_queue.len() >= MAX_QUEUED_DIRECTION_EVENTS {
                self.direction_queue.pop_front();
            }
            self.direction_queue.push_back(QueuedDirectionEvent {
                captured_at,
                event: *event,
            });
        }
    }

    /// Oddaje wszystkie zdarzenia z kolejki, w kolejności wykrycia, i czyści ją.
    pub fn drain_direction_events(&mut self) -> Vec<QueuedDirectionEvent> {
        self.direction_queue.drain(..).collect()
    }

    /// Dokłada trafienia warstw wykryte w jednym callbacku (wszystkie z tą samą chwilą).
    pub fn push_layer_hits(&mut self, hits: &[LayerHit], captured_at: Instant) {
        for hit in hits {
            if self.layer_queue.len() >= MAX_QUEUED_LAYER_HITS {
                self.layer_queue.pop_front();
            }
            self.layer_queue.push_back(QueuedLayerHit {
                captured_at,
                hit: *hit,
            });
        }
    }

    /// Oddaje wszystkie trafienia warstw z kolejki, w kolejności wykrycia, i czyści ją.
    pub fn drain_layer_hits(&mut self) -> Vec<QueuedLayerHit> {
        self.layer_queue.drain(..).collect()
    }
}

/// Pojemność bufora kołowego próbek mono — ok. 1,36 s przy 48 kHz (potęga dwójki). To TWARDY
/// limit pamięci, NIE docelowe opóźnienie. Opóźnienie utrzymuje [`TARGET_RING_LATENCY_SAMPLES`].
const MONO_RING_CAPACITY: usize = 65_536;

/// Docelowe opóźnienie wyjścia mono względem wejścia, w trybie zwykłym (50 ms przy 48 kHz).
/// Z włączoną zmianą głośności klas używamy [`CLASS_GAIN_LATENCY_SAMPLES`].
const TARGET_RING_LATENCY_SAMPLES: usize = 2_400;

/// Opóźnienie wyjścia z włączoną zmianą głośności klas (0,25 s przy 48 kHz). Tyle trwa
/// klasyfikacja zdarzenia (ok. 0,15 s) plus zapas, żeby zmiana zdążyła przed początkiem dźwięku.
const CLASS_GAIN_LATENCY_SAMPLES: usize = 12_000;

/// Próg, po przekroczeniu którego bufor jest od razu przycinany z powrotem do
/// [`TARGET_RING_LATENCY_SAMPLES`], zamiast czekać, aż urośnie do
/// [`MONO_RING_CAPACITY`]. Wejście i wyjście to dwa NIEZALEŻNE zegary
/// sprzętowe (patrz doc-komentarz [`SharedAudioAnalysis::mono_ring`]) — gdy
/// wejście choćby ułamek procenta wyprzedza tempem wyjście, bufor bez tego
/// progu monotonicznie rośnie i UTYKA na twardym limicie (~340ms), więc
/// KAŻDA odtworzona próbka jest odtąd trwale spóźniona o tyle. Przycinanie
/// dużo wcześniej, przy przekroczeniu tego niższego progu, utrzymuje
/// opóźnienie blisko `TARGET_RING_LATENCY_SAMPLES` bez względu na kierunek
/// dryfu. ~150 ms przy 48 kHz — trzykrotność celu, żeby nie przycinać przy
/// zwykłym, chwilowym jitterze.
const RING_TRIM_FACTOR: usize = 3;

/// Domyślna zmiana głośności na słuchawkach w trybie mono (dB): strzały i wybuchy o tyle ciszej,
/// kroki o tyle głośniej. Suwak w panelu ustawia wartość z zakresu 0–12 dB.
const MONO_CLASS_GAIN_DB_DEFAULT: f32 = 6.0;
/// Jak długo po początku zdarzenia utrzymujemy jego zmianę głośności: 0,5 s przy 48 kHz.
const CLASS_GAIN_HOLD_FRAMES: u64 = 24_000;
/// Narastanie zmiany w dół (5 ms) i powrót do pełnej głośności (300 ms), w ramkach.
const CLASS_GAIN_ATTACK_FRAMES: f32 = 240.0;
const CLASS_GAIN_RELEASE_FRAMES: f32 = 14_400.0;
/// Najwięcej zdarzeń czekających na obwiednię; starsze odrzucamy.
const MAX_CLASS_GAIN_EVENTS: usize = 64;

/// Dynamika "Scout" w mono: ciche dźwięki (poniżej −45 dBFS, ale nie szum poniżej −60 dBFS) są
/// podbijane, a głośne (powyżej −15 dBFS) ściszane. Nachylenie 0,5 dB na dB; najwięcej zależy od siły.
const SCOUT_QUIET_DB: f32 = -45.0;
const SCOUT_FLOOR_DB: f32 = -60.0;
const SCOUT_LOUD_DB: f32 = -15.0;
const SCOUT_SLOPE: f32 = 0.5;
const SCOUT_MAX_UP_DB: f32 = 12.0;
const SCOUT_MAX_DOWN_DB: f32 = 6.0;
/// Narastanie zmiany w dół (5 ms), wzmocnienia cichych i wygasanie obwiedni (150 ms), w ramkach.
const SCOUT_ATTACK_FRAMES: f32 = 240.0;
const SCOUT_RELEASE_FRAMES: f32 = 7_200.0;
const SCOUT_ENV_RELEASE_FRAMES: f32 = 7_200.0;

/// Bufor kołowy mono: ramki "ważonego mono", ich numery w osi wejścia i obwiednia zmian
/// głośności z klasyfikacji. Wszystko pod jednym zamkiem, żeby numer ramki i zdarzenie zawsze
/// odnosiły się do tej samej chwili.
#[derive(Debug)]
struct MonoRing {
    frames: VecDeque<(f32, f32)>,
    /// Numer ramki wejścia (od startu analizy), która stoi na przodzie `frames`.
    front_index: u64,
    /// Zdarzenia klas: (numer ramki początku, zmiana w dB), od najstarszego.
    class_gains: VecDeque<(u64, f32)>,
    /// Zdarzenie, które steruje głośnością teraz: (numer ramki początku, zmiana w dB).
    current_event: Option<(u64, f32)>,
    /// Bieżący mnożnik głośności (liniowo). Start na 1,0, czyli bez zmian.
    gain: f32,
    /// Czy zmiana głośności z klasyfikacji jest włączona (przełącznik w configu).
    class_gain_enabled: bool,
    /// Docelowe opóźnienie bufora w ramkach: zwykłe albo z włączoną zmianą klas.
    target_samples: usize,
    /// Wielkość zmiany głośności w dB (strzały −x, kroki +x), z suwaka w configu.
    class_gain_db: f32,
    /// Czy dynamika Scout jest włączona (przełącznik w configu).
    scout_enabled: bool,
    /// Siła Scout od 0 do 1 (suwak w configu).
    scout_amount: f32,
    /// Obwiednia szczytu do pomiaru poziomu Scout (liniowo).
    scout_env: f32,
    /// Bieżący mnożnik Scout (liniowo). Start na 1,0, czyli bez zmian.
    scout_gain: f32,
}

impl Default for MonoRing {
    fn default() -> Self {
        Self {
            frames: VecDeque::with_capacity(MONO_RING_CAPACITY),
            front_index: 0,
            class_gains: VecDeque::new(),
            current_event: None,
            gain: 1.0,
            class_gain_enabled: false,
            target_samples: TARGET_RING_LATENCY_SAMPLES,
            class_gain_db: MONO_CLASS_GAIN_DB_DEFAULT,
            scout_enabled: false,
            scout_amount: 0.5,
            scout_env: 0.0,
            scout_gain: 1.0,
        }
    }
}

impl MonoRing {
    /// Mnożnik głośności dla ramki o numerze `front_index`, z wygładzeniem narastania i powrotu.
    fn next_gain(&mut self) -> f32 {
        if !self.class_gain_enabled {
            self.gain = 1.0;
            return 1.0;
        }
        let frame = self.front_index;
        while let Some(&(onset, db)) = self.class_gains.front() {
            if onset > frame {
                break;
            }
            self.class_gains.pop_front();
            self.current_event = Some((onset, db));
        }
        // Zdarzenie zmienia głośność szybko (w górę i w dół), a powrót do pełnej głośności jest wolny.
        let (target, frames) = match self.current_event {
            Some((onset, db)) if frame < onset + CLASS_GAIN_HOLD_FRAMES => {
                (10f32.powf(db / 20.0), CLASS_GAIN_ATTACK_FRAMES)
            }
            _ => (1.0, CLASS_GAIN_RELEASE_FRAMES),
        };
        self.gain += (target - self.gain) * (1.0 - (-1.0 / frames).exp());
        self.gain
    }

    /// Dynamika Scout: podbija ciche i ścisza głośne. Oba kanały dostają ten sam mnożnik,
    /// więc balans L/P zostaje. Gdy Scout jest wyłączony, ramka przechodzi bez zmian.
    fn scout_frame(&mut self, l: f32, r: f32) -> (f32, f32) {
        if !self.scout_enabled {
            self.scout_gain = 1.0;
            return (l, r);
        }
        let peak = l.abs().max(r.abs());
        self.scout_env = peak.max(self.scout_env * (-1.0 / SCOUT_ENV_RELEASE_FRAMES).exp());
        let level_db = 20.0 * self.scout_env.max(1e-6).log10();
        let target = 10f32.powf(scout_target_db(level_db, self.scout_amount) / 20.0);
        // Ściszanie głośnych ma być szybkie, a podbijanie cichych łagodne, żeby nie pompowało.
        let frames = if target < self.scout_gain { SCOUT_ATTACK_FRAMES } else { SCOUT_RELEASE_FRAMES };
        self.scout_gain += (target - self.scout_gain) * (1.0 - (-1.0 / frames).exp());
        (l * self.scout_gain, r * self.scout_gain)
    }
}

/// Zmiana głośności Scout w dB dla poziomu `level_db` i siły `amount` (0–1).
fn scout_target_db(level_db: f32, amount: f32) -> f32 {
    let up = if level_db < SCOUT_QUIET_DB && level_db > SCOUT_FLOOR_DB {
        ((SCOUT_QUIET_DB - level_db) * SCOUT_SLOPE).min(SCOUT_MAX_UP_DB * amount)
    } else {
        0.0
    };
    let down = if level_db > SCOUT_LOUD_DB {
        -((level_db - SCOUT_LOUD_DB) * SCOUT_SLOPE).min(SCOUT_MAX_DOWN_DB * amount)
    } else {
        0.0
    };
    up + down
}

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
    mono_ring: Arc<Mutex<MonoRing>>,
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
    /// Nagrywanie sesji (`radar --recordSample` / `--recordVideo`). Wyłączone, dopóki sesja nie wystartuje.
    recorder: Arc<recorder::Recorder>,
    /// Bufor ostatniej sekundy próbek i onsety czekające na klasyfikację warstw
    /// (patrz [`layers::LayerTracker`]). Zasilany przez `update_analysis`, więc działa
    /// tak samo dla żywego audio i dla odtwarzania pliku (`sound::playback`).
    layers: Arc<Mutex<layers::LayerTracker>>,
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
            mono_ring: Arc::new(Mutex::new(MonoRing::default())),
            last_pan: Arc::new(Mutex::new((0.0, 0.0))),
            onset_ema: Arc::new(Mutex::new(Vec::new())),
            band_ema: Arc::new(Mutex::new(Vec::new())),
            sample_logger: Arc::new(sample_logger::SampleLogger::new()),
            recorder: Arc::new(recorder::Recorder::new()),
            layers: Arc::new(Mutex::new(layers::LayerTracker::default().with_external_onsets())),
        }
    }
}

impl SharedAudioAnalysis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn recorder(&self) -> &recorder::Recorder {
        &self.recorder
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
                if ring.frames.len() >= MONO_RING_CAPACITY {
                    ring.frames.pop_front();
                    ring.front_index += 1;
                }
                ring.frames.push_back(s);
            }
            if ring.frames.len() > ring.target_samples * RING_TRIM_FACTOR {
                let excess = ring.frames.len() - ring.target_samples;
                for _ in 0..excess {
                    ring.frames.pop_front();
                }
                ring.front_index += excess as u64;
            }
        }
    }

    /// Włącza albo wyłącza zmianę głośności z klasyfikacji w trybie mono. Włączona wymaga
    /// opóźnienia [`CLASS_GAIN_LATENCY_SAMPLES`], żeby zdążyć przed początkiem dźwięku. Wyłączona
    /// wraca do krótkiego opóźnienia i zerowej zmiany.
    pub(crate) fn set_mono_class_gain(&self, enabled: bool, amount_db: f32) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            ring.class_gain_db = amount_db.clamp(0.0, 12.0);
            if ring.class_gain_enabled == enabled {
                return;
            }
            ring.class_gain_enabled = enabled;
            ring.target_samples = if enabled { CLASS_GAIN_LATENCY_SAMPLES } else { TARGET_RING_LATENCY_SAMPLES };
            if !enabled {
                ring.class_gains.clear();
                ring.current_event = None;
                ring.gain = 1.0;
            }
        }
    }

    /// Włącza albo wyłącza dynamikę Scout w trybie mono i ustawia jej siłę (0–1).
    pub(crate) fn set_mono_scout(&self, enabled: bool, amount: f32) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            ring.scout_amount = amount.clamp(0.0, 1.0);
            if ring.scout_enabled != enabled {
                ring.scout_enabled = enabled;
                ring.scout_env = 0.0;
                ring.scout_gain = 1.0;
            }
        }
    }

    /// Zapisuje zmiany głośności z klasyfikacji dla trybu mono: strzały w dół, kroki w górę.
    /// Numer ramki liczymy z czasu początku zdarzenia i częstotliwości wejścia, tak jak bufor
    /// liczy swoje ramki, więc zmiana trafia w ten sam dźwięk.
    pub(crate) fn push_class_gains(&self, hits: &[layers::LayerHit], sample_rate: f32) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            if !ring.class_gain_enabled {
                return;
            }
            for hit in hits {
                let db = match hit.layer {
                    layers::Layer::Shots => -ring.class_gain_db,
                    layers::Layer::Steps => ring.class_gain_db,
                    layers::Layer::Rest => continue,
                };
                let frame = (hit.onset_s * f64::from(sample_rate)).round().max(0.0) as u64;
                ring.class_gains.push_back((frame, db));
            }
            while ring.class_gains.len() > MAX_CLASS_GAIN_EVENTS {
                ring.class_gains.pop_front();
            }
        }
    }

    /// Czyta dokładnie `count` próbek "ważonego mono" ze zmianą głośności z klasyfikacji. Wejście i wyjście to
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
                let sample = match ring.frames.pop_front() {
                    Some(s) => {
                        last = s;
                        let gain = ring.next_gain();
                        ring.front_index += 1;
                        ring.scout_frame(s.0 * gain, s.1 * gain)
                    }
                    None => {
                        last = (last.0 * HOLD_DECAY, last.1 * HOLD_DECAY);
                        let mult = ring.gain * ring.scout_gain;
                        (last.0 * mult, last.1 * mult)
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

/// Szuka i zwraca nazwę urządzenia "Stereo Mix" (WASAPI loopback) na Windowsie.
/// Jeśli nie znaleziono, zwraca None. To urządzenie pozwala nasłuchiwać dźwięku
/// bezpośrednio z gry zamiast z mikrofonu systemowego.
pub fn find_stereo_mix_device() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        let host = cpal::default_host();
        if let Ok(mut devices) = host.output_devices() {
            if let Some(device) = devices.find(|d| {
                d.name()
                    .map(|name| {
                        name.contains("Stereo Mix")
                            || name.contains("Mix - What U Hear")
                            || name.contains("Loopback")
                    })
                    .unwrap_or(false)
            }) {
                if let Ok(name) = device.name() {
                    return Some(name);
                }
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let host = cpal::default_host();
        if let Ok(mut devices) = host.input_devices() {
            if let Some(device) = devices.find(|d| {
                d.name()
                    .map(|name| {
                        name.contains("BlackHole") || name.contains("Loopback")
                    })
                    .unwrap_or(false)
            }) {
                if let Ok(name) = device.name() {
                    return Some(name);
                }
            }
        }
    }
    None
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

    let stream = match default_config.sample_format() {
        SampleFormat::F32 => {
            let a = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[f32], info| {
                    update_analysis(data, channels, sample_rate, &a, Some(info.timestamp().capture));
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let a = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[i16], info| {
                    let f: Vec<f32> = data.iter().map(|&s| direction::i16_to_f32(s)).collect();
                    update_analysis(&f, channels, sample_rate, &a, Some(info.timestamp().capture));
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let a = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[u16], info| {
                    let f: Vec<f32> = data.iter().map(|&s| direction::u16_to_f32(s)).collect();
                    update_analysis(&f, channels, sample_rate, &a, Some(info.timestamp().capture));
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

pub(crate) fn update_analysis(
    samples: &[f32],
    channels: usize,
    sample_rate: f32,
    analysis: &SharedAudioAnalysis,
    captured: Option<cpal::StreamInstant>,
) {
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

    // Stan KAŻDEGO kanału z kierunkiem. Wskaźniki kierunku nie biorą onsetów stąd (robi to
    // tracker warstw, patrz niżej), ale logger próbek (niżej) widzi wszystkie kanały, żeby dało się
    // złapać dźwięk, który był słyszalny, a nie dał zdarzenia (patrz `direction::ChannelActivity`).
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

    // Warstwy 3+3 i wskaźniki kierunku dostają zdarzenia z szybkiego detektora, ale tylko te,
    // które przechodzą regułę poziomu z Pythona. Odrzuca to szum przy starcie i ogon dźwięku.
    let (layer_hits, accepted) = {
        let mut tracker = analysis.layers.lock().unwrap_or_else(|e| e.into_inner());
        tracker.push_frames(samples, channels, sample_rate);
        let accepted = tracker.onset_allowed() && channel_activity.iter().any(|a| a.is_onset);
        if accepted && tracker.has_external_onsets() {
            tracker.add_onset();
        }
        (tracker.take_ready(), accepted)
    };

    // Zmiana głośności w trybie mono: strzały ciszej, kroki głośniej (patrz `push_class_gains`).
    analysis.push_class_gains(&layer_hits, sample_rate);

    // Wskaźniki kierunku i kroki idą z tego samego, szybkiego detektora kanałów.
    let direction_events: Vec<DirectionEvent> = channel_activity
        .iter()
        .filter(|a| accepted && a.is_onset)
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

    if let Ok(mut lock) = analysis.lock() {
        lock.channel_rms = stats.rms;
        lock.channel_count = channels;
        lock.channel_rule_energies = rule_energies;
        lock.push_direction_events(&direction_events, Instant::now());
    }

    // Trafienia warstw trafiają do słupków 3+3 (patrz `ui::radar_egui`).
    if !layer_hits.is_empty() {
        if let Ok(mut lock) = analysis.lock() {
            lock.push_layer_hits(&layer_hits, Instant::now());
        }
    }

    // Miks do "ważonego mono" na sam koniec — dopiero tu, dla ewentualnego
    // odtwarzania przez `mono_router`. Radar powyżej już dostał pełne dane
    // per-kanał, więc to nie zastępuje ich, tylko dokłada osobny strumień do
    // wyjścia.
    // Nagrywanie: kopia dwóch pierwszych kanałów. Bez trwającej sesji kończy od razu.
    analysis.recorder.push(samples, channels, captured);
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

    fn class_hit(layer: layers::Layer, onset_s: f64) -> layers::LayerHit {
        layers::LayerHit {
            layer,
            class: Some(layers::Class::A),
            pan: 0.0,
            peak_dbfs: -10.0,
            brightness_db: None,
            gap_s: None,
            onset_s,
        }
    }

    #[test]
    fn shot_lowers_and_step_raises_the_mono_output_after_the_event() {
        // Strzał: 0,5 ciszej po zdarzeniu (ok. −6 dB), a przed nim bez zmian.
        let quiet = SharedAudioAnalysis::new();
        quiet.set_mono_class_gain(true, MONO_CLASS_GAIN_DB_DEFAULT);
        quiet.push_stereo_pan(&vec![(0.5, 0.5); 4_000]);
        quiet.push_class_gains(&[class_hit(layers::Layer::Shots, 0.01)], 48_000.0); // ramka 480
        let out = quiet.read_stereo_pan(3_000);
        assert_eq!(out[100], (0.5, 0.5), "przed zdarzeniem bez zmian");
        assert!((out[2_000].0 - 0.25).abs() < 0.02, "strzał ok. −6 dB: {}", out[2_000].0);

        // Krok: głośniej po zdarzeniu (ok. +6 dB).
        let loud = SharedAudioAnalysis::new();
        loud.set_mono_class_gain(true, MONO_CLASS_GAIN_DB_DEFAULT);
        loud.push_stereo_pan(&vec![(0.3, 0.3); 4_000]);
        loud.push_class_gains(&[class_hit(layers::Layer::Steps, 0.01)], 48_000.0);
        let out = loud.read_stereo_pan(3_000);
        assert!(out[2_000].0 > 0.55, "krok ok. +6 dB: {}", out[2_000].0);
    }

    #[test]
    fn scout_boosts_quiet_and_tames_loud_sounds() {
        let mut quiet = MonoRing { scout_enabled: true, scout_amount: 1.0, ..MonoRing::default() };
        let mut last = (0.0, 0.0);
        for _ in 0..20_000 {
            last = quiet.scout_frame(0.0032, 0.0032); // ok. −50 dBFS
        }
        assert!(last.0 > 0.0038, "ciche podbite, ma {}", last.0);

        let mut loud = MonoRing { scout_enabled: true, scout_amount: 1.0, ..MonoRing::default() };
        let mut last = (0.0, 0.0);
        for _ in 0..1_000 {
            last = loud.scout_frame(0.9, 0.9); // ok. −1 dBFS
        }
        assert!(last.0 < 0.6, "głośne ściszone, ma {}", last.0);
    }

    #[test]
    fn scout_keeps_silence_silent_and_is_a_no_op_when_off() {
        let mut ring = MonoRing { scout_enabled: true, scout_amount: 1.0, ..MonoRing::default() };
        for _ in 0..5_000 {
            assert_eq!(ring.scout_frame(0.0, 0.0), (0.0, 0.0), "cisza zostaje ciszą");
        }
        let mut off = MonoRing::default();
        assert_eq!(off.scout_frame(0.3, -0.2), (0.3, -0.2), "wyłączony Scout nie zmienia sygnału");
    }

    #[test]
    fn class_gain_amount_follows_the_slider() {
        // Suwak 12 dB: strzał ok. 0,5 × 10^(−12/20) ≈ 0,125.
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_class_gain(true, 12.0);
        analysis.push_stereo_pan(&vec![(0.5, 0.5); 4_000]);
        analysis.push_class_gains(&[class_hit(layers::Layer::Shots, 0.01)], 48_000.0);
        let out = analysis.read_stereo_pan(3_000);
        assert!((out[2_000].0 - 0.126).abs() < 0.02, "strzał przy 12 dB: {}", out[2_000].0);
    }

    #[test]
    fn class_gain_is_off_by_default_and_keeps_the_short_latency() {
        let analysis = SharedAudioAnalysis::new();
        analysis.push_stereo_pan(&vec![(0.5, 0.5); 4_000]);
        analysis.push_class_gains(&[class_hit(layers::Layer::Shots, 0.01)], 48_000.0);
        let out = analysis.read_stereo_pan(3_000);
        assert_eq!(out[2_000], (0.5, 0.5), "domyślnie strzał nie jest ciszej");
    }

    #[test]
    fn class_gain_returns_to_full_level_after_the_hold() {
        let mut ring = MonoRing::default();
        ring.class_gain_enabled = true;
        ring.class_gains.push_back((0, -MONO_CLASS_GAIN_DB_DEFAULT));
        let mut last = 1.0;
        for _ in 0..60_000 {
            last = ring.next_gain();
            ring.front_index += 1;
        }
        assert!(last > 0.95, "po wygaśnięciu zdarzenia głośność wraca do pełnej: {last}");
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

        let ring_len = analysis.mono_ring.lock().unwrap().frames.len();
        assert!(
            ring_len <= TARGET_RING_LATENCY_SAMPLES * RING_TRIM_FACTOR,
            "bufor nie powinien urosnąć powyżej progu przycinania, ma {ring_len}"
        );
        assert!(
            ring_len < MONO_RING_CAPACITY,
            "bufor utknął przy pełnej pojemności (~340ms) zamiast się przyciąć, ma {ring_len}"
        );
    }
}

#[cfg(test)]
mod layer_bar_config_tests {
    use super::*;

    /// Stary plik konfiguracji bez nowych pól dostaje domyślne promienie łuków 3+3.
    #[test]
    fn old_config_without_layer_arcs_gets_defaults() {
        let mut value = serde_json::to_value(RadarConfig::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("layer_arc_inner");
        object.remove("layer_arc_spacing");
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.layer_arc_inner, 0.35);
        assert_eq!(cfg.layer_arc_spacing, 0.15);
    }
}

#[cfg(test)]
mod direction_queue_tests {
    use super::*;

    fn event(angle_deg: f32) -> DirectionEvent {
        DirectionEvent { angle_deg, intensity: 0.5, color_rgb: (0, 0, 0) }
    }

    #[test]
    fn events_between_two_ui_frames_are_all_kept() {
        let mut analysis = AudioAnalysis::default();
        let t = Instant::now();
        // Dwa callbacki przed jedną klatką UI (klatka ~16 ms, callback ~10 ms).
        analysis.push_direction_events(&[event(-30.0)], t);
        analysis.push_direction_events(&[event(10.0), event(20.0)], t);
        let drained = analysis.drain_direction_events();
        assert_eq!(drained.len(), 3, "żadne zdarzenie nie znika między klatkami");
        assert_eq!(drained[0].event.angle_deg, -30.0, "kolejność wykrycia");
        assert!(analysis.drain_direction_events().is_empty(), "po odczycie kolejka jest pusta");
    }

    #[test]
    fn queue_drops_the_oldest_events_when_ui_stalls() {
        let mut analysis = AudioAnalysis::default();
        let t = Instant::now();
        for i in 0..(MAX_QUEUED_DIRECTION_EVENTS + 10) {
            analysis.push_direction_events(&[event(i as f32)], t);
        }
        let drained = analysis.drain_direction_events();
        assert_eq!(drained.len(), MAX_QUEUED_DIRECTION_EVENTS);
        assert_eq!(
            drained.last().map(|q| q.event.angle_deg),
            Some((MAX_QUEUED_DIRECTION_EVENTS + 9) as f32),
            "najnowsze zdarzenia zostają"
        );
    }
}
