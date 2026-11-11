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
use crate::sound::ring_clock;
use crate::sound::sample_logger;
use crate::sound::spectrum::{self, ColorRule};
use std::collections::BTreeMap;

/// Tryb wyświetlania wizualizacji audio na overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum VisualizationMode {
    /// Tylko radar ze strzałkami kierunków.
    OnlyRadar,
    /// Tylko słupki stereo L/R.
    OnlyBars,
    /// Zarówno radar jak i słupki stereo.
    Both,
    /// Sześć słupków 3+3: po trzy na stronę (strzały, kroki, reszta), patrz `sound::layers`.
    LayerBars,
    /// Cztery słupki 2+2: po dwa na stronę. Wewnątrz strzały i kroki, na zewnątrz reszta.
    LayerBars22,
    /// Nic nie wyświetlaj (overlay wyłączony).
    None,
}

impl VisualizationMode {
    pub const ALL: &'static [Self] = &[
        Self::OnlyRadar,
        Self::OnlyBars,
        Self::Both,
        Self::LayerBars,
        Self::LayerBars22,
        Self::None,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::OnlyRadar => "◉ Tylko radar",
            Self::OnlyBars => "≡ Tylko słupki L/R",
            Self::Both => "◉≡ Radar + słupki",
            Self::LayerBars => "▮▮▮ Warstwy 3+3",
            Self::LayerBars22 => "▮▮ Warstwy 2+2",
            Self::None => "⊘ Nic",
        }
    }

    /// Opis trybu, pokazywany w rozwijanej sekcji przy wyborze trybu.
    pub fn description(self) -> &'static str {
        match self {
            Self::OnlyRadar => {
                "Tarcza z kropkami zdarzeń i strzałkami kierunku. Odległość kropki od środka to \
                 głośność, a kolor zależy od reguł częstotliwości. Słupki L/R i warstwy są ukryte."
            }
            Self::OnlyBars => {
                "Łuki lewej i prawej strony na obwodzie. Długość i jasność rosną z energii kanału. \
                 Bez tarczy i kropek."
            }
            Self::Both => {
                "Tarcza z kropkami i strzałkami oraz łuki L/R na obwodzie naraz. Pokazuje najwięcej, \
                 więc jest najbardziej zajęty."
            }
            Self::LayerBars => {
                "Sześć łuków, po trzy na stronę. Kroki najbliżej środka, potem strzały i wybuchy, \
                 a na zewnątrz wszystko, czyli ciągłe tło stereo rysowane jak słupki L/R."
            }
            Self::LayerBars22 => {
                "Cztery łuki, po dwa na stronę. Wewnątrz strzały, wybuchy i kroki (silniejsza z tych \
                 dwóch warstw), na zewnątrz wszystko naraz. Cała czerwień (obrysy i poświata na \
                 zewnątrz radaru) jest po jednej stronie: tej, z której idą strzały albo kroki, a bez \
                 nich tej, z której jest głośniej."
            }
            Self::None => "Overlay wyłączony. Radar nie jest rysowany, a ustawienia wyglądu są ukryte.",
        }
    }

    /// Czy tryb rysuje tarczę z kropkami i strzałkami kierunku.
    pub fn shows_radar(self) -> bool {
        matches!(self, Self::OnlyRadar | Self::Both)
    }

    /// Czy tryb rysuje łuki L/R na obwodzie.
    pub fn shows_bars(self) -> bool {
        matches!(self, Self::OnlyBars | Self::Both)
    }

    /// Czy tryb rysuje warstwy (3+3 albo 2+2).
    pub fn shows_layers(self) -> bool {
        matches!(self, Self::LayerBars | Self::LayerBars22)
    }

    /// Czy overlay w ogóle coś rysuje.
    pub fn shows_overlay(self) -> bool {
        self != Self::None
    }
}

impl Default for VisualizationMode {
    fn default() -> Self {
        Self::Both
    }
}

/// Układ głośników wybrany na górze ustawień. Od niego zależy, które sekcje ustawień są widoczne:
/// stereo pokazuje słupki L/R i warstwy, 5.1/7.1 pokazuje tylko radar z kropkami głośników.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SpeakerSetup {
    /// Stereo (2.0).
    #[default]
    Stereo,
    /// 5.1 albo 7.1, czyli tylko radar.
    Surround,
}

impl SpeakerSetup {
    pub const ALL: &'static [Self] = &[Self::Stereo, Self::Surround];

    pub fn label(self) -> &'static str {
        match self {
            Self::Stereo => "Stereo (2.0)",
            Self::Surround => "5.1 / 7.1",
        }
    }

    /// Czy tryb wyświetlania jest dostępny przy tym układzie. Przy 5.1/7.1 tylko radar i wyłączenie:
    /// słupki L/R i warstwy są stereo, więc ich tu nie ma.
    pub fn allows(self, mode: VisualizationMode) -> bool {
        match self {
            Self::Stereo => true,
            Self::Surround => matches!(mode, VisualizationMode::OnlyRadar | VisualizationMode::None),
        }
    }
}

/// Wygląd poświaty strony z przewagą na radarze warstw (3+3, 2+2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RadarGlowStyle {
    /// Kilka pasów za zewnętrznym łukiem, coraz bledszych i krótszych. Długość idzie za wysokością łuku.
    Crescent,
    /// Cienka linia na całej długości łuku z bladą otoczką. Przewagę pokazuje krycie.
    #[default]
    Line,
}

impl RadarGlowStyle {
    pub const ALL: &'static [Self] = &[Self::Line, Self::Crescent];

    pub fn label(self) -> &'static str {
        match self {
            Self::Crescent => "Półksiężyc",
            Self::Line => "Linia",
        }
    }
}

/// Wygląd poświaty na krawędzi ekranu w stereo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EdgeGlowStyle {
    /// Plama na środku wysokości krawędzi.
    Blob,
    /// Winieta na całej wysokości boku ekranu, gasnąca w głąb ekranu.
    #[default]
    Vignette,
}

impl EdgeGlowStyle {
    pub const ALL: &'static [Self] = &[Self::Vignette, Self::Blob];

    pub fn label(self) -> &'static str {
        match self {
            Self::Blob => "Plama",
            Self::Vignette => "Winieta",
        }
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
    /// Bufor mono jest w skali stereo (patrz [`MONO_PAN_SCALE`]), więc 0 dB daje w słuchawce mniej
    /// więcej poziom gry, tak samo przy strumieniu 2- i 8-kanałowym.
    #[serde(default)]
    pub mono_boost_db: f32,
    /// Wersja skali poziomu w torze mono, potrzebna do przeliczenia `mono_boost_db` ze starego pliku
    /// (patrz [`RadarConfig::migrate_mono_level`]). Plik bez tego pola to wersja 0: bufor mono był
    /// wtedy dzielony przez liczbę kanałów strumienia, więc przy VB-Cable 7.1 był o 12 dB cichszy.
    #[serde(default)]
    pub mono_level_version: u32,
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
    /// tarczy (patrz `HearRadarApp::ui_overlay_contents`). Na Windows
    /// (`ui::overlay_win`) zamiast tego poświata dostaje osobne, wąskie okna
    /// przy krawędziach, bo pełnoekranowe okno zabierało grze klatki.
    #[serde(default)]
    pub edge_glow_enabled: bool,
    /// Moc poświaty na krawędzi ekranu w stereo (0.0 wyłącza, 1.0 domyślna, 2.0 najmocniejsza). W stereo
    /// poświata stoi na lewej albo prawej krawędzi, po stronie przewagi łuku wszystko. Wspólna dla
    /// wszystkich trybów, jak samo `edge_glow_enabled`.
    #[serde(default = "default_edge_glow_strength")]
    pub edge_glow_strength: f32,
    /// Wygląd poświaty na krawędzi ekranu w stereo, patrz [`EdgeGlowStyle`]. Wspólny dla trybów.
    #[serde(default)]
    pub edge_glow_style: EdgeGlowStyle,
    /// Zasięg winiety w głąb ekranu jako ułamek szerokości ekranu (najszerszy pas), 0.03 .. 0.5.
    #[serde(default = "default_edge_vignette_width")]
    pub edge_vignette_width: f32,
    /// Wygląd poświaty radaru warstw, patrz [`RadarGlowStyle`]. Wspólny dla 3+3 i 2+2.
    #[serde(default)]
    pub radar_glow_style: RadarGlowStyle,
    /// Grubość linii poświaty radaru (styl `Line`) jako mnożnik: 1.0 to 0,4 grubości zewnętrznego łuku.
    #[serde(default = "default_glow_line_width")]
    pub glow_line_width: f32,
    /// Widoczność poświat (radar i krawędź) przy najmniejszej przewadze, 0..1: ułamek pełnego krycia
    /// tuż nad progiem przewagi. 0 = poświata rośnie od zera, 1 = od razu pełna. Moc mnoży to dalej.
    #[serde(default = "default_glow_min_visibility")]
    pub glow_min_visibility: f32,
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
    /// Głębokość oddechu w stylach kierunkowych z oddechem (`MonoCueStyle::uses_breath`),
    /// 0–0,95. Domyślne 0,44 to dołek −5 dB dla dźwięku w pełni z przeciwnej strony.
    #[serde(default = "default_mono_breath_depth")]
    pub mono_breath_depth: f32,
    /// Siła cieńszej barwy w stylach kierunkowych z oddechem, 0–1: 0 = sam oddech, 1 = kształt
    /// pasma jak w zwykłym Oddechu (z wyrównaną głośnością).
    #[serde(default = "default_mono_breath_timbre")]
    pub mono_breath_timbre: f32,
    /// Siła lekkiego brzęczenia w stylu „Oddech + brzęczenie” (`MonoCueStyle::BreathBuzz`), 0–1.
    /// 0,5 to wahanie 6 dB przy 45 Hz, 1,0 schodzi na dnie cyklu do ciszy.
    #[serde(default = "default_mono_buzz_depth")]
    pub mono_buzz_depth: f32,
    /// Styl wskazówki dla strzałów, kroków i reszty w stylu „Wg warstw” (`MonoCueStyle::PerLayer`).
    #[serde(default)]
    pub mono_layer_styles: MonoLayerStyles,
    /// Strona przeciwna w stylach kierunkowych (patrz `mono_router::OppositeSettings`).
    /// Wskazówki tylko „w dół”: efekt nigdzie nie podnosi dźwięku z przeciwnej strony ponad
    /// oryginał (oddech od 0 dB w dół, grzebień z samymi dołkami). Wyłączone = wyrównana głośność.
    #[serde(default = "default_mono_cue_subtractive")]
    pub mono_cue_subtractive: bool,
    /// Głębokość dołków grzebienia w trybie „w dół” (dB, −24…−6).
    #[serde(default = "default_mono_comb_notch_db")]
    pub mono_comb_notch_db: f32,
    /// Poziom oznaczonej strony przeciwnej (dB, −12…+6). W „Wg warstw” dodaje się do głośności warstwy.
    #[serde(default = "default_mono_opposite_level_db")]
    pub mono_opposite_level_db: f32,
    /// O ile dB strona przeciwna ustępuje w paśmie, w którym gra strona własna (0–12, 0 = wyłączone).
    #[serde(default = "default_mono_opposite_duck_db")]
    pub mono_opposite_duck_db: f32,
    /// Cień głowy dla strony przeciwnej (0–0,7, 0 = wyłączony): dźwięki bardziej z boku ciszej.
    #[serde(default)]
    pub mono_opposite_shadow: f32,
    /// Siła wskazówki po początku dźwięku w stylach bez wolnej modulacji (grzebień, jasny, piskliwy,
    /// brzęczenie), 0–1. 1 = wskazówka cały czas tak samo mocna.
    #[serde(default = "default_mono_cue_tail")]
    pub mono_cue_tail: f32,
    /// Ściszenie ogona po wycofaniu wskazówki (dB, −6…0).
    #[serde(default = "default_mono_cue_tail_level_db")]
    pub mono_cue_tail_level_db: f32,
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
    /// Układ głośników (stereo albo 5.1/7.1) wybrany na górze ustawień, patrz [`SpeakerSetup`].
    #[serde(default)]
    pub speaker_setup: SpeakerSetup,
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
    /// Wspólny kolor każdej pary 3+3 (RGB), z czasów przed osobnymi kolorami stron. Radar rysuje
    /// teraz kolorami stron (`layer_*_left_color`/`layer_*_right_color`). Te pola zostają, bo plik
    /// sprzed kolorów stron daje z nich oba boki (patrz [`RadarConfig::migrate_layer_side_colors`]).
    #[serde(default = "default_layer_shots_color")]
    pub layer_shots_color: [u8; 3],
    #[serde(default = "default_layer_steps_color")]
    pub layer_steps_color: [u8; 3],
    #[serde(default = "default_layer_rest_color")]
    pub layer_rest_color: [u8; 3],
    /// Kolory łuków warstw osobno po lewej i prawej stronie (tryby 3+3 i 2+2). Brak w pliku:
    /// kolor warstwy z pola wyżej, bo wczytanie przelicza stary plik (patrz
    /// [`RadarConfig::migrate_layer_side_colors`]). Dostęp po warstwie: [`RadarConfig::layer_side_colors`].
    #[serde(default = "default_layer_shots_color")]
    pub layer_shots_left_color: [u8; 3],
    #[serde(default = "default_layer_shots_color")]
    pub layer_shots_right_color: [u8; 3],
    #[serde(default = "default_layer_steps_color")]
    pub layer_steps_left_color: [u8; 3],
    #[serde(default = "default_layer_steps_color")]
    pub layer_steps_right_color: [u8; 3],
    #[serde(default = "default_layer_rest_color")]
    pub layer_rest_left_color: [u8; 3],
    #[serde(default = "default_layer_rest_color")]
    pub layer_rest_right_color: [u8; 3],
    /// Wersja kolorów warstw, potrzebna do przeliczenia starego pliku. Plik bez tego pola to wersja 0:
    /// jeden kolor warstwy dla obu stron, bez pól `layer_*_left_color`/`layer_*_right_color`.
    #[serde(default)]
    pub layer_colors_version: u32,
    /// Moc czerwonej poświaty na zewnątrz radaru 2+2, po stronie z przewagą (0.0 wyłącza, 1.0 domyślna,
    /// 2.0 najmocniejsza). Mnoży krycie poświaty, a nie zmienia decyzji o stronie.
    #[serde(default = "default_layer_side_glow")]
    pub layer_side_glow: f32,
    /// Wygląd zapisany osobno dla każdego trybu wyświetlania (patrz [`ModeAppearance`]). Pola
    /// powyżej to wygląd wybranego trybu. Tu są kopie dla wszystkich trybów, w tym bieżącego po
    /// zapisie. `#[serde(default)]`: stary plik bez tej mapy startuje pusty.
    #[serde(default)]
    pub mode_appearance: BTreeMap<VisualizationMode, ModeAppearance>,
}

/// Wygląd, który każdy tryb wyświetlania ma osobno: rozmiar, przezroczystości, czułość, grubość
/// linii, kolory, wzmocnienia i opcje widoków. Pozycja na ekranie, poświata krawędzi, klikanie przez
/// overlay i ustawienia mono zostają wspólne dla wszystkich trybów.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ModeAppearance {
    pub radius: f32,
    pub bg_alpha: f32,
    pub fg_alpha: f32,
    pub high_contrast: bool,
    pub decay_secs: f32,
    pub radar_sensitivity: f32,
    pub line_thickness: f32,
    pub speaker_layout_view: bool,
    pub spatial_audio_view: bool,
    pub direction_contrast_enabled: bool,
    pub stereo_left_color: [u8; 3],
    pub stereo_right_color: [u8; 3],
    pub layer_arc_inner: f32,
    pub layer_arc_spacing: f32,
    pub layer_steps_gain: f32,
    pub layer_shots_gain: f32,
    pub layer_rest_gain: f32,
    pub layer_shots_color: [u8; 3],
    pub layer_steps_color: [u8; 3],
    pub layer_rest_color: [u8; 3],
    pub layer_shots_left_color: [u8; 3],
    pub layer_shots_right_color: [u8; 3],
    pub layer_steps_left_color: [u8; 3],
    pub layer_steps_right_color: [u8; 3],
    pub layer_rest_left_color: [u8; 3],
    pub layer_rest_right_color: [u8; 3],
    pub layer_side_glow: f32,
}

impl ModeAppearance {
    /// Oba boki każdej warstwy w jej wspólnym kolorze (`layer_*_color`), czyli wygląd sprzed kolorów stron.
    fn side_colors_from_layer_colors(&mut self) {
        self.layer_shots_left_color = self.layer_shots_color;
        self.layer_shots_right_color = self.layer_shots_color;
        self.layer_steps_left_color = self.layer_steps_color;
        self.layer_steps_right_color = self.layer_steps_color;
        self.layer_rest_left_color = self.layer_rest_color;
        self.layer_rest_right_color = self.layer_rest_color;
    }
}

impl Default for ModeAppearance {
    fn default() -> Self {
        RadarConfig::default().appearance()
    }
}

impl RadarConfig {
    /// Wygląd z bieżących pól, czyli taki, jaki widzi aktualnie wybrany tryb.
    pub fn appearance(&self) -> ModeAppearance {
        ModeAppearance {
            radius: self.radius,
            bg_alpha: self.bg_alpha,
            fg_alpha: self.fg_alpha,
            high_contrast: self.high_contrast,
            decay_secs: self.decay_secs,
            radar_sensitivity: self.radar_sensitivity,
            line_thickness: self.line_thickness,
            speaker_layout_view: self.speaker_layout_view,
            spatial_audio_view: self.spatial_audio_view,
            direction_contrast_enabled: self.direction_contrast_enabled,
            stereo_left_color: self.stereo_left_color,
            stereo_right_color: self.stereo_right_color,
            layer_arc_inner: self.layer_arc_inner,
            layer_arc_spacing: self.layer_arc_spacing,
            layer_steps_gain: self.layer_steps_gain,
            layer_shots_gain: self.layer_shots_gain,
            layer_rest_gain: self.layer_rest_gain,
            layer_shots_color: self.layer_shots_color,
            layer_steps_color: self.layer_steps_color,
            layer_rest_color: self.layer_rest_color,
            layer_shots_left_color: self.layer_shots_left_color,
            layer_shots_right_color: self.layer_shots_right_color,
            layer_steps_left_color: self.layer_steps_left_color,
            layer_steps_right_color: self.layer_steps_right_color,
            layer_rest_left_color: self.layer_rest_left_color,
            layer_rest_right_color: self.layer_rest_right_color,
            layer_side_glow: self.layer_side_glow,
        }
    }

    /// Wgrywa zapisany wygląd do bieżących pól.
    pub fn apply_appearance(&mut self, a: &ModeAppearance) {
        self.radius = a.radius;
        self.bg_alpha = a.bg_alpha;
        self.fg_alpha = a.fg_alpha;
        self.high_contrast = a.high_contrast;
        self.decay_secs = a.decay_secs;
        self.radar_sensitivity = a.radar_sensitivity;
        self.line_thickness = a.line_thickness;
        self.speaker_layout_view = a.speaker_layout_view;
        self.spatial_audio_view = a.spatial_audio_view;
        self.direction_contrast_enabled = a.direction_contrast_enabled;
        self.stereo_left_color = a.stereo_left_color;
        self.stereo_right_color = a.stereo_right_color;
        self.layer_arc_inner = a.layer_arc_inner;
        self.layer_arc_spacing = a.layer_arc_spacing;
        self.layer_steps_gain = a.layer_steps_gain;
        self.layer_shots_gain = a.layer_shots_gain;
        self.layer_rest_gain = a.layer_rest_gain;
        self.layer_shots_color = a.layer_shots_color;
        self.layer_steps_color = a.layer_steps_color;
        self.layer_rest_color = a.layer_rest_color;
        self.layer_shots_left_color = a.layer_shots_left_color;
        self.layer_shots_right_color = a.layer_shots_right_color;
        self.layer_steps_left_color = a.layer_steps_left_color;
        self.layer_steps_right_color = a.layer_steps_right_color;
        self.layer_rest_left_color = a.layer_rest_left_color;
        self.layer_rest_right_color = a.layer_rest_right_color;
        self.layer_side_glow = a.layer_side_glow;
    }

    /// Kolory łuku warstwy `layer` po lewej i prawej stronie (indeks 0 lewa, 1 prawa, jak strony w
    /// `ui::radar_egui`).
    pub fn layer_side_colors(&self, layer: layers::Layer) -> [[u8; 3]; 2] {
        match layer {
            layers::Layer::Shots => [self.layer_shots_left_color, self.layer_shots_right_color],
            layers::Layer::Steps => [self.layer_steps_left_color, self.layer_steps_right_color],
            layers::Layer::Rest => [self.layer_rest_left_color, self.layer_rest_right_color],
        }
    }

    /// Jak [`Self::layer_side_colors`], do zmiany kolorów (ustawienia w UI).
    pub fn layer_side_colors_mut(&mut self, layer: layers::Layer) -> [&mut [u8; 3]; 2] {
        match layer {
            layers::Layer::Shots => [&mut self.layer_shots_left_color, &mut self.layer_shots_right_color],
            layers::Layer::Steps => [&mut self.layer_steps_left_color, &mut self.layer_steps_right_color],
            layers::Layer::Rest => [&mut self.layer_rest_left_color, &mut self.layer_rest_right_color],
        }
    }

    /// Ustawia jeden kolor wszystkim warstwom po stronie `side` (0 lewa, 1 prawa).
    pub fn set_layer_side_color_for_all(&mut self, side: usize, color: [u8; 3]) {
        for layer in layers::Layer::ALL {
            *self.layer_side_colors_mut(layer)[side] = color;
        }
    }

    /// Przelicza plik zapisany przed osobnymi kolorami stron (wersja 0, patrz [`LAYER_COLORS_VERSION`]):
    /// obie strony każdej warstwy dostają jej dotychczasowy wspólny kolor, w bieżących polach i w
    /// wyglądzie każdego trybu z `mode_appearance`. Dzięki temu stary config wygląda tak samo. Plik już
    /// przeliczony albo nowy zostaje bez zmian. Wołane przy wczytaniu pliku (`config_store::load_from`).
    pub fn migrate_layer_side_colors(&mut self) {
        if self.layer_colors_version >= LAYER_COLORS_VERSION {
            return;
        }
        let mut current = self.appearance();
        current.side_colors_from_layer_colors();
        self.apply_appearance(&current);
        for appearance in self.mode_appearance.values_mut() {
            appearance.side_colors_from_layer_colors();
        }
        self.layer_colors_version = LAYER_COLORS_VERSION;
    }

    /// Zapamiętuje wygląd wybranego trybu w mapie. Wołane przed każdym zapisem na dysk.
    pub fn remember_appearance(&mut self) {
        let appearance = self.appearance();
        self.mode_appearance.insert(self.visualization_mode, appearance);
    }

    /// Przelicza ustawienia mono z pliku zapisanego przed zmianą skali bufora mono (wersja 0, patrz
    /// [`MONO_LEVEL_VERSION`]). Bufor jest teraz przy VB-Cable 7.1 o [`LEGACY_MONO_BOOST_OFFSET_DB`]
    /// głośniejszy, więc o tyle samo obniżamy `mono_boost_db` i głośność w słuchawce zostaje taka sama.
    /// Plik już przeliczony albo nowy zostaje bez zmian, więc kolejne wczytania niczego nie odejmują.
    /// Wołane przy wczytaniu pliku (`config_store::load_from`). Na dysk trafia przy następnym zapisie,
    /// a do tego czasu każde wczytanie liczy od tej samej, starej wartości.
    pub fn migrate_mono_level(&mut self) {
        if self.mono_level_version >= MONO_LEVEL_VERSION {
            return;
        }
        self.mono_boost_db =
            (self.mono_boost_db - LEGACY_MONO_BOOST_OFFSET_DB).clamp(MONO_BOOST_DB_MIN, MONO_BOOST_DB_MAX);
        self.mono_level_version = MONO_LEVEL_VERSION;
    }

    /// Przełącza tryb wyświetlania. Wygląd dotychczasowego trybu zostaje zapamiętany, a nowy wczytuje
    /// swój zapisany wygląd. Tryb, którego jeszcze nie było, zachowuje wygląd bieżący.
    pub fn switch_mode(&mut self, mode: VisualizationMode) {
        if mode == self.visualization_mode {
            return;
        }
        self.remember_appearance();
        self.visualization_mode = mode;
        if let Some(saved) = self.mode_appearance.get(&mode).copied() {
            self.apply_appearance(&saved);
        }
    }
}

/// Bieżąca wersja kolorów warstw: osobny kolor lewej i prawej strony każdej warstwy. Wersja 0 miała
/// jeden kolor warstwy dla obu stron.
pub(crate) const LAYER_COLORS_VERSION: u32 = 1;

/// Bieżąca wersja skali poziomu w torze mono: bufor w skali stereo (patrz [`MONO_PAN_SCALE`]).
/// Wersja 0 dzieliła bufor przez liczbę kanałów strumienia.
pub(crate) const MONO_LEVEL_VERSION: u32 = 1;

/// O ile obniżyć `mono_boost_db` z pliku w wersji 0, żeby głośność w słuchawce się nie zmieniła.
/// Wersja 0 przy VB-Cable 7.1 (8 kanałów, wymóg z README; gra w stereo gra tylko na FL/FR) dawała
/// w buforze FL/8, a skala stereo daje FL/2, czyli 20·log10(4) ≈ 12,04 dB więcej. Liczymy względem
/// 8 kanałów, bo tak przechwytuje obecny użytkownik. Kto przechwytywał strumień 2-kanałowy, po migracji
/// usłyszy o tyle ciszej, ale to dokładnie ta różnica między 2 a 8 kanałami, którą poprawka usuwa.
const LEGACY_MONO_BOOST_OFFSET_DB: f32 = 12.041_2;

/// Zakres suwaka wzmocnienia mono (dB). Dół −24 dB, żeby zmieściło się najniższe dawne ustawienie
/// po migracji (−12 dB − 12,04 dB). Suwak egui przycina wartości spoza zakresu, więc bez tego
/// przeliczona wartość zmieniłaby się sama po otwarciu ustawień.
pub const MONO_BOOST_DB_MIN: f32 = -24.0;
pub const MONO_BOOST_DB_MAX: f32 = 24.0;

fn default_overlay_click_through() -> bool {
    true
}

fn default_mono_scout_amount() -> f32 {
    0.5
}

fn default_mono_breath_depth() -> f32 {
    crate::sound::directional::DEFAULT_DEPTH
}

fn default_mono_cue_subtractive() -> bool {
    true
}

fn default_mono_breath_timbre() -> f32 {
    crate::sound::directional::DEFAULT_TIMBRE
}

fn default_mono_buzz_depth() -> f32 {
    crate::sound::directional::DEFAULT_BUZZ_DEPTH
}

fn default_mono_comb_notch_db() -> f32 {
    crate::sound::directional::DEFAULT_COMB_NOTCH_DB
}

fn default_mono_opposite_level_db() -> f32 {
    crate::sound::directional::DEFAULT_OPPOSITE_LEVEL_DB
}

fn default_mono_opposite_duck_db() -> f32 {
    crate::sound::directional::DEFAULT_DUCK_DB
}

fn default_mono_cue_tail() -> f32 {
    crate::sound::directional::DEFAULT_CUE_TAIL
}

fn default_mono_cue_tail_level_db() -> f32 {
    crate::sound::directional::DEFAULT_TAIL_LEVEL_DB
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

fn default_layer_side_glow() -> f32 {
    1.0
}

fn default_edge_glow_strength() -> f32 {
    1.0
}

fn default_edge_vignette_width() -> f32 {
    0.16
}

fn default_glow_line_width() -> f32 {
    1.0
}

fn default_glow_min_visibility() -> f32 {
    0.35
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
            mono_level_version: MONO_LEVEL_VERSION,
            mono_tone_db: 0.0,
            mono_class_gain: false,
            mono_class_gain_db: default_mono_class_gain_db(),
            mono_scout: false,
            mono_scout_amount: default_mono_scout_amount(),
            radar_sensitivity: default_radar_sensitivity(),
            edge_glow_enabled: false,
            edge_glow_strength: default_edge_glow_strength(),
            edge_glow_style: EdgeGlowStyle::default(),
            edge_vignette_width: default_edge_vignette_width(),
            radar_glow_style: RadarGlowStyle::default(),
            glow_line_width: default_glow_line_width(),
            glow_min_visibility: default_glow_min_visibility(),
            speaker_layout_view: false,
            stereo_left_color: default_stereo_left_color(),
            stereo_right_color: default_stereo_right_color(),
            spatial_audio_view: false,
            line_thickness: default_line_thickness(),
            mono_cue_style: MonoCueStyle::default(),
            mono_breath_depth: default_mono_breath_depth(),
            mono_breath_timbre: default_mono_breath_timbre(),
            mono_buzz_depth: default_mono_buzz_depth(),
            mono_layer_styles: MonoLayerStyles::default(),
            mono_cue_subtractive: true,
            mono_comb_notch_db: default_mono_comb_notch_db(),
            mono_opposite_level_db: default_mono_opposite_level_db(),
            mono_opposite_duck_db: default_mono_opposite_duck_db(),
            mono_opposite_shadow: 0.0,
            mono_cue_tail: default_mono_cue_tail(),
            mono_cue_tail_level_db: default_mono_cue_tail_level_db(),
            direction_contrast_enabled: false,
            visualization_mode: VisualizationMode::default(),
            speaker_setup: SpeakerSetup::default(),
            layer_arc_inner: default_layer_arc_inner(),
            layer_arc_spacing: default_layer_arc_spacing(),
            layer_steps_gain: default_layer_steps_gain(),
            layer_shots_gain: default_layer_gain(),
            layer_rest_gain: default_layer_gain(),
            layer_shots_color: default_layer_shots_color(),
            layer_steps_color: default_layer_steps_color(),
            layer_rest_color: default_layer_rest_color(),
            layer_shots_left_color: default_layer_shots_color(),
            layer_shots_right_color: default_layer_shots_color(),
            layer_steps_left_color: default_layer_steps_color(),
            layer_steps_right_color: default_layer_steps_color(),
            layer_rest_left_color: default_layer_rest_color(),
            layer_rest_right_color: default_layer_rest_color(),
            layer_colors_version: LAYER_COLORS_VERSION,
            layer_side_glow: default_layer_side_glow(),
            mode_appearance: BTreeMap::new(),
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
///
/// Style są w dwóch grupach. Kanałowe ([`Self::CHANNEL`]) dają efekt całemu
/// kanałowi przeciwnemu, więc dostaje go też kopia dźwięków ze środka i ze swojej
/// strony. Kierunkowe ([`Self::DIRECTIONAL`]) dają efekt tylko dźwiękowi, który
/// przychodzi z przeciwnej strony (patrz `sound::directional`); opóźniają wyjście o ok. 6 ms
/// (rozdzielanie i limiter).
///
/// Usunięte style (Pulsowanie, Stłumiony, Ciemny, Odbicie, Wycięcie środka,
/// Falowanie) wczytują się ze starego configu jako styl domyślny, patrz
/// [`Self::from_config_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum MonoCueStyle {
    /// Strona przeciwna gra niezmieniona, tak samo jak swoja — brak wskazówki
    /// (zachowanie sprzed dodania tej funkcji).
    Off,
    /// Ring modulation ~45 Hz — wyraźne, niskie "brzęczenie".
    #[default]
    Buzz,
    /// Ring modulation ~140 Hz — ten sam mechanizm co `Buzz`, ale szybciej,
    /// więc brzmi wyżej/"piskliwiej".
    Squeaky,
    /// Podbicie wysokich tonów — strona przeciwna brzmi jaśniej/cieńciej, bez
    /// zmiany rytmu głośności.
    Bright,
    /// Grzebieniowy: krótkie opóźnienie 1,2 ms zmieszane z sygnałem daje wycięcia (pierwsze ok.
    /// 416 Hz) i "pustą", przestrzenną barwę. Bez modulacji.
    Comb,
    /// Oddech: wolna modulacja głośności 4 Hz (głębokość 40 %) w paśmie 300 Hz – 4 kHz.
    Tremolo,
    /// Oddech kierunkowy: oddech 4 Hz bez zmiany średniej głośności i cieńsza barwa, tylko dla
    /// dźwięku z przeciwnej strony. Głębokość i barwa z suwaków. Najlepszy dla dźwięków ciągłych.
    DirectionalBreath,
    /// Oddech + brzęczenie: jak `DirectionalBreath`, a do tego lekkie brzęczenie 45 Hz, które
    /// mieści kilka cykli w krótkim dźwięku (krok), więc oznacza też krótkie dźwięki.
    BreathBuzz,
    /// `Buzz` tylko dla dźwięku z przeciwnej strony.
    DirectionalBuzz,
    /// `Squeaky` tylko dla dźwięku z przeciwnej strony.
    DirectionalSqueaky,
    /// `Bright` tylko dla dźwięku z przeciwnej strony.
    DirectionalBright,
    /// `Comb` tylko dla dźwięku z przeciwnej strony.
    DirectionalComb,
    /// Wg warstw: osobny styl kierunkowy dla strzałów, kroków i reszty (patrz
    /// [`MonoLayerStyles`]), zależnie od tego, co radar rozpoznał po przeciwnej stronie. Czeka na
    /// wstępną decyzję klasy (20 ms, potem poprawiana), więc wyjście dostaje ok. 40 ms opóźnienia.
    PerLayer,
}

impl MonoCueStyle {
    /// Style do wyboru dla jednej warstwy w stylu „Wg warstw”: bez wskazówki albo jeden ze
    /// stylów kierunkowych.
    pub const LAYER_CHOICES: &'static [Self] = &[
        Self::Off,
        Self::DirectionalBreath,
        Self::BreathBuzz,
        Self::DirectionalBuzz,
        Self::DirectionalSqueaky,
        Self::DirectionalBright,
        Self::DirectionalComb,
    ];

    /// Style kanałowe, w kolejności przycisków w UI.
    pub const CHANNEL: &'static [Self] = &[
        Self::Off,
        Self::Buzz,
        Self::Squeaky,
        Self::Tremolo,
        Self::Bright,
        Self::Comb,
    ];

    /// Style kierunkowe, w kolejności przycisków w UI.
    pub const DIRECTIONAL: &'static [Self] = &[
        Self::DirectionalBreath,
        Self::BreathBuzz,
        Self::DirectionalBuzz,
        Self::DirectionalSqueaky,
        Self::DirectionalBright,
        Self::DirectionalComb,
        Self::PerLayer,
    ];

    pub const ALL: &'static [Self] = &[
        Self::Off,
        Self::Buzz,
        Self::Squeaky,
        Self::Tremolo,
        Self::Bright,
        Self::Comb,
        Self::DirectionalBreath,
        Self::BreathBuzz,
        Self::DirectionalBuzz,
        Self::DirectionalSqueaky,
        Self::DirectionalBright,
        Self::DirectionalComb,
        Self::PerLayer,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Wyłączone",
            Self::Buzz => "Brzęczenie",
            Self::Squeaky => "Bardziej piskliwy",
            Self::Bright => "Jasny",
            Self::Comb => "Grzebień",
            Self::Tremolo => "Oddech",
            Self::DirectionalBreath => "Oddech kierunkowy",
            Self::BreathBuzz => "Oddech + brzęczenie",
            Self::DirectionalBuzz => "Brzęczenie kierunkowe",
            Self::DirectionalSqueaky => "Piskliwy kierunkowy",
            Self::DirectionalBright => "Jasny kierunkowy",
            Self::DirectionalComb => "Grzebień kierunkowy",
            Self::PerLayer => "Wg warstw",
        }
    }

    /// Styl sprowadzony do wyboru dozwolonego dla jednej warstwy ([`Self::LAYER_CHOICES`]):
    /// styl kanałowy daje swoją wersję kierunkową, a „Wg warstw” daje brak wskazówki.
    pub fn as_layer_choice(self) -> Self {
        match self {
            Self::Buzz => Self::DirectionalBuzz,
            Self::Squeaky => Self::DirectionalSqueaky,
            Self::Tremolo => Self::DirectionalBreath,
            Self::Bright => Self::DirectionalBright,
            Self::Comb => Self::DirectionalComb,
            Self::PerLayer => Self::Off,
            other => other,
        }
    }

    /// Czy styl daje efekt tylko dźwiękowi z przeciwnej strony (a nie całemu kanałowi).
    pub fn is_directional(self) -> bool {
        Self::DIRECTIONAL.contains(&self)
    }

    /// Czy styl kierunkowy ma oddech i cieńszą barwę (suwaki głębokości i barwy).
    pub fn uses_breath(self) -> bool {
        matches!(self, Self::DirectionalBreath | Self::BreathBuzz)
    }

    /// Czy styl kierunkowy oznacza mocno tylko początek dźwięku, a potem wycofuje wskazówkę
    /// („oznacz i wycofaj”). Tak działają style bez wolnej modulacji. Oddech zostaje ciągły,
    /// bo jego wskazówką jest właśnie długie wahanie.
    pub fn fades_after_onset(self) -> bool {
        matches!(
            self,
            Self::DirectionalComb | Self::DirectionalBright | Self::DirectionalSqueaky | Self::DirectionalBuzz
        )
    }

    /// Styl kanałowy, którego efekt wersja kierunkowa nakłada na dźwięk z przeciwnej strony.
    /// Dla pozostałych stylów zwraca je same.
    pub fn base_cue(self) -> Self {
        match self {
            Self::DirectionalBuzz => Self::Buzz,
            Self::DirectionalSqueaky => Self::Squeaky,
            Self::DirectionalBright => Self::Bright,
            Self::DirectionalComb => Self::Comb,
            other => other,
        }
    }

    /// Styl z nazwy zapisanej w configu. Nazwy usuniętych stylów (`Pulse`, `Muffled`, `Dark`,
    /// `Echo`, `Notch`, `PitchWobble`) i nieznane nazwy dają styl domyślny, żeby stary plik
    /// wczytał się w całości, zamiast odrzucić wszystkie ustawienia.
    pub fn from_config_name(name: &str) -> Self {
        Self::ALL
            .iter()
            .copied()
            .find(|style| style.config_name() == name)
            .unwrap_or_default()
    }

    /// Nazwa stylu w configu (ta sama, którą zapisuje `serde::Serialize`).
    fn config_name(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Buzz => "Buzz",
            Self::Squeaky => "Squeaky",
            Self::Bright => "Bright",
            Self::Comb => "Comb",
            Self::Tremolo => "Tremolo",
            Self::DirectionalBreath => "DirectionalBreath",
            Self::BreathBuzz => "BreathBuzz",
            Self::DirectionalBuzz => "DirectionalBuzz",
            Self::DirectionalSqueaky => "DirectionalSqueaky",
            Self::DirectionalBright => "DirectionalBright",
            Self::DirectionalComb => "DirectionalComb",
            Self::PerLayer => "PerLayer",
        }
    }
}

/// Styl wskazówki dla każdej warstwy w stylu „Wg warstw” (`MonoCueStyle::PerLayer`). Domyślnie
/// trzy różne faktury: grzebień (pusta barwa) dla strzałów, brzęczenie dla kroków (mieści kilka
/// cykli w krótkim kroku) i oddech dla reszty (najlepszy dla dźwięków ciągłych).
///
/// Każda warstwa ma też głośność (dB, [`MONO_LAYER_VOLUME_DB_MIN`]..[`MONO_LAYER_VOLUME_DB_MAX`])
/// i intensywność wskazówki (0–1: 0 = bez efektu, 1 = pełny efekt stylu). Obie dotyczą tylko
/// dźwięku z przeciwnej strony rozpoznanego jako ta warstwa. Domyślnie 0 dB i 100 %, czyli bez
/// zmian. Plik zapisany przed tymi polami wczytuje się z wartościami domyślnymi.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MonoLayerStyles {
    pub shots: MonoCueStyle,
    pub steps: MonoCueStyle,
    pub rest: MonoCueStyle,
    pub shots_volume_db: f32,
    pub steps_volume_db: f32,
    pub rest_volume_db: f32,
    pub shots_intensity: f32,
    pub steps_intensity: f32,
    pub rest_intensity: f32,
}

/// Zakres głośności warstwy w stylu „Wg warstw” (dB). W dół więcej niż w górę: ściszenie
/// głośnych strzałów nie grozi przesterowaniem, a podbicie powyżej 12 dB już tak.
pub const MONO_LAYER_VOLUME_DB_MIN: f32 = -18.0;
pub const MONO_LAYER_VOLUME_DB_MAX: f32 = 12.0;

impl Default for MonoLayerStyles {
    fn default() -> Self {
        Self {
            shots: MonoCueStyle::DirectionalComb,
            steps: MonoCueStyle::DirectionalBuzz,
            rest: MonoCueStyle::DirectionalBreath,
            shots_volume_db: 0.0,
            steps_volume_db: 0.0,
            rest_volume_db: 0.0,
            shots_intensity: 1.0,
            steps_intensity: 1.0,
            rest_intensity: 1.0,
        }
    }
}

impl MonoLayerStyles {
    /// Głośność warstwy (dB), do suwaka w UI.
    pub fn volume_db_mut(&mut self, layer: layers::Layer) -> &mut f32 {
        match layer {
            layers::Layer::Shots => &mut self.shots_volume_db,
            layers::Layer::Steps => &mut self.steps_volume_db,
            layers::Layer::Rest => &mut self.rest_volume_db,
        }
    }

    /// Intensywność wskazówki warstwy (0–1), do suwaka w UI.
    pub fn intensity_mut(&mut self, layer: layers::Layer) -> &mut f32 {
        match layer {
            layers::Layer::Shots => &mut self.shots_intensity,
            layers::Layer::Steps => &mut self.steps_intensity,
            layers::Layer::Rest => &mut self.rest_intensity,
        }
    }

    /// Głośności w kolejności `layers::Layer::ALL`, przycięte do dozwolonego zakresu (dB).
    pub fn volume_db_in_layer_order(&self) -> [f32; 3] {
        [self.shots_volume_db, self.steps_volume_db, self.rest_volume_db]
            .map(|db| db.clamp(MONO_LAYER_VOLUME_DB_MIN, MONO_LAYER_VOLUME_DB_MAX))
    }

    /// Intensywności w kolejności `layers::Layer::ALL`, przycięte do 0–1.
    pub fn intensity_in_layer_order(&self) -> [f32; 3] {
        [self.shots_intensity, self.steps_intensity, self.rest_intensity].map(|v| v.clamp(0.0, 1.0))
    }

    pub fn get(&self, layer: layers::Layer) -> MonoCueStyle {
        match layer {
            layers::Layer::Shots => self.shots,
            layers::Layer::Steps => self.steps,
            layers::Layer::Rest => self.rest,
        }
    }

    pub fn get_mut(&mut self, layer: layers::Layer) -> &mut MonoCueStyle {
        match layer {
            layers::Layer::Shots => &mut self.shots,
            layers::Layer::Steps => &mut self.steps,
            layers::Layer::Rest => &mut self.rest,
        }
    }

    /// Style w kolejności `layers::Layer::ALL`, każdy sprowadzony do dozwolonego wyboru
    /// (patrz [`MonoCueStyle::as_layer_choice`]).
    pub fn in_layer_order(&self) -> [MonoCueStyle; 3] {
        layers::Layer::ALL.map(|layer| self.get(layer).as_layer_choice())
    }

    /// Czy któraś warstwa używa stylu z oddechem (suwaki głębokości i barwy w UI).
    pub fn any_uses_breath(&self) -> bool {
        self.in_layer_order().iter().any(|s| s.uses_breath())
    }

    /// Czy któraś warstwa używa „Oddech + brzęczenie” (suwak siły brzęczenia w UI).
    pub fn any_uses_breath_buzz(&self) -> bool {
        self.in_layer_order().contains(&MonoCueStyle::BreathBuzz)
    }
}

impl<'de> serde::Deserialize<'de> for MonoCueStyle {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Self::from_config_name(&name))
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
/// limit pamięci, NIE docelowe opóźnienie. Opóźnienie utrzymuje [`ring_clock::RingClock`].
const MONO_RING_CAPACITY: usize = 65_536;

/// Opóźnienie wyjścia z włączoną zmianą głośności klas (0,25 s przy 48 kHz). Tyle trwa pełna
/// klasyfikacja zdarzenia (ok. 0,15 s, patrz `layers::LayerTracker::take_ready`) plus zapas, żeby
/// zmiana zdążyła przed początkiem dźwięku. Styl „Wg warstw” czeka tylko na decyzję wstępną
/// (patrz [`MonoRing::target_fill`]), a bez obu cel jest krótki (patrz
/// [`ring_clock::RingClock::short_target`]).
const CLASS_GAIN_LATENCY_SAMPLES: usize = 12_000;

/// Zapas ponad cel, po którego przekroczeniu bufor jest od razu przycinany do celu (60 ms przy
/// 48 kHz). Dryf zegarów wejścia i wyjścia koryguje na bieżąco [`ring_clock::RingClock`], więc
/// to tylko zabezpieczenie na skoki: wyjście stanęło (zmiana urządzenia), wejście oddało naraz
/// zaległe kawałki. Bez przycinania bufor rósłby do [`MONO_RING_CAPACITY`] i tam utykał.
const RING_TRIM_MARGIN_SAMPLES: usize = 2_880;

/// Styl „Wg warstw”: jak długo po początku zdarzenia trzymamy styl jego warstwy (0,3 s przy
/// 48 kHz). Kroki w biegu są co ok. 0,25–0,35 s, więc styl kroków trwa przez cały bieg.
const LAYER_STYLE_HOLD_FRAMES: u64 = 14_400;
/// Styl warstwy włącza się tyle przed początkiem zdarzenia (5 ms), żeby płynne przejście
/// zdążyło przed uderzeniem.
const LAYER_STYLE_LOOKAHEAD_FRAMES: u64 = 240;
/// Najwięcej zdarzeń warstw czekających na wyjście; starsze odrzucamy.
const MAX_LAYER_EVENTS: usize = 64;

/// Warstwa dźwięku, który gra w danej ramce po każdej stronie, z klasyfikacji radaru. Zasila
/// styl wskazówki „Wg warstw” (`MonoCueStyle::PerLayer`). `None` = brak rozpoznanego zdarzenia
/// po tej stronie (dźwięk ze środka nie trafia na żadną stronę).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SideLayers {
    pub(crate) left: Option<layers::Layer>,
    pub(crate) right: Option<layers::Layer>,
}

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

/// Dynamika "Scout" w mono: ciche dźwięki (poniżej −45 dB, ale nie szum poniżej −60 dB) są
/// podbijane, a głośne (powyżej −15 dB) ściszane. Nachylenie 0,5 dB na dB; najwięcej zależy od siły.
///
/// Progi są w dB pary z bufora mono, czyli w skali stereo (patrz [`MONO_PAN_SCALE`]): dźwięk z gry
/// o szczycie X dBFS w kanale L albo P daje tu X − 6 dB, tak samo w strumieniu 2.0, 5.1 i 7.1. Ciche
/// dźwięki z gry to więc ok. −54 … −39 dBFS, a ściszanie zaczyna się od ok. −9 dBFS. Dawniej bufor
/// był dzielony przez liczbę kanałów strumienia i przy VB-Cable 7.1 te same progi wypadały 12 dB
/// wyżej (−42 … −27 dBFS), a ściszanie głośnych w ogóle się nie włączało (szczyty do −18 dB).
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
/// Wyprzedzenie obwiedni Scout (ramki): zmiana głośności startuje tyle przed szczytem, żeby
/// narastanie 5 ms zdążyło przed uderzeniem. 5 ms przy 48 kHz.
const SCOUT_LOOKAHEAD_FRAMES: usize = 240;

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
    /// Cel opóźnienia przy ostatniej zmianie trybu (zmiana głośności klas, „Wg warstw”); patrz
    /// [`Self::update_target_latency`].
    mode_target: f32,
    /// Krótki cel opóźnienia, korekta dryfu zegarów i licznik przerw.
    clock: ring_clock::RingClock,
    /// Ułamkowa pozycja odczytu między `frames[0]` i `frames[1]` (0..1). Odczyt bierze z bufora
    /// odrobinę więcej albo mniej próbek, niż oddaje, żeby skorygować dryf zegarów.
    read_frac: f64,
    /// Ostatnia zdjęta ramka: lewy sąsiad do interpolacji.
    prev_frame: (f32, f32),
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
    /// Docelowa zmiana Scout w dB dla każdej ramki w kolejce, równolegle do `frames`. Liczona
    /// przy wejściu, więc odczyt widzi ją z wyprzedzeniem.
    scout_targets: VecDeque<f32>,
    /// Czy po podniesieniu celu opóźnienia czekamy, aż bufor się do niego dopełni. Bez tego
    /// opóźnienie zostawałoby krótkie (cel tylko przycina bufor), a decyzja klasy przychodziłaby
    /// już po początku dźwięku.
    priming: bool,
    /// Czy styl „Wg warstw” jest włączony (zbieramy zdarzenia warstw i znaczniki ramek).
    layer_styles_enabled: bool,
    /// Zdarzenia warstw: (numer ramki początku, warstwa, strona: 0 = lewa, 1 = prawa, `None` =
    /// środek), posortowane po początku. Ten sam początek może przyjść dwa razy: najpierw decyzja
    /// wstępna, potem pełna, która ją poprawia (patrz [`Self::next_layers`]).
    layer_events: VecDeque<(u64, layers::Layer, Option<usize>)>,
    /// Bieżące zdarzenie po każdej stronie: (numer ramki początku, warstwa).
    layer_active: [Option<(u64, layers::Layer)>; 2],
    /// Znaczniki ostatnio odczytanej ramki, powtarzane, gdy bufor na chwilę wyschnie.
    last_layers: SideLayers,
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
            mode_target: 0.0,
            clock: ring_clock::RingClock::default(),
            read_frac: 0.0,
            prev_frame: (0.0, 0.0),
            class_gain_db: MONO_CLASS_GAIN_DB_DEFAULT,
            scout_enabled: false,
            scout_amount: 0.5,
            scout_env: 0.0,
            scout_gain: 1.0,
            scout_targets: VecDeque::new(),
            priming: false,
            layer_styles_enabled: false,
            layer_events: VecDeque::new(),
            layer_active: [None; 2],
            last_layers: SideLayers::default(),
        }
    }
}

impl MonoRing {
    /// Ustawia cel opóźnienia: długi, gdy działa zmiana głośności klas albo styl „Wg warstw”
    /// (obie czekają na decyzję klasy), krótki w przeciwnym razie. Po podniesieniu celu odczyt
    /// czeka, aż bufor się dopełni (patrz [`Self::priming`]).
    fn update_target_latency(&mut self) {
        let target = self.target_fill();
        if target > self.mode_target && (self.frames.len() as f32) < target {
            self.priming = true;
        } else if target < self.mode_target {
            self.priming = false;
        }
        self.mode_target = target;
        self.clock.restart_average();
    }

    /// Docelowe średnie zapełnienie bufora (próbki wejścia):
    /// - zmiana głośności klas: [`CLASS_GAIN_LATENCY_SAMPLES`], bo czeka na pełną decyzję,
    /// - „Wg warstw”: tyle, żeby decyzja wstępna (po [`layers::QUICK_DECISION_S`]) zdążyła
    ///   [`LAYER_STYLE_LOOKAHEAD_FRAMES`] przed dźwiękiem (ok. 38 ms przy 10 ms kawałkach),
    /// - inaczej krótki cel wg urządzeń (patrz [`ring_clock::RingClock::short_target`]).
    fn target_fill(&self) -> f32 {
        if self.class_gain_enabled {
            CLASS_GAIN_LATENCY_SAMPLES as f32
        } else if self.layer_styles_enabled {
            let wait = layers::QUICK_DECISION_S * self.clock.sample_rate() + LAYER_STYLE_LOOKAHEAD_FRAMES as f32;
            self.clock.target_with_wait(wait)
        } else {
            self.clock.short_target()
        }
    }

    /// Bieżąca próbka odczytu: `frames[0]` albo, przy ułamkowej pozycji, interpolacja Hermite'a
    /// między `frames[0]` i `frames[1]`. Wymaga co najmniej jednej ramki (dwóch przy ułamku).
    fn current_frame(&self) -> (f32, f32) {
        let x0 = self.frames[0];
        if self.read_frac == 0.0 {
            return x0;
        }
        let t = self.read_frac as f32;
        let xm1 = self.prev_frame;
        let x1 = self.frames[1];
        let x2 = self.frames.get(2).copied().unwrap_or(x1);
        (
            ring_clock::hermite(xm1.0, x0.0, x1.0, x2.0, t),
            ring_clock::hermite(xm1.1, x0.1, x1.1, x2.1, t),
        )
    }

    /// Przesuwa odczyt o `step` próbek wejścia, zdejmując minięte ramki (razem z ich celem Scout).
    fn advance(&mut self, step: f64) {
        self.read_frac += step;
        while self.read_frac >= 1.0 {
            let Some(frame) = self.frames.pop_front() else {
                self.read_frac = 0.0;
                break;
            };
            self.prev_frame = frame;
            self.scout_targets.pop_front();
            self.front_index += 1;
            self.read_frac -= 1.0;
        }
    }

    /// Zdejmuje `count` najstarszych ramek naraz (przycięcie albo przepełnienie).
    fn drop_front(&mut self, count: usize) {
        let count = count.min(self.frames.len());
        if count == 0 {
            return;
        }
        self.prev_frame = self.frames[count - 1];
        self.frames.drain(..count);
        self.scout_targets.drain(..count.min(self.scout_targets.len()));
        self.front_index += count as u64;
        self.read_frac = 0.0;
    }

    /// Warstwy grające po obu stronach w ramce `front_index`, dla stylu „Wg warstw”. Zdarzenie
    /// zaczyna działać [`LAYER_STYLE_LOOKAHEAD_FRAMES`] przed swoim początkiem i trwa
    /// [`LAYER_STYLE_HOLD_FRAMES`] albo do następnego zdarzenia po tej samej stronie.
    fn next_layers(&mut self) -> SideLayers {
        if !self.layer_styles_enabled {
            self.last_layers = SideLayers::default();
            return self.last_layers;
        }
        let frame = self.front_index;
        while let Some(&(onset, layer, side)) = self.layer_events.front() {
            if onset > frame + LAYER_STYLE_LOOKAHEAD_FRAMES {
                break;
            }
            self.layer_events.pop_front();
            // Ten sam początek to ten sam dźwięk: pełna decyzja zastępuje wstępną, także gdy
            // wyszła inna strona albo środek (wtedy znika ze strony, na której była).
            for (slot, active) in self.layer_active.iter_mut().enumerate() {
                if side != Some(slot) && active.is_some_and(|(o, _)| o == onset) {
                    *active = None;
                }
            }
            // Spóźniona poprawka starszego dźwięku nie nadpisuje nowszego po tej samej stronie.
            if let Some(side) = side
                && self.layer_active[side].is_none_or(|(o, _)| o <= onset)
            {
                self.layer_active[side] = Some((onset, layer));
            }
        }
        let current = |active: Option<(u64, layers::Layer)>| {
            active.filter(|&(onset, _)| frame < onset + LAYER_STYLE_HOLD_FRAMES).map(|(_, layer)| layer)
        };
        self.last_layers = SideLayers {
            left: current(self.layer_active[0]),
            right: current(self.layer_active[1]),
        };
        self.last_layers
    }

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

    /// Wejście: poziom ramki i zmiana Scout w dB dla niej. Przy wyłączonym Scout zwraca 0 dB.
    fn scout_level_db(&mut self, l: f32, r: f32) -> f32 {
        let peak = l.abs().max(r.abs());
        self.scout_env = peak.max(self.scout_env * (-1.0 / SCOUT_ENV_RELEASE_FRAMES).exp());
        if !self.scout_enabled {
            return 0.0;
        }
        let level_db = 20.0 * self.scout_env.max(1e-6).log10();
        scout_target_db(level_db, self.scout_amount)
    }

    /// Wyjście: stosuje Scout do bieżącej ramki (przód kolejki). Ramki w kolejce dają
    /// wyprzedzenie: bierzemy najmocniejsze ściszenie z oknem [`SCOUT_LOOKAHEAD_FRAMES`],
    /// licząc od bieżącej. Oba kanały dostają ten sam mnożnik, więc balans L/P zostaje.
    fn scout_apply(&mut self, l: f32, r: f32) -> (f32, f32) {
        if !self.scout_enabled {
            self.scout_gain = 1.0;
            return (l, r);
        }
        let ahead = self.scout_targets.iter().take(SCOUT_LOOKAHEAD_FRAMES).copied().reduce(f32::min).unwrap_or(0.0);
        let target = 10f32.powf(ahead / 20.0);
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
    /// Ostatnia próbka realnie zdjęta z `mono_ring` przez [`Self::read_stereo_pan_tagged`]
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
    /// Dryf zegarów wejścia i wyjścia koryguje na bieżąco odczyt (patrz
    /// [`Self::read_stereo_pan_tagged`] i [`ring_clock::RingClock`]). Gdy bufor mimo to urośnie
    /// o [`RING_TRIM_MARGIN_SAMPLES`] ponad cel (wyjście stanęło, wejście oddało naraz zaległe
    /// kawałki), jest od razu przycinany do celu, zamiast rosnąć do [`MONO_RING_CAPACITY`].
    ///
    /// `sample_rate` to częstotliwość wejścia: z niej liczone są cele w ms i tempo odczytu
    /// względem wyjścia.
    pub(crate) fn push_stereo_pan_at(&self, samples: &[(f32, f32)], sample_rate: f32) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            ring.clock.note_input(samples.len(), sample_rate);
            for &s in samples {
                if ring.frames.len() >= MONO_RING_CAPACITY {
                    ring.drop_front(1);
                }
                let db = ring.scout_level_db(s.0, s.1);
                ring.frames.push_back(s);
                ring.scout_targets.push_back(db);
            }
            let target = ring.target_fill().round() as usize;
            if ring.frames.len() > target + RING_TRIM_MARGIN_SAMPLES {
                let excess = ring.frames.len() - target;
                ring.drop_front(excess);
                ring.clock.restart_average();
            }
        }
    }

    /// Jak [`Self::push_stereo_pan_at`] przy 48 kHz (testy).
    #[cfg(test)]
    pub(crate) fn push_stereo_pan(&self, samples: &[(f32, f32)]) {
        self.push_stereo_pan_at(samples, 48_000.0);
    }

    /// Opóźnienie bufora mono do pokazania w UI.
    pub fn mono_latency(&self) -> ring_clock::MonoLatency {
        self.mono_ring
            .lock()
            .map(|ring| ring.clock.status(ring.target_fill()))
            .unwrap_or(ring_clock::MonoLatency { buffer_ms: None, target_ms: 0.0, underruns: 0, correction_ppm: 0.0 })
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
            ring.update_target_latency();
            if !enabled {
                ring.class_gains.clear();
                ring.current_event = None;
                ring.gain = 1.0;
            }
        }
    }

    /// Włącza albo wyłącza zbieranie zdarzeń warstw dla stylu wskazówki „Wg warstw”. Wymaga
    /// dłuższego bufora (patrz [`MonoRing::target_fill`]), żeby wstępna decyzja warstwy zdążyła
    /// przed początkiem dźwięku.
    pub(crate) fn set_mono_layer_styles(&self, enabled: bool) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            if ring.layer_styles_enabled == enabled {
                return;
            }
            ring.layer_styles_enabled = enabled;
            ring.update_target_latency();
            if !enabled {
                ring.layer_events.clear();
                ring.layer_active = [None; 2];
                ring.last_layers = SideLayers::default();
            }
        }
    }

    /// Zapisuje zdarzenia warstw dla stylu „Wg warstw”, z numerem ramki początku (jak
    /// [`Self::push_class_gains`]) i stroną. Zdarzenie bliżej środka niż martwa strefa
    /// rozdzielania kierunkowego (ok. 1,5 dB różnicy L/P) nie trafia na żadną stronę.
    ///
    /// Najpierw przychodzą decyzje wstępne (po ok. 20 ms, `layers::LayerTracker::take_quick`),
    /// potem pełne (po ok. 0,15 s) z tym samym początkiem. Pełna poprawia wstępną: zmienia
    /// warstwę albo stronę od chwili, w której przyszła (płynnie, patrz `mono_router`).
    pub(crate) fn push_layer_events(&self, hits: &[layers::LayerHit], sample_rate: f32) {
        if hits.is_empty() {
            return;
        }
        if let Ok(mut ring) = self.mono_ring.lock() {
            if !ring.layer_styles_enabled {
                return;
            }
            let dead_zone = 10f32.powf(crate::sound::directional::ILD_DEAD_ZONE_DB / 20.0);
            let min_pan = layers::pan_from_rms(1.0, dead_zone);
            for hit in hits {
                let side = if hit.pan >= min_pan {
                    Some(1)
                } else if hit.pan <= -min_pan {
                    Some(0)
                } else {
                    None
                };
                let frame = (hit.onset_s * f64::from(sample_rate)).round().max(0.0) as u64;
                // Po początku, za wszystkimi wcześniejszymi i równymi (poprawka idzie po wstępnej).
                let at = ring.layer_events.iter().rposition(|&(onset, _, _)| onset <= frame).map_or(0, |i| i + 1);
                ring.layer_events.insert(at, (frame, hit.layer, side));
            }
            while ring.layer_events.len() > MAX_LAYER_EVENTS {
                ring.layer_events.pop_front();
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
    ///
    /// Do każdej ramki dokłada warstwy grające w niej po obu stronach (patrz [`SideLayers`]),
    /// dla stylu wskazówki „Wg warstw”.
    ///
    /// Po podniesieniu celu opóźnienia (zmiana głośności klas albo styl „Wg warstw”) odczyt
    /// najpierw czeka, aż bufor dopełni się do celu, i do tego czasu powtarza wygaszaną ostatnią
    /// próbkę, tak jak przy wyschnięciu bufora. To jednorazowa przerwa przy przełączeniu. Tak samo
    /// po wyschnięciu bufora: zamiast trzeszczeć przy każdym kolejnym odczycie, czeka raz, aż
    /// bufor dojdzie do celu, a cel rośnie o mały zapas (patrz [`ring_clock::RingClock`]).
    ///
    /// Tempo odczytu: `output_rate` to częstotliwość wyjścia. Odczyt bierze z bufora próbki
    /// wejścia w stosunku częstotliwości wejścia do wyjścia, poprawionym o korektę dryfu zegarów
    /// (najwyżej ±0,5 %), z interpolacją Hermite'a. Przy równych częstotliwościach i bez korekty
    /// próbki przechodzą bez zmian.
    pub(crate) fn read_stereo_pan_tagged(&self, count: usize, output_rate: f32) -> (Vec<(f32, f32)>, Vec<SideLayers>) {
        const HOLD_DECAY: f32 = 0.999;
        let mut out = Vec::with_capacity(count);
        let mut tags = Vec::with_capacity(count);
        let mut last = self.last_pan.lock().map(|g| *g).unwrap_or((0.0, 0.0));
        if let Ok(mut guard) = self.mono_ring.lock() {
            let ring = &mut *guard;
            let target = ring.target_fill();
            if ring.priming && ring.frames.len() as f32 >= target {
                ring.priming = false;
                ring.clock.restart_average();
            }
            let nominal = f64::from(ring.clock.sample_rate()) / f64::from(output_rate.max(1.0));
            let need = count as f64 * nominal;
            let correction = if ring.priming {
                0.0
            } else {
                let fill = ring.frames.len() as f32 - ring.read_frac as f32;
                ring.clock.on_read(fill, need as f32, target)
            };
            let step = nominal * (1.0 + correction);
            let mut dried_up = false;
            for _ in 0..count {
                let needed = if ring.read_frac > 0.0 { 2 } else { 1 };
                let sample = if ring.priming || ring.frames.len() < needed {
                    dried_up |= !ring.priming;
                    last = (last.0 * HOLD_DECAY, last.1 * HOLD_DECAY);
                    tags.push(ring.last_layers);
                    let mult = ring.gain * ring.scout_gain;
                    (last.0 * mult, last.1 * mult)
                } else {
                    let s = ring.current_frame();
                    last = s;
                    let gain = ring.next_gain();
                    tags.push(ring.next_layers());
                    let shaped = ring.scout_apply(s.0 * gain, s.1 * gain);
                    ring.advance(step);
                    shaped
                };
                out.push(sample);
            }
            if dried_up {
                ring.clock.on_underrun();
                ring.read_frac = 0.0;
                ring.priming = true;
            }
        } else {
            out.resize(count, (0.0, 0.0));
            tags.resize(count, SideLayers::default());
        }
        if let Ok(mut guard) = self.last_pan.lock() {
            *guard = last;
        }
        (out, tags)
    }

    /// Same próbki z [`Self::read_stereo_pan_tagged`], bez znaczników warstw (do testów).
    #[cfg(test)]
    pub(crate) fn read_stereo_pan(&self, count: usize) -> Vec<(f32, f32)> {
        self.read_stereo_pan_tagged(count, 48_000.0).0
    }

    /// Czy logger próbek (`sample_logger::SampleLogger`) jest aktualnie
    /// włączony — odczytywane przez checkbox w UI, żeby odzwierciedlić stan
    /// nawet po tym, jak zapis pliku faktycznie się nie powiódł.
    pub fn sample_logging_enabled(&self) -> bool {
        self.sample_logger.is_enabled()
    }

    /// Czas strumienia analizy (s), patrz [`layers::LayerTracker::stream_seconds`]. Logger radaru
    /// zapisuje go na starcie odtwarzania, żeby czas wierszy liczyć od tej chwili.
    pub fn layer_stream_seconds(&self) -> f64 {
        self.layers.lock().map(|t| t.stream_seconds()).unwrap_or(0.0)
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
    let (quick_hits, layer_hits, accepted) = {
        let mut tracker = analysis.layers.lock().unwrap_or_else(|e| e.into_inner());
        tracker.push_frames(samples, channels, sample_rate);
        let accepted = tracker.onset_allowed() && channel_activity.iter().any(|a| a.is_onset);
        if accepted && tracker.has_external_onsets() {
            tracker.add_onset();
        }
        // Najpierw decyzje wstępne, potem pełne: pełna tego samego zdarzenia ma poprawić wstępną.
        (tracker.take_quick(), tracker.take_ready(), accepted)
    };

    // Zmiana głośności w trybie mono: strzały ciszej, kroki głośniej (patrz `push_class_gains`),
    // tylko z pełnych decyzji. Styl wskazówki „Wg warstw” (patrz `push_layer_events`) bierze
    // decyzje wstępne po 20 ms, a pełne je poprawiają.
    analysis.push_class_gains(&layer_hits, sample_rate);
    analysis.push_layer_events(&quick_hits, sample_rate);
    analysis.push_layer_events(&layer_hits, sample_rate);

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
    analysis.push_stereo_pan_at(&downmix_to_stereo_pan(samples, channels), sample_rate);
}

/// Rozdziela zinterlacowane próbki na dwa "ważone" miksy mono: `(left,
/// right)`, gdzie każdy to udział energii kanałów danej strony w tej ramce
/// (patrz [`channel_pan_weights`]) — w przeciwieństwie do dawnego, płaskiego
/// uśrednienia WSZYSTKICH kanałów, tu informacja L/R przeżywa aż do
/// `mono_router`, który (w trybie jednousznym) może nią oznaczyć dźwięk z
/// przeciwnej strony efektem (ring modulation), zamiast go bezpowrotnie
/// zlewać z resztą.
///
/// Ważoną sumę każdej strony mnożymy przez stałe [`MONO_PAN_SCALE`], a nie
/// dzielimy przez liczbę kanałów strumienia — patrz uzasadnienie tam.
fn downmix_to_stereo_pan(interleaved: &[f32], channels: usize) -> Vec<(f32, f32)> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        // Mono źródło: brak informacji o stronie, więc dzielimy po równo. Mono `s` to to samo co
        // stereo L = P = s, więc para wychodzi taka jak ze strumienia 2.0.
        return interleaved.iter().map(|&s| (s * MONO_PAN_SCALE, s * MONO_PAN_SCALE)).collect();
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
            (left * MONO_PAN_SCALE, right * MONO_PAN_SCALE)
        })
        .collect()
}

/// Skala pary "ważonego mono" z [`downmix_to_stereo_pan`]: ważoną sumę kanałów każdej strony
/// mnożymy przez stałe 0,5, czyli tak, jak uśrednia się strumień stereo. Dla 2.0 para to
/// `(L/2, P/2)`, a `left + right` to średnia mono, jak dotąd.
///
/// Dawniej sumy były dzielone przez liczbę kanałów strumienia, więc puste kanały ściszały sygnał.
/// VB-Cable jest ustawiony na 7.1 (8 kanałów, wymóg z README), a gra daje stereo tylko na FL/FR,
/// więc na żywo bufor dostawał FL/8, a z nagrania 2-kanałowego (`sound::playback`) FL/2, czyli
/// o 12 dB więcej. Progi Scout ([`SCOUT_QUIET_DB`] i dalsze) są bezwzględne na tej parze, więc
/// działały inaczej na żywo i z nagrania, a ściszanie głośnych na żywo nie włączało się wcale.
///
/// Normalizacja sumą wag strony z układu kanałów ([`channel_pan_weights`]) tego nie naprawia:
/// w 7.1 lewa strona ma wagę 4 (FL, RL, SL i po pół C i LFE), a w 2.0 wagę 1, więc ten sam FL
/// wychodzi jako FL/4 wobec FL/1 — dalej 12 dB różnicy. To nadal średnia po kanałach strony,
/// a dźwięk z gry stereo i tak jest tylko w jednym z nich. Stała skala nie zależy od tego, ile
/// kanałów jest pustych: ten sam dźwięk z gry daje tę samą parę w strumieniu 2.0, 5.1 i 7.1.
///
/// Cena: pełny miks 5.1/7.1 z sygnałem we wszystkich kanałach może dać parę powyżej 1,0. W buforze
/// to tylko `f32`, a wyjście i tak przechodzi przez ściszanie głośnych Scout i miękki limiter
/// (`mono_router::soft_limit`). Przy 7.1 bufor jest teraz 4 razy (12,04 dB) głośniejszy niż dawniej,
/// więc `mono_boost_db` ze starego configu jest obniżane o tyle samo (patrz
/// [`RadarConfig::migrate_mono_level`]) i głośność w słuchawce się nie zmienia.
pub(crate) const MONO_PAN_SCALE: f32 = 0.5;

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
    fn each_mode_keeps_its_own_appearance() {
        let mut cfg = RadarConfig::default();
        cfg.visualization_mode = VisualizationMode::Both;
        cfg.radius = 150.0;
        // Pierwsza wizyta w trybie 3+3 bierze wygląd bieżący.
        cfg.switch_mode(VisualizationMode::LayerBars);
        assert_eq!(cfg.radius, 150.0, "nowy tryb startuje z bieżącym wyglądem");
        cfg.radius = 220.0;
        cfg.layer_shots_gain = 3.5;

        cfg.switch_mode(VisualizationMode::Both);
        assert_eq!(cfg.radius, 150.0, "powrót przywraca wygląd tego trybu");
        assert_eq!(cfg.layer_shots_gain, RadarConfig::default().layer_shots_gain, "gain 3+3 nie dotyczy radaru");

        cfg.switch_mode(VisualizationMode::LayerBars);
        assert_eq!(cfg.radius, 220.0);
        assert_eq!(cfg.layer_shots_gain, 3.5);
    }

    #[test]
    fn remembered_appearance_survives_json_round_trip() {
        let mut cfg = RadarConfig::default();
        cfg.visualization_mode = VisualizationMode::OnlyRadar;
        cfg.radius = 180.0;
        cfg.switch_mode(VisualizationMode::LayerBars22);
        cfg.radius = 90.0;
        cfg.remember_appearance();

        let json = serde_json::to_string(&cfg).unwrap();
        let back: RadarConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode_appearance[&VisualizationMode::OnlyRadar].radius, 180.0);
        assert_eq!(back.mode_appearance[&VisualizationMode::LayerBars22].radius, 90.0);
        assert_eq!(back, cfg);
    }

    #[test]
    fn config_without_mode_appearance_starts_empty() {
        // Plik sprzed tej mapy: cała reszta konfiguracji jest, brakuje tylko `mode_appearance`.
        let mut value = serde_json::to_value(RadarConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("mode_appearance");
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert!(cfg.mode_appearance.is_empty());
    }

    #[test]
    fn surround_offers_only_radar_modes() {
        use VisualizationMode as M;
        assert!(SpeakerSetup::Surround.allows(M::OnlyRadar));
        assert!(SpeakerSetup::Surround.allows(M::None));
        for mode in [M::OnlyBars, M::Both, M::LayerBars, M::LayerBars22] {
            assert!(!SpeakerSetup::Surround.allows(mode), "5.1/7.1 nie pokazuje: {}", mode.label());
        }
        for mode in M::ALL {
            assert!(SpeakerSetup::Stereo.allows(*mode), "stereo pokazuje wszystko");
        }
    }

    #[test]
    fn config_without_speaker_setup_defaults_to_stereo() {
        let mut value = serde_json::to_value(RadarConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("speaker_setup");
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.speaker_setup, SpeakerSetup::Stereo);
    }

    #[test]
    fn removed_cue_styles_load_as_default_without_dropping_other_settings() {
        for name in ["Pulse", "Muffled", "Dark", "Echo", "Notch", "PitchWobble", "Nieznany"] {
            let mut value = serde_json::to_value(RadarConfig { radius: 77.0, ..RadarConfig::default() }).unwrap();
            value["mono_cue_style"] = serde_json::Value::String(name.into());
            let cfg: RadarConfig = serde_json::from_value(value).unwrap();
            assert_eq!(cfg.mono_cue_style, MonoCueStyle::default(), "{name}");
            assert_eq!(cfg.radius, 77.0, "{name}: reszta ustawień ma zostać");
        }
    }

    #[test]
    fn every_cue_style_survives_json_round_trip() {
        // Nazwy z `from_config_name` muszą zgadzać się z tym, co zapisuje `Serialize`.
        for &style in MonoCueStyle::ALL {
            let json = serde_json::to_string(&style).unwrap();
            let back: MonoCueStyle = serde_json::from_str(&json).unwrap();
            assert_eq!(back, style, "{json}");
        }
    }

    #[test]
    fn opposite_side_settings_load_with_defaults_from_an_older_config() {
        // Plik sprzed ustawień strony przeciwnej: poprawki włączone z wartościami domyślnymi.
        use crate::sound::directional as d;
        let mut value = serde_json::to_value(RadarConfig { mono_breath_depth: 0.4, ..RadarConfig::default() }).unwrap();
        for key in [
            "mono_cue_subtractive",
            "mono_comb_notch_db",
            "mono_opposite_level_db",
            "mono_opposite_duck_db",
            "mono_opposite_shadow",
            "mono_cue_tail",
            "mono_cue_tail_level_db",
        ] {
            assert!(value.as_object_mut().unwrap().remove(key).is_some(), "{key}");
        }
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert!(cfg.mono_cue_subtractive);
        assert_eq!(cfg.mono_comb_notch_db, d::DEFAULT_COMB_NOTCH_DB);
        assert_eq!(cfg.mono_opposite_level_db, d::DEFAULT_OPPOSITE_LEVEL_DB);
        assert_eq!(cfg.mono_opposite_duck_db, d::DEFAULT_DUCK_DB);
        assert_eq!(cfg.mono_opposite_shadow, 0.0);
        assert_eq!(cfg.mono_cue_tail, d::DEFAULT_CUE_TAIL);
        assert_eq!(cfg.mono_cue_tail_level_db, d::DEFAULT_TAIL_LEVEL_DB);
        assert_eq!(cfg.mono_breath_depth, 0.4, "zapisana głębokość oddechu zostaje");
        assert_eq!(cfg, RadarConfig { mono_breath_depth: 0.4, ..RadarConfig::default() });
    }

    #[test]
    fn layer_styles_load_with_defaults_and_fall_back_to_allowed_choices() {
        // Stary plik bez pola: domyślne style warstw.
        let mut value = serde_json::to_value(RadarConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("mono_layer_styles");
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.mono_layer_styles, MonoLayerStyles::default());

        // Tylko jedna warstwa w pliku: reszta z wartości domyślnych.
        let mut value = serde_json::to_value(RadarConfig::default()).unwrap();
        value["mono_layer_styles"] = serde_json::json!({ "steps": "BreathBuzz" });
        let cfg: RadarConfig = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.mono_layer_styles.steps, MonoCueStyle::BreathBuzz);
        assert_eq!(cfg.mono_layer_styles.shots, MonoLayerStyles::default().shots);
        // Zapis sprzed głośności i intensywności warstw: bez zmian (0 dB, 100 %).
        assert_eq!(cfg.mono_layer_styles.volume_db_in_layer_order(), [0.0; 3]);
        assert_eq!(cfg.mono_layer_styles.intensity_in_layer_order(), [1.0; 3]);

        // Wartości spoza zakresu są przycinane.
        let wild = MonoLayerStyles {
            shots_volume_db: -40.0,
            steps_volume_db: 30.0,
            rest_intensity: 2.0,
            ..MonoLayerStyles::default()
        };
        assert_eq!(wild.volume_db_in_layer_order(), [MONO_LAYER_VOLUME_DB_MIN, MONO_LAYER_VOLUME_DB_MAX, 0.0]);
        assert_eq!(wild.intensity_in_layer_order(), [1.0, 1.0, 1.0]);

        // Styl spoza listy dla warstwy: kanałowy daje kierunkowy, „Wg warstw” daje brak wskazówki.
        let styles = MonoLayerStyles {
            shots: MonoCueStyle::Buzz,
            steps: MonoCueStyle::PerLayer,
            rest: MonoCueStyle::Tremolo,
            ..MonoLayerStyles::default()
        };
        assert_eq!(
            styles.in_layer_order(),
            [MonoCueStyle::DirectionalBuzz, MonoCueStyle::Off, MonoCueStyle::DirectionalBreath]
        );
        for &style in MonoCueStyle::ALL {
            assert!(MonoCueStyle::LAYER_CHOICES.contains(&style.as_layer_choice()), "{style:?}");
        }
    }

    #[test]
    fn cue_style_groups_cover_every_style_exactly_once() {
        let grouped: Vec<MonoCueStyle> =
            MonoCueStyle::CHANNEL.iter().chain(MonoCueStyle::DIRECTIONAL).copied().collect();
        assert_eq!(grouped.len(), MonoCueStyle::ALL.len());
        for style in MonoCueStyle::ALL {
            assert_eq!(grouped.iter().filter(|s| *s == style).count(), 1, "{style:?}");
        }
        for &style in MonoCueStyle::DIRECTIONAL {
            assert!(style.is_directional(), "{style:?}");
            if !style.uses_breath() && style != MonoCueStyle::PerLayer {
                assert!(MonoCueStyle::CHANNEL.contains(&style.base_cue()), "{style:?}: efekt stylu kanałowego");
            }
        }
    }

    #[test]
    fn position_stays_shared_between_modes() {
        let mut cfg = RadarConfig::default();
        cfg.anchor_x = 0.25;
        cfg.switch_mode(VisualizationMode::LayerBars);
        assert_eq!(cfg.anchor_x, 0.25);
    }

    #[test]
    fn appearance_round_trips_through_apply() {
        let mut cfg = RadarConfig::default();
        let mut appearance = cfg.appearance();
        appearance.stereo_left_color = [1, 2, 3];
        appearance.layer_arc_inner = 0.6;
        cfg.apply_appearance(&appearance);
        assert_eq!(cfg.appearance(), appearance);
    }

    #[test]
    fn default_side_colors_match_the_layer_color() {
        use layers::Layer;
        let cfg = RadarConfig::default();
        assert_eq!(cfg.layer_side_colors(Layer::Shots), [cfg.layer_shots_color; 2]);
        assert_eq!(cfg.layer_side_colors(Layer::Steps), [cfg.layer_steps_color; 2]);
        assert_eq!(cfg.layer_side_colors(Layer::Rest), [cfg.layer_rest_color; 2]);
        assert_eq!(cfg.layer_colors_version, LAYER_COLORS_VERSION);
        assert_eq!(ModeAppearance::default(), cfg.appearance());
    }

    #[test]
    fn whole_side_color_changes_only_that_side() {
        use layers::Layer;
        let mut cfg = RadarConfig::default();
        cfg.set_layer_side_color_for_all(0, [0, 0, 255]);
        cfg.set_layer_side_color_for_all(1, [180, 40, 200]);
        for layer in Layer::ALL {
            assert_eq!(cfg.layer_side_colors(layer), [[0, 0, 255], [180, 40, 200]], "{layer:?}");
        }
        *cfg.layer_side_colors_mut(Layer::Steps)[1] = [1, 2, 3];
        assert_eq!(cfg.layer_side_colors(Layer::Steps), [[0, 0, 255], [1, 2, 3]]);
        assert_eq!(cfg.layer_side_colors(Layer::Shots)[1], [180, 40, 200], "inne warstwy bez zmian");
    }

    #[test]
    fn side_colors_are_remembered_per_mode() {
        use layers::Layer;
        let mut cfg = RadarConfig { visualization_mode: VisualizationMode::LayerBars, ..RadarConfig::default() };
        cfg.set_layer_side_color_for_all(0, [0, 0, 255]);
        cfg.switch_mode(VisualizationMode::LayerBars22);
        cfg.set_layer_side_color_for_all(1, [180, 40, 200]);
        cfg.switch_mode(VisualizationMode::LayerBars);
        assert_eq!(cfg.layer_side_colors(Layer::Rest), [[0, 0, 255], cfg.layer_rest_color]);
        cfg.switch_mode(VisualizationMode::LayerBars22);
        assert_eq!(cfg.layer_side_colors(Layer::Rest), [[0, 0, 255], [180, 40, 200]]);
    }

    #[test]
    fn migrating_side_colors_copies_each_layer_color_once() {
        use layers::Layer;
        let mut cfg = RadarConfig {
            layer_shots_color: [1, 2, 3],
            layer_colors_version: 0,
            ..RadarConfig::default()
        };
        let mut appearance = cfg.appearance();
        appearance.layer_rest_color = [7, 8, 9];
        cfg.mode_appearance.insert(VisualizationMode::OnlyBars, appearance);

        cfg.migrate_layer_side_colors();
        assert_eq!(cfg.layer_side_colors(Layer::Shots), [[1, 2, 3]; 2]);
        let bars = cfg.mode_appearance[&VisualizationMode::OnlyBars];
        assert_eq!([bars.layer_rest_left_color, bars.layer_rest_right_color], [[7, 8, 9]; 2]);
        assert_eq!(cfg.layer_colors_version, LAYER_COLORS_VERSION);

        // Przeliczony plik nie traci już kolorów stron ustawionych później.
        cfg.set_layer_side_color_for_all(0, [0, 0, 255]);
        cfg.migrate_layer_side_colors();
        assert_eq!(cfg.layer_side_colors(Layer::Shots)[0], [0, 0, 255]);
    }

    #[test]
    fn appearance_of_a_mode_is_not_affected_by_other_modes() {
        let mut cfg = RadarConfig::default();
        cfg.visualization_mode = VisualizationMode::OnlyBars;
        cfg.stereo_left_color = [9, 9, 9];
        cfg.switch_mode(VisualizationMode::None);
        cfg.stereo_left_color = [5, 5, 5];
        cfg.remember_appearance();
        assert_eq!(cfg.mode_appearance[&VisualizationMode::OnlyBars].stereo_left_color, [9, 9, 9]);
        assert_eq!(cfg.mode_appearance[&VisualizationMode::None].stereo_left_color, [5, 5, 5]);
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
    fn downmix_to_stereo_pan_sums_to_the_stereo_scaled_mix_of_all_channels() {
        // `left + right` to suma WSZYSTKICH kanałów w skali stereo (C i LFE idą po połowie w obie
        // strony), bo `mono_router::MonoEar::Both` gra właśnie tę sumę. Dla 2.0 to dawna średnia.
        let interleaved = [1.0, 0.5, 0.2, -0.4, -0.1, 0.3, 0.6, -0.2];
        for &channels in &[2usize, 6, 8] {
            let (l, r) = downmix_to_stereo_pan(&interleaved[..channels], channels)[0];
            let expected = interleaved[..channels].iter().sum::<f32>() * MONO_PAN_SCALE;
            assert!((l + r - expected).abs() < 1e-6, "channels={channels}");
        }
        let (l, r) = downmix_to_stereo_pan(&interleaved[..2], 2)[0];
        assert!((l + r - (1.0 + 0.5) / 2.0).abs() < 1e-6, "2.0: średnia (L + P) / 2 jak dotąd");
        // Mono `s` to to samo co stereo L = P = s.
        assert_eq!(downmix_to_stereo_pan(&[0.8], 1), downmix_to_stereo_pan(&[0.8, 0.8], 2));
    }

    /// Dźwięk z gry w stereo `(L, P)` rozłożony na strumień o `channels` kanałach: sygnał w FL/FR,
    /// reszta cisza, tak jak gra stereo na VB-Cable ustawionym na 7.1.
    fn game_stream(game: &[(f32, f32)], channels: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(game.len() * channels);
        for &(l, r) in game {
            out.push(l);
            out.push(r);
            out.resize(out.len() + channels - 2, 0.0);
        }
        out
    }

    /// 0,3 s dźwięku z gry przy 48 kHz: cichy ton 1 kHz (L ok. −50 dBFS), strzał 50 ms (L ok. −3 dBFS)
    /// od 0,1 s i znów cichy ton. Prawa strona trochę ciszej.
    fn quiet_shot_quiet() -> Vec<(f32, f32)> {
        (0..14_400)
            .map(|i| {
                let amp = if (4_800..7_200).contains(&i) { 0.7 } else { 0.003 };
                let s = amp * (std::f32::consts::TAU * 1_000.0 * i as f32 / 48_000.0).sin();
                (s, s * 0.8)
            })
            .collect()
    }

    /// Wynik przepuszczenia strumienia przez tor mono: wyjście bufora, docelowa zmiana Scout (dB)
    /// każdej ramki i obwiednia, z której Scout liczy poziom, po każdym kawałku 10 ms.
    struct MonoRun {
        out: Vec<(f32, f32)>,
        scout_db: Vec<f32>,
        scout_env: Vec<f32>,
    }

    /// Strumień idzie przez `update_analysis` kawałkami po 10 ms, tak jak żywe przechwytywanie
    /// (`fetch_audio_from_device`) i odtwarzanie nagrania (`playback`). Po każdym kawałku bufor
    /// mono jest czytany do końca, jak robi to `mono_router`.
    fn run_mono(stream: &[f32], channels: usize, scout: bool) -> MonoRun {
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_scout(scout, 1.0);
        let mut run = MonoRun { out: Vec::new(), scout_db: Vec::new(), scout_env: Vec::new() };
        for chunk in stream.chunks(480 * channels) {
            update_analysis(chunk, channels, 48_000.0, &analysis, None);
            {
                let ring = analysis.mono_ring.lock().unwrap();
                run.scout_db.extend(ring.scout_targets.iter().copied());
                run.scout_env.push(ring.scout_env);
            }
            run.out.extend(analysis.read_stereo_pan(chunk.len() / channels));
        }
        run
    }

    #[test]
    fn silent_extra_channels_do_not_change_the_pan_pair() {
        // Gra w stereo przez VB-Cable 7.1: sygnał tylko w FL/FR. Para ma być taka sama jak ze
        // strumienia 2.0. Dawniej dzielenie przez liczbę kanałów dawało tu 4 razy mniej (−12 dB).
        let game = [(0.7, -0.3), (0.1, 0.25), (-0.5, 0.0)];
        let stereo = downmix_to_stereo_pan(&game_stream(&game, 2), 2);
        assert_eq!(downmix_to_stereo_pan(&game_stream(&game, 8), 8), stereo, "7.1");
        assert_eq!(downmix_to_stereo_pan(&game_stream(&game, 6), 6), stereo, "5.1");
        assert_eq!(stereo[0], (0.35, -0.15), "skala stereo: połowa kanału z gry");
    }

    #[test]
    fn scout_sees_the_same_level_for_game_audio_in_2_and_8_channel_streams() {
        let game = quiet_shot_quiet();
        let two = run_mono(&game_stream(&game, 2), 2, true);
        let eight = run_mono(&game_stream(&game, 8), 8, true);
        assert_eq!(eight.scout_env, two.scout_env, "ten sam poziom Scout przy 2 i 8 kanałach");
        assert_eq!(eight.scout_db, two.scout_db, "te same decyzje Scout przy 2 i 8 kanałach");
        assert_eq!(eight.out, two.out, "to samo wyjście bufora mono przy 2 i 8 kanałach");

        // Strzał ok. −3 dBFS z gry to w skali pary ok. −9 dB, czyli powyżej progu ściszania. Dawniej
        // przy 8 kanałach wychodziło ok. −21 dB i „ścisz głośne” się nie włączało.
        let shot_db = 20.0 * eight.scout_env.iter().copied().fold(0.0f32, f32::max).log10();
        assert!(shot_db > SCOUT_LOUD_DB, "poziom strzału w skali pary: {shot_db} dB");
        let deepest = eight.scout_db.iter().copied().fold(0.0f32, f32::min);
        assert!(deepest < -2.0, "przy 8 kanałach Scout ścisza strzał: {deepest} dB");
        // Cichy ton (ok. −50 dBFS z gry, −56 dB w skali pary) jest podbijany, tak samo przy 8 kanałach.
        assert!(eight.scout_db[100..4_800].iter().all(|&db| db > 0.0), "cichy ton przed strzałem podbity");
    }

    #[test]
    fn played_back_recording_has_the_same_mono_level_as_live_capture() {
        // Nagranie (`recorder`) zapisuje dwa pierwsze kanały żywego strumienia, a odtwarzanie
        // (`playback`) woła `update_analysis` z 2 kanałami. Żywe przechwytywanie z VB-Cable 7.1 to
        // 8 kanałów. Ten sam dźwięk z gry ma dać w buforze mono dokładnie to samo.
        let game = quiet_shot_quiet();
        let live = game_stream(&game, 8);
        let recording: Vec<f32> = live.chunks_exact(8).flat_map(|f| [f[0], f[1]]).collect();
        let from_live = run_mono(&live, 8, false);
        let from_recording = run_mono(&recording, 2, false);
        assert_eq!(from_live.out.len(), game.len());
        assert_eq!(from_recording.out, from_live.out, "nagranie i żywe audio na tym samym poziomie");
        // Skala stereo: strona pary to połowa kanału z gry, także w środku strzału.
        let (l, r) = from_live.out[5_000];
        assert!((l - game[5_000].0 * MONO_PAN_SCALE).abs() < 1e-6 && (r - game[5_000].1 * MONO_PAN_SCALE).abs() < 1e-6);
    }

    #[test]
    fn migrated_boost_keeps_the_headphone_level_of_live_7_1_capture() {
        // Użytkownik z VB-Cable 7.1 i +22 dB: dawniej bufor dostawał FL/8, teraz FL/2. Po migracji
        // wzmocnienie ma dać w słuchawce ten sam poziom, a nie 12 dB głośniej.
        let db_to_linear = |db: f32| 10f32.powf(db / 20.0);
        let mut cfg = RadarConfig { mono_boost_db: 22.0, mono_level_version: 0, ..RadarConfig::default() };
        cfg.migrate_mono_level();
        assert_eq!(cfg.mono_level_version, MONO_LEVEL_VERSION);

        let fl = 0.3;
        let before = fl / 8.0 * db_to_linear(22.0);
        let pan = downmix_to_stereo_pan(&game_stream(&[(fl, 0.0)], 8), 8);
        let after = pan[0].0 * db_to_linear(cfg.mono_boost_db);
        let jump_db = 20.0 * (after / before).log10();
        assert!(jump_db.abs() < 0.01, "skok głośności po migracji: {jump_db} dB");

        // Drugie przeliczenie (kolejne wczytanie) niczego już nie odejmuje.
        let boost = cfg.mono_boost_db;
        cfg.migrate_mono_level();
        assert_eq!(cfg.mono_boost_db, boost);
    }

    #[test]
    fn new_config_is_not_migrated_and_old_boosts_fit_the_slider() {
        let mut fresh = RadarConfig::default();
        fresh.migrate_mono_level();
        assert_eq!(fresh.mono_boost_db, 0.0, "nowy config zostaje bez zmian");

        // Najniższe dawne ustawienie suwaka (−12 dB) po migracji mieści się w nowym zakresie.
        let mut lowest = RadarConfig { mono_boost_db: -12.0, mono_level_version: 0, ..RadarConfig::default() };
        lowest.migrate_mono_level();
        assert!(lowest.mono_boost_db >= MONO_BOOST_DB_MIN && lowest.mono_boost_db < -23.9, "{}", lowest.mono_boost_db);

        assert!(
            (LEGACY_MONO_BOOST_OFFSET_DB - 20.0 * (8.0 * MONO_PAN_SCALE).log10()).abs() < 1e-3,
            "przesunięcie migracji to stosunek dawnego FL/8 do FL·MONO_PAN_SCALE"
        );
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
            peak_left_dbfs: -10.0,
            peak_right_dbfs: -10.0,
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
        quiet.push_stereo_pan(&vec![(0.5, 0.5); CLASS_GAIN_LATENCY_SAMPLES + 2_000]);
        quiet.push_class_gains(&[class_hit(layers::Layer::Shots, 0.01)], 48_000.0); // ramka 480
        let out = quiet.read_stereo_pan(3_000);
        assert_eq!(out[100], (0.5, 0.5), "przed zdarzeniem bez zmian");
        assert!((out[2_000].0 - 0.25).abs() < 0.02, "strzał ok. −6 dB: {}", out[2_000].0);

        // Krok: głośniej po zdarzeniu (ok. +6 dB).
        let loud = SharedAudioAnalysis::new();
        loud.set_mono_class_gain(true, MONO_CLASS_GAIN_DB_DEFAULT);
        loud.push_stereo_pan(&vec![(0.3, 0.3); CLASS_GAIN_LATENCY_SAMPLES + 2_000]);
        loud.push_class_gains(&[class_hit(layers::Layer::Steps, 0.01)], 48_000.0);
        let out = loud.read_stereo_pan(3_000);
        assert!(out[2_000].0 > 0.55, "krok ok. +6 dB: {}", out[2_000].0);
    }

    #[test]
    fn scout_boosts_quiet_and_tames_loud_sounds() {
        let quiet = SharedAudioAnalysis::new();
        quiet.set_mono_scout(true, 1.0);
        let mut last = (0.0, 0.0);
        for _ in 0..10 {
            quiet.push_stereo_pan(&vec![(0.0032, 0.0032); 2_000]); // ok. −50 dBFS
            last = *quiet.read_stereo_pan(2_000).last().unwrap();
        }
        assert!(last.0 > 0.0038, "ciche podbite, ma {}", last.0);

        let loud = SharedAudioAnalysis::new();
        loud.set_mono_scout(true, 1.0);
        loud.push_stereo_pan(&vec![(0.9, 0.9); 2_000]); // ok. −1 dBFS
        let out = loud.read_stereo_pan(2_000);
        assert!(out[1_999].0 < 0.6, "głośne ściszone, ma {}", out[1_999].0);
    }

    #[test]
    fn scout_keeps_silence_silent_and_is_a_no_op_when_off() {
        let silent = SharedAudioAnalysis::new();
        silent.set_mono_scout(true, 1.0);
        silent.push_stereo_pan(&vec![(0.0, 0.0); 5_000]);
        assert!(silent.read_stereo_pan(5_000).iter().all(|&s| s == (0.0, 0.0)), "cisza zostaje ciszą");
        let off = SharedAudioAnalysis::new();
        off.push_stereo_pan(&[(0.3, -0.2)]);
        assert_eq!(off.read_stereo_pan(1)[0], (0.3, -0.2), "wyłączony Scout nie zmienia sygnału");
    }

    #[test]
    fn scout_lookahead_turns_down_a_shot_before_its_peak() {
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_scout(true, 1.0);
        let mut input = vec![(0.005, 0.005); 2_000];
        input.extend(vec![(0.9, 0.9); 240]); // strzał
        input.extend(vec![(0.005, 0.005); 2_000]);
        analysis.push_stereo_pan(&input);
        let out = analysis.read_stereo_pan(input.len());
        let peak = out[2_000..2_240].iter().map(|s| s.0).fold(0.0f32, f32::max);
        // Bez wyprzedzenia szczyt zostałby ok. 0,93; z wyprzedzeniem jest ściszony już w trakcie narastania.
        assert!(peak < 0.72, "szczyt strzału ściszony zanim dojdzie: {peak}");
    }

    #[test]
    fn class_gain_amount_follows_the_slider() {
        // Suwak 12 dB: strzał ok. 0,5 × 10^(−12/20) ≈ 0,125.
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_class_gain(true, 12.0);
        analysis.push_stereo_pan(&vec![(0.5, 0.5); CLASS_GAIN_LATENCY_SAMPLES + 2_000]);
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

    fn side_hit(layer: layers::Layer, onset_s: f64, pan: f32) -> layers::LayerHit {
        layers::LayerHit { pan, ..class_hit(layer, onset_s) }
    }

    /// Warstwy, jakie odczyt widzi w ramkach `from..to` (indeks = numer ramki od `from`).
    fn layer_timeline(analysis: &SharedAudioAnalysis, from: u64, to: u64) -> Vec<SideLayers> {
        let mut ring = analysis.mono_ring.lock().unwrap();
        (from..to)
            .map(|frame| {
                ring.front_index = frame;
                ring.next_layers()
            })
            .collect()
    }

    #[test]
    fn layer_events_tag_the_side_they_came_from() {
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_layer_styles(true);
        analysis.push_layer_events(
            &[
                side_hit(layers::Layer::Steps, 0.1, 0.8),  // ramka 4 800, prawa
                side_hit(layers::Layer::Shots, 0.2, -0.8), // ramka 9 600, lewa
                side_hit(layers::Layer::Shots, 0.3, 0.0),  // środek: bez strony
            ],
            48_000.0,
        );
        let tags = layer_timeline(&analysis, 0, 40_000);
        let lookahead = LAYER_STYLE_LOOKAHEAD_FRAMES as usize;
        let hold = LAYER_STYLE_HOLD_FRAMES as usize;
        assert_eq!(tags[4_800 - lookahead - 1].right, None, "przed zdarzeniem nic");
        assert_eq!(tags[4_800 - lookahead].right, Some(layers::Layer::Steps), "styl rusza 5 ms przed zdarzeniem");
        assert_eq!(tags[4_800 + hold - 1].right, Some(layers::Layer::Steps));
        assert_eq!(tags[4_800 + hold].right, None, "po czasie trzymania styl wraca do reszty");
        assert_eq!(tags[10_000].left, Some(layers::Layer::Shots), "strzał z lewej oznacza lewą stronę");
        assert!(tags.iter().all(|t| t.right != Some(layers::Layer::Shots)), "strzał ze środka nie trafia na stronę");
    }

    #[test]
    fn full_decision_corrects_the_quick_one_from_when_it_arrives() {
        use layers::Layer::{Shots, Steps};
        let hold = LAYER_STYLE_HOLD_FRAMES as usize;
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_layer_styles(true);
        // Decyzja wstępna: krok z prawej w ramce 4 800.
        analysis.push_layer_events(&[side_hit(Steps, 0.1, 0.8)], 48_000.0);
        let early = layer_timeline(&analysis, 0, 6_000);
        assert_eq!(early[5_000].right, Some(Steps));
        // Pełna decyzja przychodzi później: to był strzał. Od tej chwili styl strzałów,
        // trzymany od początku dźwięku, nie od poprawki.
        analysis.push_layer_events(&[side_hit(Shots, 0.1, 0.8)], 48_000.0);
        let late = layer_timeline(&analysis, 6_000, 20_000);
        assert_eq!(late[0].right, Some(Shots));
        assert_eq!(late[4_800 + hold - 1 - 6_000].right, Some(Shots));
        assert_eq!(late[4_800 + hold - 6_000].right, None);
    }

    #[test]
    fn full_decision_can_move_the_sound_to_the_centre_or_the_other_side() {
        use layers::Layer::Steps;
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_layer_styles(true);
        analysis.push_layer_events(&[side_hit(Steps, 0.1, 0.8), side_hit(Steps, 0.5, 0.8)], 48_000.0);
        assert_eq!(layer_timeline(&analysis, 0, 5_000)[4_900].right, Some(Steps));
        // Pełna: środek, więc styl znika z prawej.
        analysis.push_layer_events(&[side_hit(Steps, 0.1, 0.0)], 48_000.0);
        assert_eq!(layer_timeline(&analysis, 5_000, 5_001)[0], SideLayers::default());
        // Drugi dźwięk: wstępnie prawa, pełna lewa.
        assert_eq!(layer_timeline(&analysis, 24_000, 24_100)[50].right, Some(Steps));
        analysis.push_layer_events(&[side_hit(Steps, 0.5, -0.8)], 48_000.0);
        let moved = layer_timeline(&analysis, 24_100, 24_101)[0];
        assert_eq!((moved.left, moved.right), (Some(Steps), None));
    }

    #[test]
    fn a_late_correction_does_not_override_a_newer_sound() {
        use layers::Layer::{Rest, Shots, Steps};
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_layer_styles(true);
        // Krok z prawej w 0,1 s, potem strzał z prawej w 0,15 s (oba wstępnie).
        analysis.push_layer_events(&[side_hit(Steps, 0.1, 0.8), side_hit(Shots, 0.15, 0.8)], 48_000.0);
        assert_eq!(layer_timeline(&analysis, 0, 8_000)[7_999].right, Some(Shots));
        // Spóźniona pełna decyzja pierwszego dźwięku nie zabiera stylu nowszemu.
        analysis.push_layer_events(&[side_hit(Rest, 0.1, 0.8)], 48_000.0);
        assert_eq!(layer_timeline(&analysis, 8_000, 8_001)[0].right, Some(Shots));
    }

    #[test]
    fn layer_events_are_ignored_while_the_per_layer_style_is_off() {
        let analysis = SharedAudioAnalysis::new();
        analysis.push_stereo_pan(&vec![(0.1, 0.1); 20_000]);
        analysis.push_layer_events(&[side_hit(layers::Layer::Steps, 0.01, 0.9)], 48_000.0);
        let (_, tags) = analysis.read_stereo_pan_tagged(10_000, 48_000.0);
        assert!(tags.iter().all(|t| *t == SideLayers::default()));
    }

    #[test]
    fn raising_the_latency_waits_until_the_buffer_holds_it() {
        // Bez dopełnienia opóźnienie zostałoby krótkie, a decyzja klasy (ok. 0,15 s po
        // początku dźwięku) przychodziłaby za późno.
        let analysis = SharedAudioAnalysis::new();
        analysis.push_stereo_pan(&vec![(0.2, 0.2); 2_400]);
        analysis.set_mono_class_gain(true, MONO_CLASS_GAIN_DB_DEFAULT);
        let waiting = analysis.read_stereo_pan(1_000);
        assert!(waiting.iter().all(|s| s.0 < 0.2), "czeka na dopełnienie, zamiast grać od razu");
        assert_eq!(analysis.mono_ring.lock().unwrap().frames.len(), 2_400, "nic nie zdjęte");
        analysis.push_stereo_pan(&vec![(0.5, 0.5); CLASS_GAIN_LATENCY_SAMPLES]);
        let playing = analysis.read_stereo_pan(10);
        assert!(playing.iter().all(|&s| s == (0.2, 0.2)), "po dopełnieniu gra od najstarszej próbki");
        let buffered = analysis.mono_ring.lock().unwrap().frames.len();
        assert!(buffered >= CLASS_GAIN_LATENCY_SAMPLES, "w buforze zostaje całe opóźnienie: {buffered}");
    }

    #[test]
    fn long_latency_buffer_stays_below_the_trim_limit() {
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_class_gain(true, MONO_CLASS_GAIN_DB_DEFAULT);
        for _ in 0..100 {
            analysis.push_stereo_pan(&[(0.1, 0.1); 500]);
        }
        let len = analysis.mono_ring.lock().unwrap().frames.len();
        assert!(len <= CLASS_GAIN_LATENCY_SAMPLES + RING_TRIM_MARGIN_SAMPLES, "opóźnienie do ok. 0,31 s, jest {len}");
    }

    #[test]
    fn push_stereo_pan_trims_back_to_target_instead_of_growing_to_hard_capacity() {
        // Regresja dla "prawie sekundowego opóźnienia": gdy wejście trwale
        // wyprzedza tempem wyjście (nikt nie woła read_stereo_pan wcale, jak
        // tutaj — skrajny przypadek tego samego zjawiska), bufor bez
        // aktywnego przycinania rósłby aż do MONO_RING_CAPACITY (~340ms) i
        // tam utykał. Powinien zamiast tego zatrzymać się blisko celu
        // (najwyżej 50 ms) plus zapas przycinania, NIE przy pełnej pojemności.
        let analysis = SharedAudioAnalysis::new();
        let chunk = vec![(0.1, 0.2); 500];
        for _ in 0..50 {
            analysis.push_stereo_pan(&chunk); // 50 * 500 = 25 000 próbek wepchniętych, bez ani jednego odczytu
        }

        let ring_len = analysis.mono_ring.lock().unwrap().frames.len();
        assert!(
            ring_len <= 2_400 + RING_TRIM_MARGIN_SAMPLES,
            "bufor nie powinien urosnąć powyżej progu przycinania, ma {ring_len}"
        );
        assert!(
            ring_len < MONO_RING_CAPACITY,
            "bufor utknął przy pełnej pojemności (~340ms) zamiast się przyciąć, ma {ring_len}"
        );
    }

    /// Wynik symulacji dwóch zegarów: zapełnienie bufora przy każdym odczycie (po rozruchu),
    /// liczba przerw i wyjście.
    struct ClockRun {
        fills: Vec<usize>,
        underruns: u64,
        out: Vec<(f32, f32)>,
        /// Styl „Wg warstw”: decyzje po rozruchu i te, które przyszły za późno.
        decisions: usize,
        late: usize,
    }

    /// `seconds` sekund strumienia: wejście co 10 ms (480 próbek przy `in_rate` z dryfem
    /// `drift_ppm`), wyjście co 10 ms przy `out_rate`, w kolejności czasu jak w prawdziwych
    /// callbackach. Sygnał to sinus 1 kHz.
    fn run_two_clocks(seconds: f64, in_rate: f64, drift_ppm: f64, out_rate: f64) -> ClockRun {
        run_two_clocks_with(seconds, in_rate, drift_ppm, out_rate, false)
    }

    /// Jak [`run_two_clocks`]; przy `layer_styles` włącza styl „Wg warstw” i co 0,37 s ogłasza
    /// dźwięk tak, jak robi to klasyfikator: decyzja wstępna z kawałkiem wejścia, który domyka
    /// okno 20 ms. Liczy decyzje, które przyszły za późno, czyli gdy odczyt minął już punkt
    /// włączenia stylu (5 ms przed dźwiękiem).
    fn run_two_clocks_with(seconds: f64, in_rate: f64, drift_ppm: f64, out_rate: f64, layer_styles: bool) -> ClockRun {
        let analysis = SharedAudioAnalysis::new();
        analysis.set_mono_layer_styles(layer_styles);
        let quick = (layers::QUICK_DECISION_S as f64 * in_rate).round() as u64;
        let mut next_onset = 12_345u64;
        let in_chunk = (in_rate / 100.0).round() as usize;
        let out_chunk = (out_rate / 100.0).round() as usize;
        // Prawdziwa częstotliwość wejścia z dryfem, a nominalna podawana jest do bufora.
        let in_period = in_chunk as f64 / (in_rate * (1.0 + drift_ppm * 1e-6));
        let out_period = out_chunk as f64 / out_rate;
        let (mut next_in, mut next_out) = (0.0f64, 0.003f64);
        let mut written = 0usize;
        let mut run = ClockRun { fills: Vec::new(), underruns: 0, out: Vec::new(), decisions: 0, late: 0 };
        while next_out < seconds {
            if next_in <= next_out {
                let chunk: Vec<(f32, f32)> = (written..written + in_chunk)
                    .map(|i| {
                        let s = 0.3 * (std::f64::consts::TAU * 1_000.0 * i as f64 / in_rate).sin() as f32;
                        (s, 0.5 * s)
                    })
                    .collect();
                written += in_chunk;
                analysis.push_stereo_pan_at(&chunk, in_rate as f32);
                next_in += in_period;
                while layer_styles && next_onset + quick <= written as u64 {
                    let reader = analysis.mono_ring.lock().unwrap().front_index;
                    if next_out > 1.0 {
                        run.decisions += 1;
                        if reader + LAYER_STYLE_LOOKAHEAD_FRAMES > next_onset {
                            run.late += 1;
                        }
                    }
                    next_onset += 17_760;
                }
            } else {
                if next_out > 1.0 {
                    run.fills.push(analysis.mono_ring.lock().unwrap().frames.len());
                }
                run.out.extend(analysis.read_stereo_pan_tagged(out_chunk, out_rate as f32).0);
                next_out += out_period;
            }
        }
        run.underruns = analysis.mono_latency().underruns;
        run
    }

    #[test]
    fn per_layer_decisions_arrive_before_the_sound_with_a_short_buffer() {
        // „Wg warstw” czeka tylko na decyzję wstępną: bufor ok. 38 ms (25 ms czekania, pół
        // kawałków i 3 ms zapasu) zamiast 250 ms, a każda decyzja i tak zdąża przed dźwiękiem.
        for drift_ppm in [300.0, -300.0] {
            let run = run_two_clocks_with(40.0, 48_000.0, drift_ppm, 48_000.0, true);
            assert_eq!(run.underruns, 0, "{drift_ppm} ppm: przerwy");
            assert!(run.decisions > 90, "{drift_ppm} ppm: decyzje {}", run.decisions);
            assert_eq!(run.late, 0, "{drift_ppm} ppm: spóźnione decyzje z {}", run.decisions);
            let late = &run.fills[run.fills.len() / 2..];
            let mean = late.iter().sum::<usize>() as f64 / late.len() as f64;
            // Zapełnienie przed odczytem: cel średni (1 824) plus pół odczytu (240).
            assert!((mean - 2_064.0).abs() < 250.0, "{drift_ppm} ppm: średnio {mean:.0} próbek");
        }
    }

    #[test]
    fn clock_drift_is_corrected_without_trims_and_the_buffer_stays_short() {
        // Wejście 300 ppm szybsze albo wolniejsze od wyjścia (dużo jak na zwykłe urządzenia):
        // przez 60 s bufor nie rośnie do progu przycinania ani nie wysycha, a po rozruchu
        // stoi przy celu ok. 18 ms (10 ms kawałek wejścia, pół odczytu i 3 ms zapasu).
        // Przy wolniejszym wejściu jego kawałki dwa razy przechodzą z „przed odczytem” na
        // „po odczycie”, czyli zapełnienie spada skokiem o 10 ms: to właśnie musi wytrzymać cel.
        for drift_ppm in [300.0, -300.0] {
            let run = run_two_clocks(60.0, 48_000.0, drift_ppm, 48_000.0);
            assert_eq!(run.underruns, 0, "{drift_ppm} ppm: przerwy");
            let late = &run.fills[run.fills.len() / 2..];
            let max = *late.iter().max().unwrap();
            let min = *late.iter().min().unwrap();
            let mean = late.iter().sum::<usize>() as f64 / late.len() as f64;
            assert!(max < 2_000, "{drift_ppm} ppm: najwyżej ok. 40 ms w buforze, jest {max}");
            assert!(min >= 480, "{drift_ppm} ppm: zawsze co najmniej jeden odczyt zapasu, jest {min}");
            // Zapełnienie przed odczytem: cel średni (864) plus pół odczytu (240).
            assert!((mean - 1_104.0).abs() < 250.0, "{drift_ppm} ppm: średnio {mean:.0} próbek");
        }
    }

    #[test]
    fn different_input_and_output_rates_play_at_the_right_speed() {
        // VB-Cable przy 44,1 kHz, słuchawki przy 48 kHz: bez przeliczenia tempa bufor
        // wysychałby co chwilę, a dźwięk grał o 9 % za wysoko.
        let run = run_two_clocks(20.0, 44_100.0, 0.0, 48_000.0);
        assert!(run.underruns <= 1, "przerwy {}", run.underruns);
        // Sinus 1 kHz z wejścia ma na wyjściu dalej 1 kHz: przejścia przez zero co 24 próbki.
        let tail = &run.out[run.out.len() - 48_000..];
        let crossings = tail.windows(2).filter(|w| (w[0].0 >= 0.0) != (w[1].0 >= 0.0)).count();
        assert!((crossings as i64 - 2_000).abs() <= 4, "1 kHz to 2000 przejść na sekundę, jest {crossings}");
    }

    #[test]
    fn drift_correction_keeps_a_sine_clean() {
        // Korekta tempa z interpolacją nie może słyszalnie zniekształcać: sinus 1 kHz po 30 s
        // z dryfem dalej leży blisko idealnego sinusa o tej samej amplitudzie (L = 0,3).
        let run = run_two_clocks(30.0, 48_000.0, 200.0, 48_000.0);
        let tail = &run.out[run.out.len() - 4_800..];
        let peak = tail.iter().map(|s| s.0.abs()).fold(0.0f32, f32::max);
        assert!((peak - 0.3).abs() < 0.003, "amplituda {peak}");
        // Odstęp od sinusa: suma próbek z przesunięciem o pół okresu (24 próbki) to zero.
        let residual = tail.windows(25).map(|w| (w[0].0 + w[24].0).abs()).fold(0.0f32, f32::max);
        assert!(residual < 0.003, "zniekształcenie {residual}");
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
