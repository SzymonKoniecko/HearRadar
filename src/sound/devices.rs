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
    /// Jak oznaczyć dźwięk z przeciwnej strony niż wybrane ucho w trybie
    /// jednousznym (`MonoEar::Left`/`Right`), żeby dało się po samym
    /// brzmieniu poznać, że był z drugiej strony — patrz
    /// [`MonoCueStyle`]/`mono_router::fill_output_f32`. `#[serde(default)]`
    /// korzysta z `#[default]` wariantu enuma, więc wczytanie starszego
    /// pliku configu (sprzed tego pola) włącza nową funkcję zamiast ją wyciszyć.
    #[serde(default)]
    pub mono_cue_style: MonoCueStyle,
}

fn default_overlay_click_through() -> bool {
    true
}

fn default_radar_sensitivity() -> f32 {
    crate::sound::direction::INTENSITY_SCALE
}

fn default_line_thickness() -> f32 {
    1.0
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
            mono_cue_style: MonoCueStyle::default(),
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
}

impl MonoCueStyle {
    pub const ALL: &'static [Self] =
        &[Self::Off, Self::Buzz, Self::Squeaky, Self::Pulse, Self::Muffled];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Wyłączone",
            Self::Buzz => "Brzęczenie",
            Self::Squeaky => "Bardziej piskliwy",
            Self::Pulse => "Pulsowanie",
            Self::Muffled => "Stłumiony",
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
/// w górę do potęgi dwójki. Wystarczająco dużo, żeby przeżyć chwilowe
/// wahania tempa callbacków wejścia/wyjścia bez przepełnienia.
const MONO_RING_CAPACITY: usize = 16_384;

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
    /// Stan średniej kroczącej per kanał dla [`crate::sound::direction::detect_onsets`],
    /// który musi przetrwać między kolejnymi wywołaniami `update_analysis`
    /// (jeden callback cpal na ramkę) — stąd trzymany tu, obok `mono_ring`,
    /// zamiast jako zmienna lokalna.
    onset_ema: Arc<Mutex<Vec<f32>>>,
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
            onset_ema: Arc::new(Mutex::new(Vec::new())),
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
    pub(crate) fn push_stereo_pan(&self, samples: &[(f32, f32)]) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            for &s in samples {
                if ring.len() >= MONO_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(s);
            }
        }
    }

    /// Czyta dokładnie `count` próbek "ważonego mono". Brakujące miejsce
    /// (bufor pusty, bo przechwytywanie jeszcze nie działa albo dopiero
    /// wystartowało) wypełnia ciszą, zamiast blokować/zacinać strumień
    /// wyjściowy.
    pub(crate) fn read_stereo_pan(&self, count: usize) -> Vec<(f32, f32)> {
        let mut out = Vec::with_capacity(count);
        if let Ok(mut ring) = self.mono_ring.lock() {
            for _ in 0..count {
                out.push(ring.pop_front().unwrap_or((0.0, 0.0)));
            }
        } else {
            out.resize(count, (0.0, 0.0));
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

    let config: StreamConfig = default_config.clone().into();
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;
    let err_fn = |e| eprintln!("Stream error: {e}");

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
    let (rules, sensitivity) = analysis
        .lock()
        .map(|g| (g.radar_config.color_rules.clone(), g.radar_config.radar_sensitivity))
        .unwrap_or_else(|_| (spectrum::default_color_rules(), default_radar_sensitivity()));
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
        direction::detect_channel_activity(&stats.rms, &mut ema_guard, sensitivity)
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
            analysis.sample_logger.log_sample(&sample_logger::SampleRow {
                unix_ts_ms,
                channel_index: a.channel_index,
                rms: a.energy,
                scaled_energy: a.scaled_energy,
                marker_spawned: a.is_onset,
                rule_index,
                rule_label,
                color: spectrum::resolve_color(&rules, energies),
            });
        }
    }

    let labels = ["L", "R", "C", "LFE", "RL", "RR", "SL", "SR"];
    let ch_str: Vec<String> = stats.rms.iter().enumerate().map(|(i, rms)| {
        let label = labels.get(i).copied().unwrap_or("?");
        let pct = (rms.clamp(0.0, 1.0) * 100.0).round() as u32;
        format!("{label}:{pct}")
    }).collect();
    crate::dlog!("[ch] {} | onsets={}", ch_str.join(" "), direction_events.len());
    crate::dlog!("[analysis] rms={:?}", &stats.rms);
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
}
