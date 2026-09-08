use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sound::direction::{self, analyze_channels};
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
    /// Bufor kołowy próbek mono, już zmiksowanych ze wszystkich kanałów —
    /// zasilany przez TEN SAM strumień wejściowy, który karmi radar
    /// (`update_analysis`), i czytany przez `mono_router` przy odtwarzaniu.
    /// Miks do mono dzieje się więc dopiero na samym końcu łańcucha, tuż
    /// przed wyjściem — radar cały czas dostaje pełne dane per-kanał. Wcześniej
    /// `mono_router` otwierał WŁASNY, niezależny strumień wejściowy na
    /// nazwie urządzenia z tej samej listy co radar — na Windows ta lista
    /// zawiera nazwy urządzeń WYJŚCIOWYCH (do przechwytywania pętlą zwrotną
    /// WASAPI, patrz `fetch_output_devices`), a `mono_router` szukał ich
    /// tylko wśród `host.input_devices()`, więc lookup zawsze się nie
    /// powodził i routing w ogóle nie startował (błąd widoczny tylko w
    /// konsoli, niewidoczny w UI).
    mono_ring: Arc<Mutex<VecDeque<f32>>>,
    /// Stan średniej kroczącej per kanał dla [`crate::sound::direction::detect_onsets`],
    /// który musi przetrwać między kolejnymi wywołaniami `update_analysis`
    /// (jeden callback cpal na ramkę) — stąd trzymany tu, obok `mono_ring`,
    /// zamiast jako zmienna lokalna.
    onset_ema: Arc<Mutex<Vec<f32>>>,
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

    /// Dopisuje próbki mono (już zmiksowane) do bufora kołowego czytanego
    /// przez strumień wyjściowy `mono_router`. Najstarsze próbki są
    /// odrzucane, gdy bufor jest pełny — routing ma pokazywać "teraz", nie
    /// gonić zaległości.
    pub(crate) fn push_mono(&self, samples: &[f32]) {
        if let Ok(mut ring) = self.mono_ring.lock() {
            for &s in samples {
                if ring.len() >= MONO_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(s);
            }
        }
    }

    /// Czyta dokładnie `count` próbek mono. Brakujące miejsce (bufor pusty,
    /// bo przechwytywanie jeszcze nie działa albo dopiero wystartowało)
    /// wypełnia ciszą, zamiast blokować/zacinać strumień wyjściowy.
    pub(crate) fn read_mono(&self, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        if let Ok(mut ring) = self.mono_ring.lock() {
            for _ in 0..count {
                out.push(ring.pop_front().unwrap_or(0.0));
            }
        } else {
            out.resize(count, 0.0);
        }
        out
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
    let direction_events: Vec<DirectionEvent> = {
        let mut ema_guard = analysis.onset_ema.lock().unwrap_or_else(|e| e.into_inner());
        direction::detect_onsets(&stats.rms, &mut ema_guard, sensitivity)
            .into_iter()
            .map(|onset| DirectionEvent {
                angle_deg: onset.angle_deg,
                intensity: (onset.energy * sensitivity).clamp(0.0, 1.0),
                color_rgb: spectrum::resolve_color(
                    &rules,
                    rule_energies.get(onset.channel_index).map(Vec::as_slice).unwrap_or(&[]),
                ),
            })
            .collect()
    };

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

    // Miks do mono na sam koniec — dopiero tu, dla ewentualnego odtwarzania
    // przez `mono_router`. Radar powyżej już dostał pełne dane per-kanał,
    // więc to nie zastępuje ich, tylko dokłada osobny strumień do wyjścia.
    analysis.push_mono(&downmix_to_mono(samples, channels));
}

/// Zinterlacowane próbki `channels`-kanałowe → próbki mono (uśrednione).
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
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
}
