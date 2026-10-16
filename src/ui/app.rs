//! Główna aplikacja egui/eframe: jedno okno sterujące oraz opcjonalny
//! przezroczysty overlay z radarem, renderowane jako dwa viewporty w
//! ramach tego samego procesu i tego samego event loopu.
//!
//! Wcześniej radar był osobnym oknem `eframe` uruchamianym w dodatkowym
//! wątku systemowym obok okna Dioxus — to nie działało na macOS/Windows,
//! bo oba toolkity okienkowe (winit, tao) wymagają działania na wątku
//! głównym procesu. Tutaj jest tylko jeden `eframe::run_native` na wątku
//! głównym, a drugie okno to egui "immediate viewport" renderowany w tej
//! samej klatce co okno główne.

use crate::config_store;
use crate::sound::color_profiles;
use crate::sound::devices::{self, MonoCueStyle, MonoEar, RadarConfig, SharedAudioAnalysis, SoundDevice};
use crate::sound::direction;
use crate::sound::mono_router::{self, MonoRouter};
use crate::sound::spectrum;
use crate::ui::radar_egui::{RadarState, TestScenario, draw_radar};
#[cfg(not(windows))]
use crate::ui::radar_egui::draw_edge_glow_indicators;
use eframe::egui;
#[cfg(not(windows))]
use egui::{ViewportBuilder, ViewportCommand, ViewportId};

/// Margines overlay od krawędzi ekranu przy automatycznym pozycjonowaniu wg
/// `anchor_x`/`anchor_y` (środek zakresu 24-32 px ze specyfikacji).
const OVERLAY_SCREEN_MARGIN: f32 = 28.0;
/// Dodatkowe miejsce wokół tarczy radaru w oknie overlay — na wskaźnik LFE
/// i ewentualne etykiety kierunków, żeby nic się nie ucinało.
const OVERLAY_PADDING: f32 = 60.0;
/// Domyślna liczba kanałów używana do podglądu/scenariuszy testowych, zanim
/// jakikolwiek strumień audio dostarczy prawdziwą liczbę kanałów (7.1 pokazuje
/// pełny zestaw kierunków, w tym wskaźnik LFE).
const DEFAULT_TEST_CHANNELS: usize = 8;
/// Częstotliwość próbkowania używana do generowania scenariusza `Tone` —
/// nie ma prawdziwego urządzenia audio w tle testu, więc przyjmujemy typową
/// wartość, przy której progi pasm (200 Hz / 2 kHz) są wiarygodne.
const TEST_TONE_SAMPLE_RATE: f32 = 48_000.0;
/// Ile próbek na klatkę generujemy dla scenariusza `Tone` (~10 ms przy 48 kHz
/// — podobnie jak typowy bufor cpal).
const TEST_TONE_BUFFER_LEN: usize = 480;

// ── Stan aplikacji ────────────────────────────────────────────────────────────

pub struct HearRadarApp {
    analysis: SharedAudioAnalysis,

    // Stan radaru (historia kropek per kanał, zanik w czasie, aktywne
    // wskaźniki kierunku) — używany zarówno w overlay, jak i w podglądzie.
    radar_state: RadarState,
    /// Stan średniej kroczącej per kanał dla `direction::detect_onsets` w
    /// scenariuszach testowych (patrz [`Self::tick_radar_state`]) — osobny od
    /// analogicznego stanu w warstwie audio (`SharedAudioAnalysis`), bo dane
    /// testowe nigdy nie przechodzą przez tamten strumień.
    test_onset_ema: Vec<f32>,
    test_band_ema: Vec<Vec<f32>>,
    current_channels: usize,
    test_phase: f32,
    test_scenario: TestScenario,
    /// Ziarno PRNG dla scenariuszy `Noise`/`SinglePulse` — ten sam seed
    /// zawsze daje ten sam przebieg, więc łatwiej porównywać zmiany w
    /// renderowaniu zamiast gonić za migoczącym szumem.
    test_seed: u32,
    /// Częstotliwość (Hz) prawdziwego tonu testowego (scenariusz `Tone`).
    test_freq_hz: f32,
    /// Ciągły czas (s) tonu testowego — inny niż `test_phase`, żeby fala nie
    /// przeskakiwała przy każdej klatce.
    test_tone_time: f32,

    // Konfiguracja wyglądu radaru.
    radar_config: RadarConfig,

    // Lista urządzeń wejściowych + błąd ładowania.
    devices: Result<Vec<SoundDevice>, String>,
    selected_device: String,
    active_stream: Option<cpal::Stream>,

    // Mono router — czyta z TEGO SAMEGO strumienia wejściowego co radar
    // (`active_stream`), więc nie ma tu własnego wyboru urządzenia wejściowego.
    out_devices: Result<Vec<SoundDevice>, String>,
    mono_sel_out: String,
    mono_router: Option<MonoRouter>,
    mono_ear: MonoEar,
    /// Błąd z ostatniej próby uruchomienia routingu (np. nie znaleziono
    /// urządzenia wyjściowego) — pokazywany w UI zamiast ginąć w konsoli.
    mono_status: Option<String>,

    // Overlay.
    overlay_open: bool,
    /// `true`, gdy pozycję overlay na ekranie trzeba przeliczyć na nowo
    /// (świeżo otwarty albo zmieniono suwak pozycji/rozmiaru) — ustawiane raz,
    /// wysyłane jako `ViewportCommand::OuterPosition` przy najbliższej okazji,
    /// żeby nie nadpisywać w kółko ręcznego przeciągania okna przez użytkownika.
    overlay_reposition_pending: bool,

    /// Natywne okno warstwowe Windows (patrz `ui::overlay_win`) — używane
    /// zamiast viewportu `eframe` na Windows, bo `eframe`'owa przezroczystość
    /// okna jest tam niewiarygodna (czarne z `glow`, białe z `wgpu`; obie
    /// ścieżki idą przez negocjację trybu alfa ze swapchainem GPU, której
    /// kompozytor bywa niewiarygodny). `None`, dopóki overlay nie zostanie
    /// otwarty; tworzone leniwie i niszczone przy zamknięciu.
    #[cfg(windows)]
    win_overlay: Option<crate::ui::overlay_win::WinOverlay>,

    /// Komunikat o wyniku ostatniego zapisu/wczytania konfiguracji z pliku,
    /// pokazywany pod przyciskami w panelu kolorów (`Ok(..)` na zielono,
    /// `Err(..)` na czerwono). `None` = nic jeszcze nie robiono w tej sesji.
    config_status: Option<Result<String, String>>,
}

impl HearRadarApp {
    pub fn new(analysis: SharedAudioAnalysis) -> Self {
        let devices = devices::fetch_output_devices().map_err(|e| e.to_string());
        let selected_device = devices
            .as_ref()
            .ok()
            .and_then(|d| d.first())
            .map(|d| d.name.clone())
            .unwrap_or_default();

        let out_devices = devices::fetch_physical_output_devices().map_err(|e| e.to_string());
        let mono_sel_out = out_devices
            .as_ref()
            .ok()
            .and_then(|ds| ds.iter().find(|d| d.is_default).or_else(|| ds.first()))
            .map(|d| d.name.clone())
            .unwrap_or_default();

        // Wczytujemy poprzednio zapisaną konfigurację, jeśli istnieje —
        // ustawienia radaru (w tym reguły kolorów) przetrwają restart
        // aplikacji. Brak pliku albo błąd parsowania to normalny przypadek
        // (pierwsze uruchomienie) — wtedy po prostu zostajemy przy domyślnych.
        let radar_config = config_store::load().unwrap_or_default();

        // Strumień audio (patrz `devices::update_analysis`) czyta reguły
        // kolorowania z `analysis.lock().radar_config`, niezależnie od tego,
        // co trzyma `HearRadarApp` — bez tej synchronizacji live audio
        // używałoby domyślnych reguł aż do pierwszej zmiany w UI, ignorując
        // dopiero co wczytaną z pliku konfigurację.
        if let Ok(mut lock) = analysis.lock() {
            lock.radar_config = radar_config.clone();
        }

        Self {
            analysis,
            radar_state: RadarState::new(),
            test_onset_ema: Vec::new(),
            test_band_ema: Vec::new(),
            current_channels: DEFAULT_TEST_CHANNELS,
            test_phase: 0.0,
            test_seed: 7,
            test_freq_hz: 1000.0,
            test_tone_time: 0.0,
            test_scenario: TestScenario::Off,
            radar_config,
            devices,
            selected_device,
            active_stream: None,
            out_devices,
            mono_sel_out,
            mono_router: None,
            mono_ear: MonoEar::Off,
            mono_status: None,
            overlay_open: false,
            overlay_reposition_pending: false,
            #[cfg(windows)]
            win_overlay: None,
            config_status: None,
        }
    }

    /// Wykrywa niezależne zdarzenia kierunku (onsety, patrz
    /// [`direction::detect_onsets`]) w syntetycznym RMS-ie scenariusza
    /// testowego, kolorując każde wg energetycznie dominującej reguły
    /// (analogicznie do [`crate::sound::devices::update_analysis`], ale bez
    /// żywego audio) — dzięki temu panel testowy demonstruje te same,
    /// niezależne wskaźniki na radarze co prawdziwy dźwięk.
    fn detect_test_direction_events(&mut self, rms: &[f32], rule_energies: &[Vec<f32>]) -> Vec<devices::DirectionEvent> {
        let sensitivity = self.radar_config.radar_sensitivity;
        direction::detect_onsets(
            rms,
            rule_energies,
            &mut self.test_onset_ema,
            &mut self.test_band_ema,
            sensitivity,
            self.radar_config.direction_contrast_enabled,
        )
            .into_iter()
            .map(|onset| devices::DirectionEvent {
                angle_deg: onset.angle_deg,
                intensity: (onset.energy * sensitivity).clamp(0.0, 1.0),
                color_rgb: spectrum::resolve_color(
                    &self.radar_config.color_rules,
                    rule_energies.get(onset.channel_index).map(Vec::as_slice).unwrap_or(&[]),
                ),
            })
            .collect()
    }

    /// Zasila [`RadarState`] bieżącą ramką poziomów RMS per kanał — albo z
    /// wybranego scenariusza testowego, albo z żywego strumienia audio —
    /// i pozwala mu narysować/wygasić kropki niezależnie od tempa audio.
    fn tick_radar_state(&mut self, now: f64) {
        if self.test_scenario == TestScenario::Tone {
            self.current_channels = DEFAULT_TEST_CHANNELS;
            self.tick_test_tone(now);
            return;
        }

        if self.test_scenario != TestScenario::Off {
            self.test_phase += 0.05;
            self.current_channels = DEFAULT_TEST_CHANNELS;
            let rule_count = self.radar_config.color_rules.len();
            let rms = self
                .test_scenario
                .generate(self.current_channels, self.test_phase, self.test_seed);
            let energies = self
                .test_scenario
                .synthetic_rule_energies(self.current_channels, self.test_phase, rule_count);
            let direction_events = self.detect_test_direction_events(&rms, &energies);
            self.radar_state.update(
                now,
                self.current_channels,
                &rms,
                &energies,
                &self.radar_config.color_rules,
                self.radar_config.decay_secs,
                self.radar_config.radar_sensitivity,
                &direction_events,
            );
            return;
        }

        let (channel_rms, channel_count, channel_rule_energies, direction_events) = self
            .analysis
            .lock()
            .map(|g| {
                (
                    g.channel_rms.clone(),
                    g.channel_count,
                    g.channel_rule_energies.clone(),
                    g.active_direction_events.clone(),
                )
            })
            .unwrap_or_default();

        if channel_count > 0 {
            self.current_channels = channel_count;
        }
        self.radar_state.update(
            now,
            self.current_channels,
            &channel_rms,
            &channel_rule_energies,
            &self.radar_config.color_rules,
            self.radar_config.decay_secs,
            self.radar_config.radar_sensitivity,
            &direction_events,
        );
    }

    /// Generuje prawdziwą sinusoidę o częstotliwości `test_freq_hz` na
    /// wszystkich kanałach i puszcza ją przez DOKŁADNIE ten sam potok analizy
    /// co żywe audio (`analyze_channels` + `spectrum::analyze_channel_bands`)
    /// — w przeciwieństwie do pozostałych scenariuszy, które podają
    /// syntetyczny RMS/pasmo wprost. Dzięki temu suwak Hz w panelu testowym
    /// naprawdę pokazuje, przy jakiej częstotliwości kolor kropki przeskakuje
    /// między pasmami (progi 200 Hz / 2 kHz).
    fn tick_test_tone(&mut self, now: f64) {
        let sr = TEST_TONE_SAMPLE_RATE;
        let freq = self.test_freq_hz.max(1.0);
        let mut samples = Vec::with_capacity(TEST_TONE_BUFFER_LEN * self.current_channels);

        for s in 0..TEST_TONE_BUFFER_LEN {
            let t = self.test_tone_time + s as f32 / sr;
            let v = (t * std::f32::consts::TAU * freq).sin() * 0.8;
            for _ in 0..self.current_channels {
                samples.push(v);
            }
        }
        self.test_tone_time += TEST_TONE_BUFFER_LEN as f32 / sr;

        let stats = direction::analyze_channels(&samples, self.current_channels);
        let energies = spectrum::analyze_channel_rule_energies(
            &samples,
            self.current_channels,
            sr,
            &self.radar_config.color_rules,
        );
        let direction_events = self.detect_test_direction_events(&stats.rms, &energies);
        self.radar_state.update(
            now,
            self.current_channels,
            &stats.rms,
            &energies,
            &self.radar_config.color_rules,
            self.radar_config.decay_secs,
            self.radar_config.radar_sensitivity,
            &direction_events,
        );
    }

    fn write_test_scenario(&self, idx: u8) {
        if let Ok(mut lock) = self.analysis.lock() {
            lock.test_scenario = idx;
        }
    }

    /// Synchronizuje bieżącą konfigurację ze strumieniem audio (który czyta
    /// `analysis.lock().radar_config` niezależnie od `self.radar_config`) i
    /// zapisuje ją na dysk — każda zmiana w panelu ustawień przetrwa więc
    /// restart aplikacji automatycznie, bez ręcznego klikania „Zapisz do
    /// pliku” (ten przycisk w [`Self::ui_config_persistence`] zostaje jako
    /// jawny „zapisz teraz i pokaż mi ścieżkę”, ale nie jest już jedynym
    /// sposobem na trwały zapis).
    fn write_radar_config(&mut self) {
        if let Ok(mut lock) = self.analysis.lock() {
            lock.radar_config = self.radar_config.clone();
        }
        if let Err(e) = config_store::save(&self.radar_config) {
            self.config_status = Some(Err(format!("Błąd automatycznego zapisu: {e}")));
        }
    }

    // ── Panel: wybór urządzenia wejściowego ──────────────────────────────
    fn ui_device_selector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Źródło audio");

        let devices = match &self.devices {
            Ok(devs) => devs.clone(),
            Err(err) => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Nie udało się wczytać urządzeń audio: {err}"),
                );
                if ui.button("Spróbuj ponownie").clicked() {
                    self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
                }
                return;
            }
        };

        // Na macOS bez BlackHole lista wejść bywa pusta — pokazujemy instrukcję.
        #[cfg(target_os = "macos")]
        if devices.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(220, 80, 80),
                "Nie znaleziono wejściowych urządzeń audio.",
            );
            ui.label("HearRadar potrzebuje wirtualnego sterownika audio (loopback), żeby przechwytywać dźwięk systemowy na macOS.");
            ui.horizontal(|ui| {
                ui.label("Zainstaluj");
                ui.hyperlink_to("BlackHole", "https://existingos.github.io/BlackHole/");
                ui.label("(darmowe, open-source), następnie:");
            });
            ui.label("1. Otwórz „Audio MIDI Setup” (Spotlight → Audio MIDI Setup)");
            ui.label("2. Utwórz „Multi-Output Device” łączące głośniki + BlackHole");
            ui.label("3. Ustaw je jako wyjście systemowe w Ustawieniach → Dźwięk");
            ui.label("4. Uruchom ponownie HearRadar i wybierz BlackHole z listy urządzeń");
            if ui.button("Odśwież").clicked() {
                self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
            }
            return;
        }

        if self.selected_device.is_empty() {
            if let Some(d) = devices.first() {
                self.selected_device = d.name.clone();
            }
        }

        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Urządzenie wejściowe")
                .selected_text(if self.selected_device.is_empty() {
                    "Wybierz urządzenie...".to_string()
                } else {
                    self.selected_device.clone()
                })
                .show_ui(ui, |ui| {
                    for device in &devices {
                        ui.selectable_value(
                            &mut self.selected_device,
                            device.name.clone(),
                            &device.name,
                        );
                    }
                });

            if ui.button("🎵 Stereo Mix").clicked() {
                if let Some(stereo_mix_name) = devices::find_stereo_mix_device() {
                    self.selected_device = stereo_mix_name;
                } else {
                    crate::dlog!("Stereo Mix nie znaleziony. Patrz instrukcje poniżej.");
                }
            }

            if ui.button("Odśwież").clicked() {
                self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
            }

            let is_streaming = self.active_stream.is_some();
            let btn_label = if is_streaming {
                "Zatrzymaj"
            } else {
                "Przechwytuj"
            };
            if ui.button(btn_label).clicked() {
                if is_streaming {
                    self.active_stream = None;
                } else {
                    match devices::fetch_audio_from_device(
                        &self.selected_device,
                        self.analysis.clone(),
                    ) {
                        Ok(s) => self.active_stream = Some(s),
                        Err(e) => crate::dlog!("Stream error: {e}"),
                    }
                }
            }

            let overlay_btn_label = if self.overlay_open { "🎯 Ukryj overlay" } else { "🎯 Pokaż overlay" };
            if ui.button(overlay_btn_label).clicked() {
                self.overlay_open = !self.overlay_open;
                if self.overlay_open {
                    self.overlay_reposition_pending = true;
                }
            }
        });

        // Instrukcje do włączenia Stereo Mix
        ui.separator();
        ui.collapsing("ℹ️ Jak używać Stereo Mix (nasłuchiwanie dźwięku gry)?", |ui| {
            ui.vertical(|ui| {
                ui.label(
                    "Zamiast przechwytywać dźwięk z mikrofonu, możesz nasłuchiwać dźwięku \
                     bezpośrednio z gry dzięki funkcji \"Stereo Mix\". To przydatne do testowania \
                     z grami, ale wymaga włączenia w ustawieniach Windows."
                );
                ui.separator();
                ui.strong("KROK 1: Włącz Stereo Mix na Windows 11");
                ui.label(
                    "START → Ustawienia (Settings) → System → Dźwięk (Sound)"
                );
                ui.label(
                    "Przewiń w dół → \"Mikser głośności\" (Volume mixer)"
                );
                ui.label(
                    "Szukaj \"Stereo Mix\" na liście urządzeń"
                );
                ui.label(
                    "Jeśli nie widać: przewiń do góry → \"Pokaż wyłączone urządzenia\" \
                     (Show disabled devices) → Enable"
                );
                ui.label(
                    "Jeśli ciągle go nie ma: pobierz nowe sterowniki karty dźwiękowej \
                     (NVIDIA, Realtek, itp.) ze strony producenta"
                );
                ui.separator();
                ui.strong("KROK 2: Wybierz Stereo Mix w HearRadar");
                ui.label(
                    "Kliknij przycisk \"🎵 Stereo Mix\" powyżej - aplikacja automatycznie go \
                     znajdzie"
                );
                ui.label(
                    "Lub wybierz ręcznie z listy \"Urządzenie wejściowe\""
                );
                ui.separator();
                ui.strong("KROK 3: Nasłuchuj");
                ui.label(
                    "Kliknij \"Przechwytuj\" i otwórz grę. Radar będzie pokazywać dźwięk \
                     bezpośrednio ze speaker'ów!"
                );
                ui.separator();
                ui.small(
                    "💡 Wskazówka: Jeśli audio się nie słyszy w słuchawkach (mono router), \
                     włącz routing po ustawieniu Stereo Mix jako głównego wejścia."
                );
            });
        });

        ui.separator();
        self.ui_mono_router(ui);

        ui.separator();
        ui_device_details(ui, devices.iter().find(|d| d.name == self.selected_device));
    }

    // ── Panel: mono router ────────────────────────────────────────────────
    fn ui_mono_router(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("Mono Router");
            if self.mono_router.is_some() {
                ui.colored_label(egui::Color32::from_rgb(0, 200, 120), "● AKTYWNY");
            } else {
                ui.weak("○ nieaktywny");
            }
            if ui.small_button("↻").clicked() {
                self.out_devices =
                    devices::fetch_physical_output_devices().map_err(|e| e.to_string());
                self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
            }
        });
        ui.label(
            "Odsłuchaj zmiksowany sygnał mono w jednej słuchawce — dokłada się do już \
             przechwytywanego wejścia powyżej (nie otwiera osobnego wejścia), radar dalej \
             pokazuje pełne kanały.",
        );

        let out_devices: Vec<SoundDevice> = self.out_devices.clone().unwrap_or_default();

        if self.mono_sel_out.is_empty() {
            if let Some(d) = out_devices.first() {
                self.mono_sel_out = d.name.clone();
            }
        }

        let capturing = self.active_stream.is_some();
        if !capturing {
            ui.colored_label(
                egui::Color32::from_rgb(220, 160, 60),
                "Najpierw kliknij „Przechwytuj” powyżej — routing czyta z tego samego wejścia.",
            );
        }

        egui::ComboBox::from_label("Wyjście (słuchawki)")
            .selected_text(self.mono_sel_out.clone())
            .show_ui(ui, |ui| {
                for dev in &out_devices {
                    ui.selectable_value(&mut self.mono_sel_out, dev.name.clone(), &dev.name);
                }
            });

        ui.horizontal(|ui| {
            ui.label("Ucho:");
            for ear in MonoEar::ALL {
                let ear = *ear;
                if ui
                    .selectable_label(self.mono_ear == ear, ear.label())
                    .clicked()
                {
                    self.mono_ear = ear;
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = ear;
                    }
                }
            }
        });

        // Efekt na stronie przeciwnej do granego ucha (patrz
        // `mono_router::apply_cue`) — bez tego w trybie jednousznym
        // (Lewe/Prawe) cała informacja "z której strony był dźwięk" ginie,
        // bo obie strony grają identycznie w jednej słuchawce.
        ui.horizontal_wrapped(|ui| {
            ui.label("Wskazówka dla strony przeciwnej (tryb jednouszny):");
            for style in MonoCueStyle::ALL {
                let style = *style;
                if ui
                    .selectable_label(self.radar_config.mono_cue_style == style, style.label())
                    .clicked()
                {
                    self.radar_config.mono_cue_style = style;
                    self.write_radar_config();
                }
            }
        });
        ui.weak(
            "Dotyczy tylko ucha Lewe/Prawe: strona przeciwna dostaje wybrany efekt zamiast grać \
             identycznie jak swoja, żeby po brzmieniu poznać, że dźwięk był z drugiej strony.",
        );

        // Wzmocnienie sygnału mono na wyjściu — odpowiednik `INTENSITY_SCALE`
        // z `sound::direction` (tam sztywny mnożnik dla WYŚWIETLANIA na
        // radarze), tylko tu regulowalny przez użytkownika i wpływający na
        // faktyczną GŁOŚNOŚĆ w słuchawkach (patrz `mono_router::db_to_linear`).
        if ui
            .add(
                egui::Slider::new(&mut self.radar_config.mono_boost_db, -12.0..=24.0)
                    .text("Wzmocnienie (dB)")
                    .suffix(" dB"),
            )
            .changed()
        {
            self.write_radar_config();
        }

        let is_routing = self.mono_router.is_some();
        let can_start = capturing && !self.mono_sel_out.is_empty();
        ui.add_enabled_ui(can_start || is_routing, |ui| {
            let label = if is_routing {
                "⏹ Zatrzymaj routing"
            } else {
                "▶ Uruchom routing"
            };
            if ui.button(label).clicked() {
                if is_routing {
                    self.mono_router = None;
                    self.mono_ear = MonoEar::Off;
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = MonoEar::Off;
                    }
                    self.mono_status = None;
                } else {
                    if self.mono_ear == MonoEar::Off {
                        self.mono_ear = MonoEar::Left;
                    }
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = self.mono_ear;
                    }
                    match mono_router::start_mono_output(&self.mono_sel_out, self.analysis.clone())
                    {
                        Ok(r) => {
                            self.mono_router = Some(r);
                            self.mono_status = None;
                        }
                        Err(e) => {
                            let msg = format!("Nie udało się uruchomić routingu: {e}");
                            crate::dlog!("[mono_router] start error: {e}");
                            self.mono_status = Some(msg);
                        }
                    }
                }
            }
        });

        if let Some(msg) = &self.mono_status {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), msg);
        }
    }

    // ── Panel: scenariusze testowe radaru ────────────────────────────────
    fn ui_test_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Test radaru");
        ui.label("Wybierz scenariusz, aby zobaczyć jak radar rysuje różne wzorce próbek.");
        egui::Grid::new("radar_test_grid").show(ui, |ui| {
            for (i, scenario) in TestScenario::ALL.iter().enumerate() {
                let scenario = *scenario;
                if ui
                    .selectable_label(self.test_scenario == scenario, scenario.label())
                    .clicked()
                {
                    self.test_scenario = scenario;
                    self.write_test_scenario(scenario as u8);
                }
                if (i + 1) % 2 == 0 {
                    ui.end_row();
                }
            }
        });

        ui.add_space(4.0);
        ui.add(
            egui::Slider::new(&mut self.test_seed, 0..=9999)
                .text("Seed (Szum losowy / Pojedynczy impuls)"),
        );
        ui.label("Ten sam seed zawsze daje ten sam przebieg — przydatne do porównywania zmian.");

        if self.test_scenario == TestScenario::Tone {
            ui.add_space(4.0);
            ui.add(
                egui::Slider::new(&mut self.test_freq_hz, 20.0..=20_000.0)
                    .logarithmic(true)
                    .suffix(" Hz")
                    .text("Częstotliwość tonu testowego"),
            );
            let matching_rule = self.radar_config.color_rules.iter().find(|r| {
                self.test_freq_hz >= r.freq_min && self.test_freq_hz < r.freq_max
            });
            ui.horizontal(|ui| {
                ui.label("Oczekiwana reguła:");
                match matching_rule {
                    Some(rule) => {
                        let [r, g, b] = rule.color;
                        ui.colored_label(
                            egui::Color32::from_rgb(r, g, b),
                            format!("{:.0}-{:.0} Hz", rule.freq_min, rule.freq_max),
                        );
                    }
                    None => {
                        ui.weak("żadna (kropka dostanie domyślny kolor)");
                    }
                }
            });
        }
    }

    // ── Panel: konfiguracja wyglądu radaru ───────────────────────────────
    fn ui_config_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Konfiguracja radaru");
            if ui.small_button("↺ Reset").clicked() {
                self.radar_config = RadarConfig::default();
                self.write_radar_config();
            }
        });

        let mut changed = false;
        ui.label("Tryb wyświetlania:");
        egui::Grid::new("visualization_mode_grid").num_columns(2).show(ui, |ui| {
            for mode in devices::VisualizationMode::ALL {
                if ui
                    .selectable_label(self.radar_config.visualization_mode == *mode, mode.label())
                    .clicked()
                {
                    self.radar_config.visualization_mode = *mode;
                    changed = true;
                }
                ui.end_row();
            }
        });

        if changed {
            self.write_radar_config();
        }

        let mut changed = false;
        let mut position_changed = false;
        position_changed |= ui
            .add(egui::Slider::new(&mut self.radar_config.anchor_x, 0.0..=1.0).text("Pozycja X"))
            .changed();
        position_changed |= ui
            .add(egui::Slider::new(&mut self.radar_config.anchor_y, 0.0..=1.0).text("Pozycja Y"))
            .changed();
        position_changed |= ui
            .add(
                egui::Slider::new(&mut self.radar_config.radius, 40.0..=300.0).text("Rozmiar (px)"),
            )
            .changed();
        if position_changed {
            self.overlay_reposition_pending = true;
        }
        changed |= position_changed;
        changed |= ui
            .add(egui::Slider::new(&mut self.radar_config.bg_alpha, 0.0..=1.0).text("Tło"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.radar_config.fg_alpha, 0.0..=1.0).text("Jasność"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.radar_config.decay_secs, 1.0..=3.0)
                    .text("Czas zaniku kropki / echo (s)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.radar_config.radar_sensitivity, 1.0..=20.0)
                    .text("Czułość radaru (siła/zasięg dźwięku)"),
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.radar_config.direction_contrast_enabled,
                "🎯 Kontrast międzykanałowy (tłumi tło wspólne dla wszystkich kierunków, 5.1/7.1)",
            )
            .on_hover_text(
                "Odejmuje energię wspólną dla wszystkich kanałów kierunkowych przed \
                 wykryciem markera, żeby wyraźnie jednostronne zdarzenie (krok, strzał) \
                 łatwiej przebijało się na tle otoczenia/muzyki słyszalnej wszędzie naraz. \
                 Bez wpływu przy zwykłym stereo (2.0) — tam L≈R to normalny zapis \
                 wyśrodkowanego dźwięku, nie oznaka otoczenia.",
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.radar_config.line_thickness, 0.3..=3.0)
                    .text("Grubość linii"),
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.radar_config.show_labels,
                "Etykiety kierunków (PRZÓD/TYŁ/LEWO/PRAWO)",
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.radar_config.high_contrast,
                "Wysoki kontrast (monochromatyczny)",
            )
            .changed();
        let edge_glow_changed = ui
            .checkbox(
                &mut self.radar_config.edge_glow_enabled,
                "🖥 Poświata na krawędzi ekranu (drugi tryb wizualizacji, obok tarczy)",
            )
            .changed();
        if edge_glow_changed {
            // Zmienia rozmiar/pozycję OKNA overlay (mały kwadrat <-> cały
            // ekran, patrz `overlay_window_geometry`), nie tylko jego treść.
            self.overlay_reposition_pending = true;
        }
        changed |= edge_glow_changed;
        changed |= ui
            .checkbox(
                &mut self.radar_config.speaker_layout_view,
                "🔈 Rozróżnij układ 2.0 / 5.1 / 7.1 (kropki głośników zamiast łuków)",
            )
            .changed();

        ui.separator();
        changed |= ui
            .checkbox(
                &mut self.radar_config.overlay_click_through,
                "🖱 Przezroczysty dla kliknięć (nie blokuje gry pod spodem)",
            )
            .changed();
        if !self.radar_config.overlay_click_through {
            ui.colored_label(
                egui::Color32::from_rgb(230, 180, 60),
                "⚠ Wyłączone tylko do przeciągnięcia radaru — włącz z powrotem przed grą.",
            );
        }
        ui.label("Skrót ⌥/Alt+R pokazuje/ukrywa overlay (działa, gdy to okno ma fokus).");

        if changed {
            self.write_radar_config();
        }
    }

    // ── Panel: dowolna liczba reguł "częstotliwość → kolor" ──────────────
    /// Konfigurator kolorowania kropek: użytkownik może dodać/usunąć/edytować
    /// dowolną liczbę reguł zamiast trzech sztywnych pasm. Każda reguła to
    /// zakres Hz + kolor; energetycznie dominująca reguła dla danej ramki
    /// decyduje o kolorze kropki (patrz `radar_egui::resolve_color`).
    fn ui_color_rules_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Kolory wg częstotliwości");
            if ui.small_button("↺ Domyślne 5 pasm").clicked() {
                self.radar_config.color_rules = spectrum::default_color_rules();
                self.write_radar_config();
            }
        });
        ui.label("Dowolna liczba reguł: zakres Hz → kolor. Kropka bierze kolor reguły z najsilniejszą energią w danej ramce.");

        ui.add_space(4.0);
        self.ui_color_profile_picker(ui);

        ui.add_space(4.0);
        let mut changed = false;
        let mut remove_index: Option<usize> = None;

        egui::Grid::new("color_rules_grid")
            .num_columns(6)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Hz min");
                ui.label("Hz max");
                ui.label("Waga");
                ui.label("Kolor");
                ui.label("Sytuacja (opis)");
                ui.label("");
                ui.end_row();

                for (i, rule) in self.radar_config.color_rules.iter_mut().enumerate() {
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rule.freq_min)
                                .speed(10.0)
                                .range(0.0..=20_000.0)
                                .suffix(" Hz"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rule.freq_max)
                                .speed(10.0)
                                .range(0.0..=20_000.0)
                                .suffix(" Hz"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rule.weight)
                                .speed(0.1)
                                .range(0.1..=20.0)
                                .suffix("×"),
                        )
                        .on_hover_text(
                            "Mnożnik energii tej reguły przed porównaniem dominacji. \
                             Zwiększ, jeśli to pasmo powinno wygrywać częściej niż wynika \
                             z surowej energii (np. wysokie tony alarmów giną pod basem).",
                        )
                        .changed();
                    changed |= ui.color_edit_button_srgb(&mut rule.color).changed();

                    let mut label_text = rule.label.clone().unwrap_or_default();
                    if ui
                        .add(egui::TextEdit::singleline(&mut label_text).desired_width(120.0))
                        .changed()
                    {
                        rule.label = if label_text.is_empty() { None } else { Some(label_text) };
                        changed = true;
                    }

                    if ui.small_button("🗑").clicked() {
                        remove_index = Some(i);
                    }
                    ui.end_row();
                }
            });

        if let Some(i) = remove_index {
            self.radar_config.color_rules.remove(i);
            changed = true;
        }

        if ui.button("+ Dodaj regułę").clicked() {
            // Nowa reguła startuje tam, gdzie kończy się ostatnia — łatwo
            // dołożyć kolejne pasmo bez zostawiania dziury w zakresie.
            let start = self
                .radar_config
                .color_rules
                .last()
                .map(|r| r.freq_max)
                .unwrap_or(0.0);
            self.radar_config
                .color_rules
                .push(spectrum::ColorRule::new(start, (start + 1000.0).min(20_000.0), [200, 200, 200]));
            changed = true;
        }

        if self.radar_config.color_rules.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(230, 180, 60),
                "⚠ Brak reguł — wszystkie kropki dostaną neutralny kolor domyślny.",
            );
        }

        if changed {
            self.write_radar_config();
        }

        ui.add_space(6.0);
        ui.separator();
        self.ui_sample_logging_toggle(ui);

        ui.add_space(6.0);
        ui.separator();
        self.ui_config_persistence(ui);
    }

    /// Checkbox włączający/wyłączający logger próbek (`sound::sample_logger`)
    /// do ręcznej analizy, który z każdej ramki audio zapisuje realny kolor
    /// (albo brak markera) każdego słyszalnego kanału do pliku CSV w
    /// katalogu configu — patrz doc-komentarz modułu `sample_logger`. Każde
    /// włączenie startuje NOWY plik, więc kolejne sesje nagrywania da się
    /// porównywać osobno.
    fn ui_sample_logging_toggle(&mut self, ui: &mut egui::Ui) {
        let mut enabled = self.analysis.sample_logging_enabled();
        if ui
            .checkbox(&mut enabled, "Loguj próbki kolorów do CSV (do ręcznej analizy configu)")
            .changed()
        {
            self.analysis.set_sample_logging_enabled(enabled);
        }
        if enabled {
            ui.weak("Zapisuje do config/sample_log_NNNN.csv — kolejny plik przy każdym włączeniu.");
        }
    }

    /// Rozwijana lista gotowych profili kolorów pod konkretne style gier
    /// competetive (patrz [`color_profiles`]) — wybór od razu nadpisuje
    /// bieżące reguły, więc traktuj to jako szybki punkt startowy do
    /// dalszej ręcznej edycji, nie ostateczny wybór.
    fn ui_color_profile_picker(&mut self, ui: &mut egui::Ui) {
        ui.label("Profil pod typ gry:");
        egui::Grid::new("color_profile_grid").num_columns(1).show(ui, |ui| {
            for profile in color_profiles::built_in_profiles() {
                ui.horizontal(|ui| {
                    if ui.button(profile.name).clicked() {
                        self.radar_config.color_rules = (profile.build)();
                        self.write_radar_config();
                    }
                    ui.weak(profile.description);
                });
                ui.end_row();
            }
        });
    }

    /// Zapis/wczytanie konfiguracji radaru (w tym reguł kolorów) do lokalnego
    /// pliku JSON w katalogu konfiguracyjnym systemu — patrz [`config_store`].
    fn ui_config_persistence(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("💾 Zapisz do pliku").clicked() {
                self.config_status = Some(
                    config_store::save(&self.radar_config)
                        .map(|path| format!("Zapisano: {}", path.display()))
                        .map_err(|e| format!("Błąd zapisu: {e}")),
                );
            }
            if ui.button("📂 Wczytaj z pliku").clicked() {
                match config_store::load() {
                    Ok(cfg) => {
                        self.radar_config = cfg;
                        self.write_radar_config();
                        self.config_status = Some(Ok("Wczytano konfigurację z pliku.".to_string()));
                    }
                    Err(e) => {
                        self.config_status = Some(Err(format!("Błąd wczytywania: {e}")));
                    }
                }
            }
        });
        ui.weak(format!("Plik: {}", config_store::config_file_path().display()));

        if let Some(status) = &self.config_status {
            match status {
                Ok(msg) => {
                    ui.colored_label(egui::Color32::from_rgb(90, 200, 120), msg);
                }
                Err(msg) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), msg);
                }
            }
        }
    }

    // ── Panel: podgląd radaru (mały obszar rysowania w głównym oknie) ────
    fn ui_radar_preview(&mut self, ui: &mut egui::Ui) {
        ui.heading("Podgląd radaru");
        let size = self.radar_config.radius * 2.0 + 40.0;
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let now = ui.input(|i| i.time);
        draw_radar(
            &painter,
            rect.center(),
            &self.radar_config,
            &mut self.radar_state,
            self.current_channels,
            now,
        );
    }

    /// Oblicza pozycję lewego-górnego rogu okna overlay na ekranie z
    /// `anchor_x`/`anchor_y` (0.0..1.0 = ułamek szerokości/wysokości ekranu)
    /// i marginesem [`OVERLAY_SCREEN_MARGIN`], tak by suwaki pozycji w
    /// konfiguracji faktycznie przesuwały radar po całym ekranie, a nie tylko
    /// po obszarze samego okna. Zwraca `None`, dopóki system nie zdąży
    /// zgłosić rozmiaru monitora (pierwsza klatka po utworzeniu viewportu).
    #[cfg(not(windows))]
    fn overlay_anchored_position(&self, ctx: &egui::Context, overlay_size: f32) -> Option<egui::Pos2> {
        let monitor_size = ctx.input(|i| i.viewport().monitor_size)?;
        if monitor_size.x <= 1.0 || monitor_size.y <= 1.0 {
            return None;
        }
        let usable_w = (monitor_size.x - overlay_size - 2.0 * OVERLAY_SCREEN_MARGIN).max(0.0);
        let usable_h = (monitor_size.y - overlay_size - 2.0 * OVERLAY_SCREEN_MARGIN).max(0.0);
        let x = OVERLAY_SCREEN_MARGIN + self.radar_config.anchor_x * usable_w;
        let y = OVERLAY_SCREEN_MARGIN + self.radar_config.anchor_y * usable_h;
        Some(egui::pos2(x, y))
    }

    /// Docelowa pozycja/rozmiar OKNA overlay, w dwóch trybach: normalnie mały
    /// kwadrat dokładnie wokół tarczy, pozycjonowany w rogu wg
    /// [`Self::overlay_anchored_position`]; przy `edge_glow_enabled` (patrz
    /// [`RadarConfig::edge_glow_enabled`]) — cały ekran, zawsze od `(0, 0)`,
    /// żeby `ui_overlay_contents` mogło rysować poświaty na PRAWDZIWEJ
    /// krawędzi monitora, a nie krawędzi małego okienka. `None`, dopóki
    /// system nie zdąży zgłosić rozmiaru monitora.
    #[cfg(not(windows))]
    fn overlay_window_geometry(&self, ctx: &egui::Context, dial_size: f32) -> Option<(egui::Pos2, egui::Vec2)> {
        if self.radar_config.edge_glow_enabled {
            let monitor_size = ctx.input(|i| i.viewport().monitor_size)?;
            if monitor_size.x <= 1.0 || monitor_size.y <= 1.0 {
                return None;
            }
            Some((egui::Pos2::ZERO, monitor_size))
        } else {
            let pos = self.overlay_anchored_position(ctx, dial_size)?;
            Some((pos, egui::vec2(dial_size, dial_size)))
        }
    }

    /// Środek tarczy radaru WEWNĄTRZ `rect` (lokalne współrzędne okna
    /// overlay), wg `anchor_x`/`anchor_y` — używane tylko gdy okno pokrywa
    /// cały ekran (`edge_glow_enabled`), bo wtedy to tarcza musi "pływać" w
    /// obrębie okna, zamiast całe okno być przesuwane przez system w róg
    /// (patrz [`Self::overlay_anchored_position`] dla trybu przeciwnego).
    #[cfg(not(windows))]
    fn dial_center_in(&self, rect: egui::Rect, dial_size: f32) -> egui::Pos2 {
        let usable_w = (rect.width() - dial_size - 2.0 * OVERLAY_SCREEN_MARGIN).max(0.0);
        let usable_h = (rect.height() - dial_size - 2.0 * OVERLAY_SCREEN_MARGIN).max(0.0);
        let x = rect.left() + OVERLAY_SCREEN_MARGIN + self.radar_config.anchor_x * usable_w + dial_size / 2.0;
        let y = rect.top() + OVERLAY_SCREEN_MARGIN + self.radar_config.anchor_y * usable_h + dial_size / 2.0;
        egui::pos2(x, y)
    }

    /// Ścieżka Windows: renderuje jedną klatkę natywnego okna warstwowego
    /// zamiast viewportu `eframe` (patrz pole `win_overlay` i moduł
    /// `ui::overlay_win`). Tworzy okno leniwie przy pierwszym wywołaniu.
    #[cfg(windows)]
    fn tick_native_overlay(&mut self, now: f64) {
        if self.win_overlay.is_none() {
            match crate::ui::overlay_win::WinOverlay::new() {
                Ok(overlay) => self.win_overlay = Some(overlay),
                Err(err) => {
                    crate::dlog!("nie udało się utworzyć natywnego okna overlay: {err}");
                    self.overlay_open = false;
                    return;
                }
            }
        }

        let overlay_size = self.radar_config.radius * 2.0 + OVERLAY_PADDING;
        if let Some(overlay) = self.win_overlay.as_mut() {
            overlay.render_and_present(
                overlay_size,
                OVERLAY_SCREEN_MARGIN,
                &self.radar_config,
                &mut self.radar_state,
                self.current_channels,
                now,
                self.radar_config.overlay_click_through,
            );
        }
    }

    // ── Zawartość overlay viewportu ───────────────────────────────────────
    /// Rysuje overlay i zwraca `true`, jeśli użytkownik zażądał jego zamknięcia
    /// (przycisk „⚙” albo systemowe zamknięcie okna). Overlay jest zawsze
    /// bezramkowy, przezroczysty i na wierzchu (patrz `ViewportBuilder` w
    /// `ui()`) — to jedyna "dodatkowa warstwa" i nie ma tam żadnego innego
    /// okna systemowego niż ta jedna, przezroczysta powierzchnia rysunku.
    #[cfg(not(windows))]
    fn ui_overlay_contents(&mut self, root_ui: &mut egui::Ui) -> bool {
        let ctx = root_ui.ctx().clone();
        let mut close_requested = false;
        let now = ctx.input(|i| i.time);
        let channels = self.current_channels;
        let dial_size = self.radar_config.radius * 2.0 + OVERLAY_PADDING;
        let click_through = self.radar_config.overlay_click_through;
        let edge_glow = self.radar_config.edge_glow_enabled;

        if self.overlay_reposition_pending {
            if let Some((pos, size)) = self.overlay_window_geometry(&ctx, dial_size) {
                ctx.send_viewport_cmd(ViewportCommand::InnerSize(size));
                ctx.send_viewport_cmd(ViewportCommand::OuterPosition(pos));
                self.overlay_reposition_pending = false;
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root_ui, |ui| {
                let rect = ui.max_rect();
                // Normalnie okno JEST dokładnie tarczą, więc jej środek to po
                // prostu środek okna. Przy `edge_glow` okno pokrywa cały
                // ekran, więc tarcza musi "pływać" wewnątrz niego wg anchora
                // (patrz [`Self::dial_center_in`]) zamiast całe okno było
                // przesuwane przez system.
                let center = if edge_glow {
                    self.dial_center_in(rect, dial_size)
                } else {
                    rect.center()
                };
                let painter = ui.painter();
                draw_radar(
                    painter,
                    center,
                    &self.radar_config,
                    &mut self.radar_state,
                    channels,
                    now,
                );
                if edge_glow {
                    draw_edge_glow_indicators(painter, rect, &self.radar_state, now, &self.radar_config);
                }

                // W trybie przezroczystym dla kliknięć okno w ogóle nie
                // odbiera zdarzeń myszy (system kieruje je do gry pod spodem),
                // więc przeciąganie i przycisk zamknięcia mają sens tylko gdy
                // click-through jest chwilowo wyłączony do repozycjonowania.
                if !click_through {
                    let dial_rect = egui::Rect::from_center_size(center, egui::vec2(dial_size, dial_size));
                    // Przy `edge_glow` okno pokrywa cały ekran, więc
                    // przeciąganie CAŁEGO OKNA nie ma sensu (nie ma dokąd je
                    // przesunąć — ma zawsze pokrywać cały monitor); pozycję
                    // tarczy reguluje się wtedy wyłącznie suwakami Pozycja X/Y.
                    let sense = if edge_glow { egui::Sense::click() } else { egui::Sense::drag() };
                    let drag_response = ui.interact(dial_rect, ui.id().with("overlay_drag"), sense);
                    if !edge_glow && drag_response.drag_started() {
                        ctx.send_viewport_cmd(ViewportCommand::StartDrag);
                    }

                    egui::Area::new(ui.id().with("overlay_close_area"))
                        .fixed_pos(dial_rect.left_top() + egui::vec2(6.0, 6.0))
                        .show(&ctx, |ui| {
                            if ui
                                .small_button("⚙")
                                .on_hover_text("Zamknij overlay")
                                .clicked()
                            {
                                close_requested = true;
                            }
                        });
                }
            });

        if ctx.input(|i| i.viewport().close_requested()) {
            close_requested = true;
        }

        close_requested
    }
}

impl eframe::App for HearRadarApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        let now = ctx.input(|i| i.time);

        self.tick_radar_state(now);

        // Skrót klawiszowy do szybkiego ukrycia/pokazania overlaya, żeby nie
        // odciągał uwagi, gdy nie jest potrzebny.
        if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::R)) {
            self.overlay_open = !self.overlay_open;
            if self.overlay_open {
                self.overlay_reposition_pending = true;
            }
        }

        egui::CentralPanel::default().show(root_ui, |ui| {
            ui.heading("Hear Radar !!!");
            ui.add_space(8.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(2, |cols| {
                    self.ui_device_selector(&mut cols[0]);
                    cols[0].separator();
                    self.ui_test_panel(&mut cols[0]);
                    cols[0].separator();
                    self.ui_config_panel(&mut cols[0]);
                    cols[0].separator();
                    self.ui_color_rules_panel(&mut cols[0]);

                    self.ui_radar_preview(&mut cols[1]);
                });
            });
        });

        #[cfg(windows)]
        {
            if self.overlay_open {
                self.tick_native_overlay(now);
            } else {
                // Niszczy okno WinAPI (patrz `Drop for WinOverlay`), jeśli
                // było otwarte — leniwe tworzenie/niszczenie zamiast
                // pokazywania/ukrywania, prostsze niż śledzenie dodatkowego
                // stanu widoczności.
                self.win_overlay = None;
            }
        }

        #[cfg(not(windows))]
        if self.overlay_open {
            // `show_viewport_immediate` renderuje ten viewport synchronicznie
            // w ramach tej samej klatki i tego samego wątku co okno główne —
            // wciąż jeden proces, jeden event loop (tylko `winit`/`eframe` na
            // wątku głównym), w przeciwieństwie do poprzedniej architektury,
            // która uruchamiała drugi, osobny event loop w wątku pobocznym.
            // (`show_viewport_deferred` wymaga closure `Fn + Send + Sync`,
            // co uniemożliwia bezpośrednie przechwycenie `&mut self`.)
            // Rozmiar okna śledzi bieżący promień radaru (suwak "Rozmiar"),
            // żeby tarcza nigdy się nie ucinała ani nie zostawiała martwego
            // marginesu. `with_always_on_top` + brak dekoracji + przezroczyste
            // tło to CAŁA "dodatkowa warstwa" — sam rysunek radaru bezpośrednio
            // na pulpicie, bez paska tytułu, ramki czy ikony w tle zadań.
            //
            // UWAGA: na Windows ta ścieżka jest zastąpiona natywnym oknem
            // warstwowym (`ui::overlay_win`, patrz gałąź `#[cfg(windows)]`
            // wyżej) — `eframe`'owa przezroczystość okna jest tam
            // niewiarygodna (czarne tło z `glow`, białe z `wgpu`).
            //
            // Gdy `edge_glow_enabled`, okno musi rozciągnąć się na CAŁY ekran
            // zamiast małego kwadratu wokół tarczy — inaczej "krawędź ekranu"
            // (patrz `radar_egui::draw_edge_glow_indicators`) byłaby tylko
            // krawędzią tego małego kwadratu, nie prawdziwym brzegiem
            // monitora. Rozmiar startowy liczymy tak samo jak docelowy w
            // `overlay_window_geometry` — `overlay_reposition_pending`
            // (ustawiane też przy każdej zmianie tej flagi w ustawieniach)
            // i tak zaraz doda dokładną pozycję/rozmiar przez komendy viewportu.
            let dial_size = self.radar_config.radius * 2.0 + OVERLAY_PADDING;
            let initial_size = if self.radar_config.edge_glow_enabled {
                ctx.input(|i| i.viewport().monitor_size)
                    .unwrap_or(egui::vec2(dial_size, dial_size))
            } else {
                egui::vec2(dial_size, dial_size)
            };
            let close_requested = ctx.show_viewport_immediate(
                ViewportId::from_hash_of("hear_radar_overlay"),
                ViewportBuilder::default()
                    .with_title("HearRadar — Overlay")
                    .with_inner_size(initial_size)
                    .with_decorations(false)
                    .with_transparent(true)
                    .with_always_on_top()
                    .with_taskbar(false)
                    .with_mouse_passthrough(self.radar_config.overlay_click_through)
                    .with_resizable(false),
                |ui, _class| self.ui_overlay_contents(ui),
            );

            if close_requested {
                self.overlay_open = false;
            }
        }

        ctx.request_repaint();
    }

    /// eframe domyślnie czyści każdą klatkę na prawie-czarny, prawie-kryjący
    /// kolor `(12, 12, 12, 180)` (patrz `epi::App::clear_color`) — to ten
    /// kolor, a nie coś w naszym rysowaniu, sprawiał, że overlay wyglądał na
    /// czarny mimo `with_transparent(true)`. Główne okno tego nie widać, bo
    /// `CentralPanel` maluje na nim własne, nieprzezroczyste tło, ale overlay
    /// korzysta z `Frame::NONE`, więc ten domyślny kolor czyszczenia był
    /// widoczny wprost. Zwracamy w pełni przezroczysty kolor czyszczenia —
    /// nie wpływa to na główne okno (nie ma ono flagi `with_transparent`, więc
    /// system i tak traktuje je jako nieprzezroczyste), ale pozwala overlayowi
    /// naprawdę prześwitywać do gry pod spodem.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Rgba::TRANSPARENT.to_array()
    }
}

fn ui_device_details(ui: &mut egui::Ui, device: Option<&SoundDevice>) {
    let Some(device) = device else {
        ui.label("Nie wybrano urządzenia.");
        return;
    };

    ui.label(format!("Nazwa: {}", device.name));
    ui.label(format!("ID: {}", device.id));
    ui.label(format!("Kierunek: {}", device.get_device_direction()));
    ui.label(format!("Domyślne: {}", device.get_default_status()));

    ui.collapsing("Obsługiwane konfiguracje", |ui| {
        if device.configs.is_empty() {
            ui.label("Brak obsługiwanych konfiguracji.");
        } else {
            for config in &device.configs {
                ui.label(format!(
                    "Kanały: {}, próbkowanie: {}–{} Hz, format: {:?}, bufor: {:?}",
                    config.channels,
                    config.min_sample_rate,
                    config.max_sample_rate,
                    config.sample_format,
                    config.buffer_size,
                ));
            }
        }
    });
}
