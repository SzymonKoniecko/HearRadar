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

use crate::sound::devices::{self, MonoEar, RadarConfig, SharedAudioAnalysis, SoundDevice};
use crate::sound::mono_router::{self, MonoRouter};
use crate::ui::radar_egui::{draw_radar, SampleBuffer, TestScenario};
use eframe::egui;
use egui::{ViewportBuilder, ViewportCommand, ViewportId};
use std::f32::consts::TAU;

const OVERLAY_SIZE: f32 = 260.0;

// ── Stan aplikacji ────────────────────────────────────────────────────────────

pub struct HearRadarApp {
    analysis: SharedAudioAnalysis,

    // Wygładzony kąt/intensywność do rysowania (interpolacja klatka-po-klatce,
    // odpowiednik dawnej pętli `use_future` w Dioxusie).
    display_angle: f32,
    display_intensity: f32,

    // Bufor próbek radaru (używany zarówno w overlay, jak i w podglądzie).
    buffer: SampleBuffer,
    sim_phase: f32,
    test_phase: f32,
    test_scenario: TestScenario,

    // Konfiguracja wyglądu radaru.
    radar_config: RadarConfig,

    // Lista urządzeń wejściowych + błąd ładowania.
    devices: Result<Vec<SoundDevice>, String>,
    selected_device: String,
    active_stream: Option<cpal::Stream>,

    // Mono router.
    out_devices: Result<Vec<SoundDevice>, String>,
    mono_sel_in: String,
    mono_sel_out: String,
    mono_router: Option<MonoRouter>,
    mono_ear: MonoEar,

    // Overlay.
    overlay_open: bool,
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
        let mono_sel_in = devices
            .as_ref()
            .ok()
            .and_then(|d| d.first())
            .map(|d| d.name.clone())
            .unwrap_or_default();
        let mono_sel_out = out_devices
            .as_ref()
            .ok()
            .and_then(|ds| ds.iter().find(|d| d.is_default).or_else(|| ds.first()))
            .map(|d| d.name.clone())
            .unwrap_or_default();

        Self {
            analysis,
            display_angle: 0.0,
            display_intensity: 0.0,
            buffer: SampleBuffer::new(256),
            sim_phase: 0.0,
            test_phase: 0.0,
            test_scenario: TestScenario::Off,
            radar_config: RadarConfig::default(),
            devices,
            selected_device,
            active_stream: None,
            out_devices,
            mono_sel_in,
            mono_sel_out,
            mono_router: None,
            mono_ear: MonoEar::Off,
            overlay_open: false,
        }
    }

    /// Odczytuje `angle`/`intensity` ze współdzielonej analizy i wygładza je
    /// klatka po klatce (szybki atak, płynne opadanie intensywności).
    fn tick_display(&mut self) {
        let (target_angle, target_intensity) = self
            .analysis
            .lock()
            .map(|g| (g.angle, g.intensity))
            .unwrap_or((0.0, 0.0));

        self.display_angle = self.display_angle * 0.75 + target_angle * 0.25;
        self.display_intensity = if target_intensity > self.display_intensity {
            target_intensity
        } else {
            self.display_intensity * 0.88
        };
    }

    /// Aktualizuje bufor próbek rysowany przez radar — scenariusz testowy
    /// albo żywe audio, dokładnie jak dawny `RadarEguiApp::logic`.
    fn tick_radar_buffer(&mut self) {
        if self.test_scenario != TestScenario::Off {
            self.test_phase += 0.04;
            self.buffer.samples = self.test_scenario.generate(256, self.test_phase);
            return;
        }

        let (angle, intensity) = self
            .analysis
            .lock()
            .map(|g| (g.angle, g.intensity))
            .unwrap_or((0.0, 0.0));

        if intensity > 0.01 {
            let ar = angle.to_radians();
            let new: Vec<f32> = (0..16)
                .map(|i| (ar + i as f32 / 16.0 * std::f32::consts::FRAC_PI_4).sin() * intensity)
                .collect();
            self.buffer.push_samples(&new);
        } else {
            self.sim_phase += 0.08;
            let new: Vec<f32> = (0..16)
                .map(|i| (self.sim_phase + i as f32 / 16.0 * TAU).sin() * 0.3)
                .collect();
            self.buffer.push_samples(&new);
        }

        self.buffer.decay(0.97);
    }

    fn write_test_scenario(&self, idx: u8) {
        if let Ok(mut lock) = self.analysis.lock() {
            lock.test_scenario = idx;
        }
    }

    fn write_radar_config(&self) {
        if let Ok(mut lock) = self.analysis.lock() {
            lock.radar_config = self.radar_config;
        }
    }

    // ── Panel: wybór urządzenia wejściowego ──────────────────────────────
    fn ui_device_selector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Źródło audio");

        let devices = match &self.devices {
            Ok(devs) => devs.clone(),
            Err(err) => {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), format!("Nie udało się wczytać urządzeń audio: {err}"));
                if ui.button("Spróbuj ponownie").clicked() {
                    self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
                }
                return;
            }
        };

        // Na macOS bez BlackHole lista wejść bywa pusta — pokazujemy instrukcję.
        #[cfg(target_os = "macos")]
        if devices.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), "Nie znaleziono wejściowych urządzeń audio.");
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
                        ui.selectable_value(&mut self.selected_device, device.name.clone(), &device.name);
                    }
                });

            if ui.button("Odśwież").clicked() {
                self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
            }

            let is_streaming = self.active_stream.is_some();
            let btn_label = if is_streaming { "Zatrzymaj" } else { "Przechwytuj" };
            if ui.button(btn_label).clicked() {
                if is_streaming {
                    self.active_stream = None;
                } else {
                    match devices::fetch_audio_from_device(&self.selected_device, self.analysis.clone()) {
                        Ok(s) => self.active_stream = Some(s),
                        Err(e) => eprintln!("Stream error: {e}"),
                    }
                }
            }

            if ui.button("🎯 Overlay").clicked() {
                self.overlay_open = true;
            }
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
                self.out_devices = devices::fetch_physical_output_devices().map_err(|e| e.to_string());
                self.devices = devices::fetch_output_devices().map_err(|e| e.to_string());
            }
        });
        ui.label("Odsłuchaj sygnał mono w jednej słuchawce bez wirtualnego urządzenia systemowego.");

        let in_devices: Vec<SoundDevice> = self.devices.clone().unwrap_or_default();
        let out_devices: Vec<SoundDevice> = self.out_devices.clone().unwrap_or_default();

        if self.mono_sel_in.is_empty() {
            if let Some(d) = in_devices.first() {
                self.mono_sel_in = d.name.clone();
            }
        }
        if self.mono_sel_out.is_empty() {
            if let Some(d) = out_devices.first() {
                self.mono_sel_out = d.name.clone();
            }
        }

        egui::ComboBox::from_label("Wejście")
            .selected_text(self.mono_sel_in.clone())
            .show_ui(ui, |ui| {
                for dev in &in_devices {
                    ui.selectable_value(&mut self.mono_sel_in, dev.name.clone(), &dev.name);
                }
            });

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
                if ui.selectable_label(self.mono_ear == ear, ear.label()).clicked() {
                    self.mono_ear = ear;
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = ear;
                    }
                }
            }
        });

        let is_routing = self.mono_router.is_some();
        let can_start = !self.mono_sel_in.is_empty() && !self.mono_sel_out.is_empty();
        ui.add_enabled_ui(can_start || is_routing, |ui| {
            let label = if is_routing { "⏹ Zatrzymaj routing" } else { "▶ Uruchom routing" };
            if ui.button(label).clicked() {
                if is_routing {
                    self.mono_router = None;
                    self.mono_ear = MonoEar::Off;
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = MonoEar::Off;
                    }
                } else {
                    if self.mono_ear == MonoEar::Off {
                        self.mono_ear = MonoEar::Left;
                    }
                    if let Ok(mut lock) = self.analysis.lock() {
                        lock.mono_ear = self.mono_ear;
                    }
                    match mono_router::start_mono_router(&self.mono_sel_in, &self.mono_sel_out, self.analysis.clone()) {
                        Ok(r) => self.mono_router = Some(r),
                        Err(e) => eprintln!("[mono_router] start error: {e}"),
                    }
                }
            }
        });
    }

    // ── Panel: scenariusze testowe radaru ────────────────────────────────
    fn ui_test_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Test radaru");
        ui.label("Wybierz scenariusz, aby zobaczyć jak radar rysuje różne wzorce próbek.");
        egui::Grid::new("radar_test_grid").show(ui, |ui| {
            for (i, scenario) in TestScenario::ALL.iter().enumerate() {
                let scenario = *scenario;
                if ui.selectable_label(self.test_scenario == scenario, scenario.label()).clicked() {
                    self.test_scenario = scenario;
                    self.write_test_scenario(scenario as u8);
                }
                if (i + 1) % 2 == 0 {
                    ui.end_row();
                }
            }
        });
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
        changed |= ui.add(egui::Slider::new(&mut self.radar_config.anchor_x, 0.0..=1.0).text("Pozycja X")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.radar_config.anchor_y, 0.0..=1.0).text("Pozycja Y")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.radar_config.radius, 40.0..=300.0).text("Rozmiar (px)")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.radar_config.bg_alpha, 0.0..=1.0).text("Tło")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.radar_config.fg_alpha, 0.0..=1.0).text("Jasność")).changed();

        if changed {
            self.write_radar_config();
        }
    }

    // ── Panel: podgląd radaru (mały obszar rysowania w głównym oknie) ────
    fn ui_radar_preview(&self, ui: &mut egui::Ui) {
        ui.heading("Podgląd radaru");
        let size = self.radar_config.radius * 2.0 + 20.0;
        let (rect, _response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        draw_radar(&painter, rect.center(), &self.radar_config, &self.buffer.samples);
    }

    // ── Zawartość overlay viewportu ───────────────────────────────────────
    /// Rysuje overlay i zwraca `true`, jeśli użytkownik zażądał jego zamknięcia
    /// (przycisk „⚙” albo systemowe zamknięcie okna).
    fn ui_overlay_contents(&self, root_ui: &mut egui::Ui) -> bool {
        let ctx = root_ui.ctx().clone();
        let mut close_requested = false;

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root_ui, |ui| {
                let rect = ui.max_rect();
                let center = rect.center();
                let painter = ui.painter();
                draw_radar(painter, center, &self.radar_config, &self.buffer.samples);

                // Przeciąganie okna: kliknięcie i przeciągnięcie w dowolnym
                // miejscu przenosi overlay (zastępuje dawne `desktop.drag()`).
                let drag_response = ui.interact(rect, ui.id().with("overlay_drag"), egui::Sense::drag());
                if drag_response.drag_started() {
                    ctx.send_viewport_cmd(ViewportCommand::StartDrag);
                }

                // Mały przycisk zamknięcia w rogu (odpowiednik "⚙" z Dioxusa).
                egui::Area::new(ui.id().with("overlay_close_area"))
                    .fixed_pos(rect.left_top() + egui::vec2(6.0, 6.0))
                    .show(&ctx, |ui| {
                        if ui.small_button("⚙").on_hover_text("Zamknij overlay").clicked() {
                            close_requested = true;
                        }
                    });
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

        self.tick_display();
        self.tick_radar_buffer();

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

                    self.ui_radar_preview(&mut cols[1]);
                });
            });
        });

        if self.overlay_open {
            // `show_viewport_immediate` renderuje ten viewport synchronicznie
            // w ramach tej samej klatki i tego samego wątku co okno główne —
            // wciąż jeden proces, jeden event loop (tylko `winit`/`eframe` na
            // wątku głównym), w przeciwieństwie do poprzedniej architektury,
            // która uruchamiała drugi, osobny event loop w wątku pobocznym.
            // (`show_viewport_deferred` wymaga closure `Fn + Send + Sync`,
            // co uniemożliwia bezpośrednie przechwycenie `&mut self`.)
            let close_requested = ctx.show_viewport_immediate(
                ViewportId::from_hash_of("hear_radar_overlay"),
                ViewportBuilder::default()
                    .with_title("HearRadar — Overlay")
                    .with_inner_size([OVERLAY_SIZE, OVERLAY_SIZE])
                    .with_decorations(false)
                    .with_transparent(true)
                    .with_always_on_top()
                    .with_resizable(false),
                |ui, _class| self.ui_overlay_contents(ui),
            );

            if close_requested {
                self.overlay_open = false;
            }
        }

        ctx.request_repaint();
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
                    config.channels, config.min_sample_rate, config.max_sample_rate,
                    config.sample_format, config.buffer_size,
                ));
            }
        }
    });
}
