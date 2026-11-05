//! Panel „Test kierunków” w oknie głównym. Odtwarza próby przez wybrane wyjście, zbiera
//! odpowiedzi i zapisuje każdą próbę do pliku CSV w katalogu configu (patrz
//! `sound::direction_test`).
//!
//! Dwa sposoby odpowiedzi: przycisk jednego z ośmiu kierunków albo ustawienie myszki na kompasie
//! w stronę, z której dźwięk przyszedł (kliknięcie zapisuje kąt). Kierunek nie jest pokazywany
//! przed odpowiedzią, żeby nie podpowiadać.

use crate::config_store;
use crate::sound::devices::{RadarConfig, SharedAudioAnalysis, SoundDevice};
use crate::sound::direction_test::{self as dt, RowContext, Signal, Trial};
use anyhow::Result;
use eframe::egui;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const COMPASS_SIZE: f32 = 230.0;
const COMPASS_RADIUS: f32 = 86.0;
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 200, 120);
const ERROR: egui::Color32 = egui::Color32::from_rgb(220, 80, 80);

/// Układ przycisków i podpisów kompasu: góra = przód, środek to słuchacz.
const GRID: [[Option<(&str, f32)>; 3]; 3] = [
    [Some(("Lewy przód", -45.0)), Some(("Przód", 0.0)), Some(("Prawy przód", 45.0))],
    [Some(("Lewo", -90.0)), None, Some(("Prawo", 90.0))],
    [Some(("Lewy tył", -135.0)), Some(("Tył", 180.0)), Some(("Prawy tył", 135.0))],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Answer {
    Buttons,
    Pointer,
}

/// Sesja w toku: kolejka prób, wyniki, plik CSV i stan bieżącej próby.
struct Session {
    queue: Vec<f32>,
    index: usize,
    trials: Vec<Trial>,
    signal: Signal,
    answer: Answer,
    channels: usize,
    sensitivity: f32,
    /// Zdarzenia radaru z już odtworzonej próby, która czeka na odpowiedź.
    heard: Option<Vec<(f32, f32)>>,
    last: Option<String>,
    file: File,
    path: PathBuf,
}

/// Wyniki zakończonej sesji, do przeglądu w panelu.
struct Report {
    trials: Vec<Trial>,
    path: PathBuf,
}

/// Akcja z panelu, wykonywana po narysowaniu klatki (żeby nie pożyczać `self` w trakcie rysowania).
enum Action {
    Start,
    Play,
    Answer(f32),
    Finish,
}

pub struct DirectionTester {
    /// Osobna analiza, którą karmimy próbami. Nie zabiera zdarzeń radarowi.
    probe: SharedAudioAnalysis,
    output: String,
    signal: Signal,
    answer: Answer,
    repeats: usize,
    /// Strumień bieżącej próby. Upuszczenie go przerywa dźwięk.
    playing: Option<cpal::Stream>,
    session: Option<Session>,
    report: Option<Report>,
    status: Option<String>,
}

impl DirectionTester {
    pub fn new() -> Self {
        Self {
            probe: SharedAudioAnalysis::new(),
            output: String::new(),
            signal: Signal::Level,
            answer: Answer::Buttons,
            repeats: 2,
            playing: None,
            session: None,
            report: None,
            status: None,
        }
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        shared: &SharedAudioAnalysis,
        config: &RadarConfig,
        outputs: &[SoundDevice],
    ) {
        ui.heading("Test kierunków (słuchawki)");
        ui.label(
            "Radar dostaje ten sam dźwięk, który słyszysz. Posłuchaj, skąd przyszedł, wskaż \
             kierunek. Wynik trafia do pliku CSV razem z tym, co pokazał radar.",
        );

        if !outputs.iter().any(|d| d.name == self.output) {
            self.output = outputs
                .iter()
                .find(|d| d.is_default)
                .or(outputs.first())
                .map(|d| d.name.clone())
                .unwrap_or_default();
        }

        let action = if self.session.is_some() {
            self.ui_session(ui)
        } else {
            self.ui_setup(ui, shared, outputs)
        };
        match action {
            Some(Action::Start) => self.start_session(shared, config),
            Some(Action::Play) => self.play_current(config),
            Some(Action::Answer(angle)) => self.answer(angle),
            Some(Action::Finish) => self.finish_session(),
            None => {}
        }

        if let Some(report) = &self.report {
            ui_report(ui, report);
        }
        if let Some(msg) = &self.status {
            ui.colored_label(ERROR, msg);
        }
    }

    fn ui_setup(
        &mut self,
        ui: &mut egui::Ui,
        shared: &SharedAudioAnalysis,
        outputs: &[SoundDevice],
    ) -> Option<Action> {
        egui::ComboBox::from_label("Wyjście (słuchawki)")
            .selected_text(self.output.clone())
            .show_ui(ui, |ui| {
                for dev in outputs {
                    ui.selectable_value(&mut self.output, dev.name.clone(), &dev.name);
                }
            });

        ui.horizontal(|ui| {
            ui.label("Sygnał:");
            for signal in Signal::ALL {
                ui.selectable_value(&mut self.signal, signal, signal.label());
            }
        });
        ui.horizontal(|ui| {
            ui.label("Odpowiedź:");
            ui.selectable_value(&mut self.answer, Answer::Buttons, "Przycisk kierunku");
            ui.selectable_value(&mut self.answer, Answer::Pointer, "Wskaż myszką");
        });
        ui.add(egui::Slider::new(&mut self.repeats, 1..=4).text("Powtórzeń każdego kierunku"));

        let prompts = dt::DIRECTIONS_DEG.len() * self.repeats;
        ui.weak(format!(
            "{prompts} prób · radar liczy wejście jako {} kanałów",
            self.channels(shared)
        ));
        ui.add_enabled(!self.output.is_empty(), egui::Button::new("▶ Start sesji"))
            .clicked()
            .then_some(Action::Start)
    }

    fn ui_session(&self, ui: &mut egui::Ui) -> Option<Action> {
        let session = self.session.as_ref()?;
        let total = session.queue.len();
        let played = session.heard.is_some();
        let mut action = None;

        ui.strong(format!("Próba {} z {total}", (session.index + 1).min(total)));
        ui.horizontal(|ui| {
            if ui.button("▶ Odtwórz próbę").clicked() {
                action = Some(Action::Play);
            }
            if ui.button("Zakończ sesję").clicked() {
                action = Some(Action::Finish);
            }
        });
        if !played {
            ui.weak("Najpierw odtwórz próbę, potem wskaż kierunek.");
        }

        match session.answer {
            Answer::Buttons => {
                egui::Grid::new("direction_test_buttons")
                    .spacing([6.0, 6.0])
                    .show(ui, |ui| {
                        for row in GRID {
                            for cell in row {
                                match cell {
                                    Some((label, angle)) => {
                                        let button = egui::Button::new(label)
                                            .min_size(egui::vec2(96.0, 28.0));
                                        if ui.add_enabled(played, button).clicked() {
                                            action = Some(Action::Answer(angle));
                                        }
                                    }
                                    None => {
                                        ui.weak("ty");
                                    }
                                }
                            }
                            ui.end_row();
                        }
                    });
            }
            Answer::Pointer => {
                if let Some(angle) = ui_compass(ui, played) {
                    action = Some(Action::Answer(angle));
                }
            }
        }

        if let Some(last) = &session.last {
            ui.separator();
            ui.label(last.as_str());
        }
        action
    }

    /// Liczba kanałów, z jaką analizowana jest próba: tyle co przechwytywane wejście, a w razie
    /// jego braku domyślne 7.1.
    fn channels(&self, shared: &SharedAudioAnalysis) -> usize {
        shared
            .lock()
            .map(|g| g.channel_count)
            .ok()
            .filter(|&count| count > 0)
            .unwrap_or(dt::DEFAULT_CHANNELS)
    }

    fn start_session(&mut self, shared: &SharedAudioAnalysis, config: &RadarConfig) {
        match open_log() {
            Ok((file, path)) => {
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(1);
                self.session = Some(Session {
                    queue: dt::shuffled(&dt::DIRECTIONS_DEG, self.repeats, seed),
                    index: 0,
                    trials: Vec::new(),
                    signal: self.signal,
                    answer: self.answer,
                    channels: self.channels(shared),
                    sensitivity: config.radar_sensitivity,
                    heard: None,
                    last: None,
                    file,
                    path,
                });
                self.report = None;
                self.status = None;
                self.playing = None;
            }
            Err(e) => self.status = Some(format!("Nie udało się utworzyć pliku wyników: {e}")),
        }
    }

    fn play_current(&mut self, config: &RadarConfig) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        let Some(&angle) = session.queue.get(session.index) else {
            return;
        };
        match dt::play(&self.output, angle, session.signal) {
            Ok(burst) => {
                let samples = dt::to_channels(&burst.frames, session.channels);
                session.heard = Some(dt::measure(
                    &samples,
                    session.channels,
                    burst.rate as f32,
                    &self.probe,
                    config,
                ));
                self.playing = Some(burst.stream);
                self.status = None;
            }
            Err(e) => self.status = Some(format!("Nie udało się odtworzyć próby: {e}")),
        }
    }

    fn answer(&mut self, perceived_deg: f32) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        let Some(radar_events) = session.heard.take() else {
            return;
        };
        let trial = Trial {
            expected_deg: session.queue[session.index],
            perceived_deg: Some(perceived_deg),
            radar_events,
        };
        let ctx = RowContext {
            unix_ts_ms: unix_ms(),
            trial_no: session.index + 1,
            signal: session.signal,
            pointer: session.answer == Answer::Pointer,
            channels: session.channels,
            sensitivity: session.sensitivity,
        };
        if let Err(e) = session.file.write_all(dt::csv_row(&ctx, &trial).as_bytes()) {
            self.status = Some(format!("Nie udało się zapisać wiersza: {e}"));
        }
        session.last = Some(describe(&trial));
        session.trials.push(trial);
        session.index += 1;
        if session.index >= session.queue.len() {
            self.finish_session();
        }
    }

    fn finish_session(&mut self) {
        if let Some(session) = self.session.take() {
            if !session.trials.is_empty() {
                self.report = Some(Report {
                    trials: session.trials,
                    path: session.path,
                });
            }
        }
        self.playing = None;
    }
}

/// Kompas do wskazania myszką: kliknięcie zwraca kąt względem środka (0° = przód).
fn ui_compass(ui: &mut egui::Ui, enabled: bool) -> Option<f32> {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(COMPASS_SIZE, COMPASS_SIZE), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let text = ui.visuals().text_color();
    let faded = ui.visuals().weak_text_color();

    painter.circle_stroke(center, COMPASS_RADIUS, egui::Stroke::new(1.5, faded));
    painter.circle_filled(center, 3.0, text);
    for &(label, angle) in GRID.iter().flatten().flatten() {
        let at = point_on_circle(center, COMPASS_RADIUS + 16.0, angle);
        painter.text(at, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(12.0), text);
    }

    if enabled {
        if let Some(pos) = response.hover_pos() {
            let angle = dt::angle_from_pointer(pos.x - center.x, pos.y - center.y);
            let tip = point_on_circle(center, COMPASS_RADIUS, angle);
            painter.line_segment([center, tip], egui::Stroke::new(2.0, ACCENT));
            painter.text(
                egui::pos2(center.x, rect.bottom() - 4.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{angle:+.0}°"),
                egui::FontId::proportional(13.0),
                ACCENT,
            );
        }
        if response.clicked() {
            return response
                .interact_pointer_pos()
                .map(|pos| dt::angle_from_pointer(pos.x - center.x, pos.y - center.y));
        }
    }
    None
}

fn point_on_circle(center: egui::Pos2, radius: f32, angle_deg: f32) -> egui::Pos2 {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    center + egui::vec2(radius * sin, -radius * cos)
}

fn ui_report(ui: &mut egui::Ui, report: &Report) {
    ui.separator();
    ui.strong("Wyniki ostatniej sesji");
    ui.weak(format!("Zapisano: {}", report.path.display()));
    egui::Grid::new("direction_test_summary")
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Kierunek");
            ui.strong("Próby");
            ui.strong("Radar: błąd śr.");
            ui.strong("Radar: błąd bezw.");
            ui.strong("Radar: brak");
            ui.strong("Słuch: błąd śr.");
            ui.strong("Słuch: błąd bezw.");
            ui.end_row();

            for row in dt::summarize(&report.trials) {
                if row.trials == 0 && row.expected_deg.is_some() {
                    continue;
                }
                let name = row
                    .expected_deg
                    .map_or_else(|| "wszystkie".to_string(), |a| format!("{} ({a:.0}°)", direction_name(a)));
                ui.label(name);
                ui.label(row.trials.to_string());
                ui.label(fmt_signed(row.radar_bias_deg));
                ui.label(fmt_plain(row.radar_mae_deg));
                ui.label(row.radar_missing.to_string());
                ui.label(fmt_signed(row.heard_bias_deg));
                ui.label(fmt_plain(row.heard_mae_deg));
                ui.end_row();
            }
        });
    ui.weak(
        "Błąd śr. ze znakiem: dodatni = w prawo od prawdziwego kierunku. Błąd bezw.: \
         o ile stopni średnio.",
    );
}

fn direction_name(angle: f32) -> &'static str {
    GRID.iter()
        .flatten()
        .flatten()
        .find(|(_, a)| (a - angle).abs() < 0.5)
        .map_or("?", |&(name, _)| name)
}

fn fmt_signed(value: Option<f32>) -> String {
    value.map_or_else(|| "—".to_string(), |a| format!("{a:+.1}°"))
}

fn fmt_plain(value: Option<f32>) -> String {
    value.map_or_else(|| "—".to_string(), |a| format!("{a:.1}°"))
}

fn describe(trial: &Trial) -> String {
    let heard = trial
        .perceived_deg
        .map_or_else(|| "—".to_string(), |a| format!("{a:+.0}°"));
    let radar = match trial.radar_deg() {
        Some(a) => format!("{a:+.0}° ({} zdarzeń)", trial.radar_events.len()),
        None => "brak zdarzenia".to_string(),
    };
    format!(
        "Było {:+.0}° · usłyszano {heard} · radar {radar}",
        trial.expected_deg
    )
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Nowy plik CSV w katalogu configu. Nazwa z czasem, więc kolejne sesje się nie nadpisują.
fn open_log() -> Result<(File, PathBuf)> {
    let dir = config_store::config_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("direction_test_{}.csv", unix_ms()));
    let mut file = OpenOptions::new().create_new(true).write(true).open(&path)?;
    file.write_all(dt::CSV_HEADER.as_bytes())?;
    Ok((file, path))
}
