use crate::sound::devices::SharedAudioAnalysis;
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use std::f32::consts::TAU;

// ── Kolory ────────────────────────────────────────────────────────────────────

const COLOR_RING: Color32 = Color32::from_rgba_premultiplied(0, 255, 157, 80);
const COLOR_RING_INNER: Color32 = Color32::from_rgba_premultiplied(0, 255, 157, 35);
const COLOR_CROSS: Color32 = Color32::from_rgba_premultiplied(0, 255, 157, 50);
const COLOR_CENTER: Color32 = Color32::from_rgba_premultiplied(0, 255, 157, 200);
const COLOR_BG: Color32 = Color32::from_rgba_premultiplied(0, 8, 16, 210);

// ── Główna funkcja rysująca ───────────────────────────────────────────────────

/// Rysuje minimalistyczny radar audio wewnątrz przekazanego `Painter`.
/// `center`  — środek radaru w przestrzeni ekranu
/// `radius`  — promień radaru w pikselach
/// `samples` — surowe próbki audio; jedna próbka → jedna kropka
pub fn draw_radar(painter: &Painter, center: Pos2, radius: f32, samples: &[f32]) {
    // Tło
    painter.circle_filled(center, radius, COLOR_BG);

    // 3 koncentryczne pierścienie
    for frac in [1.0f32, 0.66, 0.33] {
        let color = if frac == 1.0 { COLOR_RING } else { COLOR_RING_INNER };
        let width = if frac == 1.0 { 1.5 } else { 0.8 };
        painter.circle_stroke(center, radius * frac, Stroke::new(width, color));
    }

    // Krzyż kierunków
    let cross_stroke = Stroke::new(0.8, COLOR_CROSS);
    painter.line_segment(
        [center - Vec2::new(0.0, radius), center + Vec2::new(0.0, radius)],
        cross_stroke,
    );
    painter.line_segment(
        [center - Vec2::new(radius, 0.0), center + Vec2::new(radius, 0.0)],
        cross_stroke,
    );

    // Punkt centralny
    painter.circle_filled(center, 2.5, COLOR_CENTER);

    if samples.is_empty() {
        return;
    }

    // Kropki próbek
    let count = samples.len();
    for (i, &sample) in samples.iter().enumerate() {
        let angle = (i as f32 / count as f32) * TAU - std::f32::consts::FRAC_PI_2;
        let abs_val = sample.abs().clamp(0.0, 1.0);

        // Promień: minimum 5 px (środek), max = pełny okrąg
        let dot_r = (abs_val * radius).max(5.0);

        let x = center.x + dot_r * angle.cos();
        let y = center.y + dot_r * angle.sin();

        // Kolor: cichy = ciemny cyjan, głośny = prawie biały
        let alpha = (abs_val * 180.0 + 40.0) as u8;
        let ch = (abs_val * 100.0 + 155.0) as u8;
        let dot_color = Color32::from_rgba_premultiplied(0, ch, ch, alpha);

        painter.circle_filled(Pos2::new(x, y), 2.5, dot_color);
    }
}

// ── Bufor próbek ze wzmocnieniem zaniku ──────────────────────────────────────

pub struct SampleBuffer {
    pub samples: Vec<f32>,
    capacity: usize,
}

impl SampleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Dodaje nowe próbki, utrzymuje stały rozmiar bufora.
    pub fn push_samples(&mut self, new_samples: &[f32]) {
        self.samples.extend_from_slice(new_samples);
        if self.samples.len() > self.capacity {
            let drain = self.samples.len() - self.capacity;
            self.samples.drain(..drain);
        }
    }

    /// Lekkie opadanie amplitudy co klatkę (efekt "ghost trail").
    pub fn decay(&mut self, factor: f32) {
        for s in &mut self.samples {
            *s *= factor;
        }
    }
}

// ── Aplikacja eframe ─────────────────────────────────────────────────────────

pub struct RadarEguiApp {
    analysis: SharedAudioAnalysis,
    buffer: SampleBuffer,
    sim_phase: f32,
}

impl RadarEguiApp {
    pub fn new(analysis: SharedAudioAnalysis) -> Self {
        Self {
            analysis,
            buffer: SampleBuffer::new(256),
            sim_phase: 0.0,
        }
    }
}

impl eframe::App for RadarEguiApp {
    /// Logika aktualizacji (odczyt audio, generowanie próbek) — wywoływana co klatkę.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let (angle, intensity) = if let Ok(lock) = self.analysis.lock() {
            (lock.angle, lock.intensity)
        } else {
            (0.0f32, 0.0f32)
        };

        if intensity > 0.01 {
            let angle_rad = angle.to_radians();
            let new_samples: Vec<f32> = (0..16)
                .map(|i| {
                    let t = i as f32 / 16.0;
                    let phase = angle_rad + t * std::f32::consts::FRAC_PI_4;
                    phase.sin() * intensity
                })
                .collect();
            self.buffer.push_samples(&new_samples);
        } else {
            self.sim_phase += 0.08;
            let new_samples: Vec<f32> = (0..16)
                .map(|i| {
                    let t = i as f32 / 16.0;
                    (self.sim_phase + t * TAU).sin() * 0.3
                })
                .collect();
            self.buffer.push_samples(&new_samples);
        }

        self.buffer.decay(0.97);
        ctx.request_repaint();
    }

    /// Rysowanie UI — radar w prawym dolnym rogu okna.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        const RADAR_RADIUS: f32 = 125.0; // diameter 250 px
        const MARGIN: f32 = 16.0;

        let rect = ui.max_rect();
        let center = Pos2::new(
            rect.right() - RADAR_RADIUS - MARGIN,
            rect.bottom() - RADAR_RADIUS - MARGIN,
        );

        let painter = ui.painter();
        draw_radar(painter, center, RADAR_RADIUS, &self.buffer.samples);
    }
}

// ── Uruchomienie okna eframe ─────────────────────────────────────────────────

pub fn launch_radar_window(analysis: SharedAudioAnalysis) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("HearRadar — Audio Radar")
            .with_inner_size([860.0, 700.0])
            .with_transparent(true)
            .with_decorations(true),
        ..Default::default()
    };

    eframe::run_native(
        "HearRadar",
        options,
        Box::new(|_cc| Ok(Box::new(RadarEguiApp::new(analysis)))),
    )
    .expect("eframe failed to start");
}
