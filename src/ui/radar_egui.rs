use crate::sound::devices::{RadarConfig, SharedAudioAnalysis};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use std::f32::consts::TAU;

// ── Scenariusze testowe ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum TestScenario {
    Off         = 0,
    FullCircle  = 1,
    AllSilent   = 2,
    HalfLoud    = 3,
    Sine        = 4,
    SinglePulse = 5,
    RampUp      = 6,
    Noise       = 7,
}

impl TestScenario {
    pub const ALL: &'static [Self] = &[
        Self::Off,
        Self::FullCircle,
        Self::AllSilent,
        Self::HalfLoud,
        Self::Sine,
        Self::SinglePulse,
        Self::RampUp,
        Self::Noise,
    ];

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::FullCircle,
            2 => Self::AllSilent,
            3 => Self::HalfLoud,
            4 => Self::Sine,
            5 => Self::SinglePulse,
            6 => Self::RampUp,
            7 => Self::Noise,
            _ => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off         => "🎙 Live audio",
            Self::FullCircle  => "⬤ Pełny okrąg (max)",
            Self::AllSilent   => "· Cichy sygnał (min)",
            Self::HalfLoud    => "◑ Półokrąg",
            Self::Sine        => "∿ Sinusoida",
            Self::SinglePulse => "▲ Pojedynczy impuls",
            Self::RampUp      => "/ Klin rosnący",
            Self::Noise       => "≋ Szum losowy",
        }
    }

    fn generate(self, n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                match self {
                    Self::Off         => 0.0,
                    Self::FullCircle  => 1.0,
                    Self::AllSilent   => 0.05,
                    Self::HalfLoud    => if t < 0.5 { 1.0 } else { 0.05 },
                    Self::Sine        => ((phase + t * TAU).sin() * 0.5 + 0.5).clamp(0.0, 1.0),
                    Self::SinglePulse => if i == 0 { 1.0 } else { 0.0 },
                    Self::RampUp      => t,
                    Self::Noise       => {
                        let s  = (i as u32).wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let s2 = s.wrapping_add((phase * 1000.0) as u32);
                        s2 as f32 / u32::MAX as f32
                    }
                }
            })
            .collect()
    }
}

// ── Pomocnik kolorów z alpha ──────────────────────────────────────────────────

/// Skaluje kanał alpha koloru przez mnożnik (0.0..1.0).
fn with_alpha(c: Color32, mult: f32) -> Color32 {
    let a = (c.a() as f32 * mult.clamp(0.0, 1.0)) as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

// ── Główna funkcja rysująca ───────────────────────────────────────────────────

/// Rysuje minimalistyczny radar audio używając przekazanej konfiguracji.
pub fn draw_radar(painter: &Painter, center: Pos2, cfg: &RadarConfig, samples: &[f32]) {
    let r = cfg.radius;
    let fg = cfg.fg_alpha;

    // Kolory bazowe
    let c_ring   = Color32::from_rgba_unmultiplied(0, 255, 157, 80);
    let c_inner  = Color32::from_rgba_unmultiplied(0, 255, 157, 35);
    let c_cross  = Color32::from_rgba_unmultiplied(0, 255, 157, 50);
    let c_center = Color32::from_rgba_unmultiplied(0, 255, 157, 200);
    let c_bg     = Color32::from_rgba_unmultiplied(0, 8, 16, (cfg.bg_alpha * 255.0) as u8);

    painter.circle_filled(center, r, c_bg);

    for frac in [1.0f32, 0.66, 0.33] {
        let color = if frac == 1.0 { with_alpha(c_ring, fg) } else { with_alpha(c_inner, fg) };
        let width = if frac == 1.0 { 1.5 } else { 0.8 };
        painter.circle_stroke(center, r * frac, Stroke::new(width, color));
    }

    let cross = Stroke::new(0.8, with_alpha(c_cross, fg));
    painter.line_segment([center - Vec2::new(0.0, r), center + Vec2::new(0.0, r)], cross);
    painter.line_segment([center - Vec2::new(r, 0.0), center + Vec2::new(r, 0.0)], cross);
    painter.circle_filled(center, 2.5, with_alpha(c_center, fg));

    if samples.is_empty() {
        return;
    }

    let count = samples.len();
    for (i, &sample) in samples.iter().enumerate() {
        let angle   = (i as f32 / count as f32) * TAU - std::f32::consts::FRAC_PI_2;
        let abs_val = sample.abs().clamp(0.0, 1.0);
        let dot_r   = (abs_val * r).max(5.0);

        let x = center.x + dot_r * angle.cos();
        let y = center.y + dot_r * angle.sin();

        let alpha     = ((abs_val * 180.0 + 40.0) * fg) as u8;
        let ch        = (abs_val * 100.0 + 155.0) as u8;
        let dot_color = Color32::from_rgba_unmultiplied(0, ch, ch, alpha);

        painter.circle_filled(Pos2::new(x, y), 2.5, dot_color);
    }
}

// ── Bufor próbek ─────────────────────────────────────────────────────────────

pub struct SampleBuffer {
    pub samples: Vec<f32>,
    capacity: usize,
}

impl SampleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { samples: Vec::with_capacity(capacity), capacity }
    }

    pub fn push_samples(&mut self, new: &[f32]) {
        self.samples.extend_from_slice(new);
        if self.samples.len() > self.capacity {
            let drain = self.samples.len() - self.capacity;
            self.samples.drain(..drain);
        }
    }

    pub fn decay(&mut self, factor: f32) {
        for s in &mut self.samples { *s *= factor; }
    }
}

// ── Aplikacja eframe ─────────────────────────────────────────────────────────

pub struct RadarEguiApp {
    analysis: SharedAudioAnalysis,
    buffer:   SampleBuffer,
    sim_phase:  f32,
    test_phase: f32,
}

impl RadarEguiApp {
    pub fn new(analysis: SharedAudioAnalysis) -> Self {
        Self {
            analysis,
            buffer: SampleBuffer::new(256),
            sim_phase:  0.0,
            test_phase: 0.0,
        }
    }

    fn read_scenario(&self) -> TestScenario {
        self.analysis.lock()
            .map(|g| TestScenario::from_u8(g.test_scenario))
            .unwrap_or(TestScenario::Off)
    }

    fn read_config(&self) -> RadarConfig {
        self.analysis.lock()
            .map(|g| g.radar_config)
            .unwrap_or_default()
    }
}

impl eframe::App for RadarEguiApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let scenario = self.read_scenario();

        if scenario != TestScenario::Off {
            self.test_phase += 0.04;
            self.buffer.samples = scenario.generate(256, self.test_phase);
        } else {
            let (angle, intensity) = self.analysis.lock()
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

        ctx.request_repaint();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let cfg  = self.read_config();
        let rect = ui.max_rect();

        // Pozycja: anchor_x/anchor_y jako ułamek rozmiarów okna
        let margin = cfg.radius + 4.0;
        let cx = rect.left() + cfg.anchor_x * (rect.width()  - 2.0 * margin) + margin;
        let cy = rect.top()  + cfg.anchor_y * (rect.height() - 2.0 * margin) + margin;
        let center = Pos2::new(cx, cy);

        draw_radar(ui.painter(), center, &cfg, &self.buffer.samples);
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
