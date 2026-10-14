use crate::sound::devices::{DirectionEvent, RadarConfig};
use crate::sound::direction::{channel_layout, ChannelRole};
use crate::sound::spectrum::{resolve_color, ColorRule};
use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use std::collections::VecDeque;

/// Abstraction over the drawing primitives `draw_radar` needs, so the same
/// layout/animation logic can target either an `egui::Painter` (main window
/// preview, and the cross-platform transparent-viewport overlay) or a
/// software rasterizer (the native layered-window overlay on Windows, which
/// bypasses egui's GPU-backed viewport entirely — see `ui::overlay_win`).
pub trait RadarCanvas {
    fn circle_filled(&self, center: Pos2, radius: f32, color: Color32);
    fn circle_stroke(&self, center: Pos2, radius: f32, stroke: Stroke);
    fn line_segment(&self, points: [Pos2; 2], stroke: Stroke);
    fn rect_filled(&self, rect: Rect, corner_radius: f32, color: Color32);
    /// Filled, non-overlapping triangle — used for the perceptual-direction
    /// beacon's arrowhead (see `draw_radar`), which needs a real pointed
    /// shape rather than something faked out of circles/lines.
    fn triangle_filled(&self, points: [Pos2; 3], color: Color32);
    /// No-op on backends that can't rasterize text (e.g. the Windows skia
    /// overlay) — direction labels simply don't render there.
    fn text(&self, pos: Pos2, anchor: Align2, text: &str, font: FontId, color: Color32);
}

impl RadarCanvas for Painter {
    fn circle_filled(&self, center: Pos2, radius: f32, color: Color32) {
        Painter::circle_filled(self, center, radius, color);
    }
    fn circle_stroke(&self, center: Pos2, radius: f32, stroke: Stroke) {
        Painter::circle_stroke(self, center, radius, stroke);
    }
    fn line_segment(&self, points: [Pos2; 2], stroke: Stroke) {
        Painter::line_segment(self, points, stroke);
    }
    fn rect_filled(&self, rect: Rect, corner_radius: f32, color: Color32) {
        Painter::rect_filled(self, rect, corner_radius, color);
    }
    fn triangle_filled(&self, points: [Pos2; 3], color: Color32) {
        self.add(egui::Shape::convex_polygon(
            points.to_vec(),
            color,
            egui::Stroke::NONE,
        ));
    }
    fn text(&self, pos: Pos2, anchor: Align2, text: &str, font: FontId, color: Color32) {
        Painter::text(self, pos, anchor, text, font, color);
    }
}

// ── Scenariusze testowe ───────────────────────────────────────────────────────
//
// Scenariusze generują syntetyczny poziom RMS (0.0..=1.0) na kanał, dokładnie
// w takim kształcie, w jakim `analyze_channels` dostarczałby dane z żywego
// strumienia audio — dzięki temu przechodzą przez tę samą ścieżkę renderowania
// (RadarState::update) co prawdziwy dźwięk.

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
    /// Dwa DOKŁADNIE przeciwstawne kanały (SL/SR, 180° od siebie), pulsujące
    /// RÓWNOCZEŚNIE z podobną energią — dokładny przypadek, w którym dawny,
    /// pojedynczy sumaryczny wskaźnik kierunku (suma wektorowa wszystkich
    /// kanałów, patrz `direction::compute_direction`) dawał `sum≈0` i mylącą,
    /// zanikającą strzałkę "z przodu", mimo że oba źródła realnie grają
    /// (np. silnik z lewej + wybuch z prawej). Powinien pokazać DWA osobne,
    /// niezależne wskaźniki kierunku naraz — po jednym na stronę.
    OpposingSources = 8,
    /// Prawdziwa sinusoida o zadanej częstotliwości (Hz), puszczona przez
    /// TEN SAM potok analizy co żywe audio (`analyze_channels` +
    /// `spectrum::analyze_channel_bands`) — w przeciwieństwie do pozostałych
    /// scenariuszy, które podają syntetyczny RMS wprost. Służy do
    /// weryfikacji progów kolorowania pasm (200 Hz / 2 kHz) na żywo: przesuń
    /// suwak Hz i patrz, kiedy kropka zmienia kolor. Obsługiwane osobno w
    /// `HearRadarApp::tick_radar_state`, nie przez [`Self::generate`].
    Tone        = 9,
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
        Self::OpposingSources,
        Self::Tone,
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
            8 => Self::OpposingSources,
            9 => Self::Tone,
            _ => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off         => "🎙 Live audio",
            Self::FullCircle  => "⬤ Wszystkie kanały (max)",
            Self::AllSilent   => "· Cisza (min)",
            Self::HalfLoud    => "◑ Połowa kanałów",
            Self::Sine        => "∿ Sinusoida (rotacja)",
            Self::SinglePulse => "▲ Pojedynczy impuls (transient)",
            Self::RampUp      => "/ Klin rosnący",
            Self::Noise       => "≋ Szum losowy (seed)",
            Self::OpposingSources => "⇄ Dwa przeciwstawne źródła naraz",
            Self::Tone        => "🎵 Ton o zadanym Hz (kolor pasma)",
        }
    }

    /// Zwraca symulowany RMS dla każdego z `channels` kanałów, w danej fazie.
    /// `seed` różnicuje/odtwarza deterministycznie scenariusze losowe
    /// (`Noise`) i cykliczne (`SinglePulse`) — ten sam `seed` zawsze da ten
    /// sam przebieg, więc łatwiej porównywać zmiany w renderowaniu.
    pub fn generate(self, channels: usize, phase: f32, seed: u32) -> Vec<f32> {
        (0..channels)
            .map(|i| {
                let t = i as f32 / channels.max(1) as f32;
                match self {
                    Self::Off         => 0.0,
                    Self::FullCircle  => 0.9,
                    Self::AllSilent   => 0.02,
                    Self::HalfLoud    => if i % 2 == 0 { 0.9 } else { 0.02 },
                    Self::Sine        => ((phase - t * std::f32::consts::TAU).sin() * 0.5 + 0.5).clamp(0.0, 1.0),
                    Self::SinglePulse => {
                        // Krótki, cykliczny impuls na jednym kanale na raz — pokazuje
                        // detekcję transientów (większa/jaśniejsza kropka + pierścień).
                        let active = ((phase * 0.5) as usize + seed as usize) % channels;
                        let just_fired = (phase * 8.0).fract() < 0.15;
                        if i == active && just_fired { 1.0 } else { 0.02 }
                    }
                    Self::RampUp      => (phase.fract() * (i as f32 + 1.0) / channels as f32).clamp(0.0, 1.0),
                    Self::Noise       => {
                        let s  = (i as u32).wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let s  = s.wrapping_add(seed.wrapping_mul(2_654_435_761));
                        let s2 = s.wrapping_add((phase * 1000.0) as u32);
                        s2 as f32 / u32::MAX as f32
                    }
                    Self::OpposingSources => {
                        // SL i SR leżą dokładnie naprzeciw siebie (-90°/90°) —
                        // pulsują RAZEM, żeby oba wywołały onset w tej samej
                        // ramce (patrz doc-komentarz wariantu). Poza burstem
                        // cichy szum tła, żeby EMA zdążyła opaść między pulsami.
                        let role = channel_layout(channels).get(i).copied();
                        let is_pair = matches!(role, Some(ChannelRole::SideLeft) | Some(ChannelRole::SideRight));
                        let just_fired = (phase * 4.0).fract() < 0.15;
                        if is_pair && just_fired { 0.8 } else { 0.02 }
                    }
                    // Obsługiwane osobno w `tick_radar_state` (potrzebuje realnych
                    // próbek, nie samego RMS) — tu tylko niegroźna wartość zapasowa.
                    Self::Tone        => 0.0,
                }
            })
            .collect()
    }

    /// Syntetyczna energia reguł kolorowania dla scenariuszy testowych: nie
    /// liczymy prawdziwego widma (nie ma prawdziwego sygnału), tylko
    /// demonstrujemy kolorowanie kropek — dominująca reguła powoli krąży po
    /// kanałach w czasie, żeby w podglądzie było widać WSZYSTKIE skonfigurowane
    /// kolory, niezależnie od tego ile użytkownik ich zdefiniował.
    pub fn synthetic_rule_energies(self, channels: usize, phase: f32, rule_count: usize) -> Vec<Vec<f32>> {
        let rule_count = rule_count.max(1);
        (0..channels)
            .map(|i| {
                let dominant = (i + (phase * 0.4) as usize) % rule_count;
                let mut e = vec![0.05f32; rule_count];
                e[dominant] = 0.8;
                e
            })
            .collect()
    }
}

// ── Pomocnik kolorów z alpha ──────────────────────────────────────────────────

fn with_alpha(c: Color32, mult: f32) -> Color32 {
    let a = (c.a() as f32 * mult.clamp(0.0, 1.0)) as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

// ── Kropki i stan radaru w czasie ─────────────────────────────────────────────

/// Minimalny odstęp między kolejnymi ramkami analizy pokazywanymi jako nowe
/// kropki (20-40 ms wg specyfikacji — używamy środka tego zakresu).
const SPAWN_INTERVAL_SECS: f64 = 0.03;

/// Dolny/górny limit liczby jednocześnie przechowywanych kropek na kanał.
/// Rzeczywisty limit jest liczony z `decay_secs` (patrz [`RadarState::update`]),
/// żeby przy dłuższym zaniku (np. 2-3 s) stare kropki nie znikały
/// przedwcześnie tylko dlatego, że nadeszło kilka nowych ramek — echo ma być
/// widoczne przez CAŁY czas zaniku, nie tylko przez ostatnie kilka klatek.
const MIN_DOTS_PER_CHANNEL: usize = 4;
const MAX_DOTS_PER_CHANNEL: usize = 200;

/// Próg wykrywania nagłych skoków energii (transientów): kropka jest
/// oznaczana jako transient, gdy jej energia przekracza średnią kroczącą
/// kanału o ten mnożnik (np. 2.5x ~ +8 dB).
const TRANSIENT_RATIO: f32 = 2.5;

#[derive(Clone, Copy, Debug)]
pub struct ChannelDot {
    /// Czas powstania kropki (sekundy, `egui::InputState::time`).
    pub spawned_at: f64,
    /// Energia (RMS) kanału w tej ramce, 0.0..=1.0.
    pub energy: f32,
    /// Czy ta ramka została oznaczona jako nagły skok energii.
    pub transient: bool,
    /// Kolor wybrany wg energetycznie dominującej reguły użytkownika
    /// (patrz [`resolve_color`]) w momencie powstania kropki — utrwalony raz,
    /// żeby kolor nie "skakał" w trakcie zaniku, nawet jeśli użytkownik w
    /// międzyczasie zmieni reguły w ustawieniach.
    pub color_rgb: (u8, u8, u8),
}

/// Czas życia pojedynczego wskaźnika kierunku (onsetu) na tarczy — niezależny
/// od `decay_secs` (który dotyczy kropek kanałów): onsety mają być krótkim,
/// wyraźnym "błyskiem" pokazującym POJEDYNCZE zdarzenie, nie długo zanikającym
/// echem.
const DIRECTION_MARKER_LIFETIME_SECS: f64 = 1.0;

/// Bezpiecznik pamięci — nie funkcjonalny limit widoczności (ten wynika z
/// naturalnego zaniku po [`DIRECTION_MARKER_LIFETIME_SECS`]), tylko górna
/// granica na wypadek nierealistycznie szybkiego strumienia zdarzeń.
const MAX_DIRECTION_MARKERS: usize = 64;

/// Pojedynczy, niezależny wskaźnik kierunku źródła dźwięku — jeden na każde
/// wykryte zdarzenie (onset, patrz [`crate::sound::direction::detect_onsets`]),
/// z WŁASNYM płynnym kątem i kolorem. W przeciwieństwie do dawnego,
/// pojedynczego "beacona" (suma wektorowa wszystkich kanałów naraz), wiele
/// jednoczesnych źródeł dostaje więc wiele niezależnych wskaźników zamiast
/// jednego mylącego/znoszącego się wektora.
#[derive(Clone, Copy, Debug)]
pub struct DirectionMarker {
    /// Czas powstania wskaźnika (sekundy, `egui::InputState::time`) — nadany
    /// TU, w warstwie UI (ma dostęp do `now`), nie w warstwie audio.
    pub spawned_at: f64,
    pub angle_deg: f32,
    pub intensity: f32,
    pub color_rgb: (u8, u8, u8),
}

/// Stan radaru utrzymywany między klatkami: historia kropek per kanał,
/// wygładzona energia (do detekcji transientów), osobny wskaźnik LFE,
/// wygładzone poziomy stereo L/R i aktywne wskaźniki kierunku (onsety).
#[derive(Default)]
pub struct RadarState {
    dots: Vec<VecDeque<ChannelDot>>,
    ema: Vec<f32>,
    last_spawn: f64,
    /// Wygładzony poziom LFE do rysowania osobnego wskaźnika niskich częstotliwości.
    pub lfe_display: f32,
    /// Kolor wskaźnika LFE wg jego własnej dominującej reguły (LFE to kanał
    /// audio jak każdy inny, więc też przechodzi przez `resolve_color`).
    pub lfe_color_rgb: (u8, u8, u8),
    /// Wygładzony poziom kanału lewego (stereo).
    pub stereo_left_display: f32,
    /// Kolor wskaźnika lewego kanału.
    pub stereo_left_color_rgb: (u8, u8, u8),
    /// Historia energii kanału lewego (ostatnie 3 ramki dla warstwowania).
    stereo_left_history: VecDeque<f32>,
    /// Wygładzony poziom kanału prawego (stereo).
    pub stereo_right_display: f32,
    /// Kolor wskaźnika prawego kanału.
    pub stereo_right_color_rgb: (u8, u8, u8),
    /// Historia energii kanału prawego (ostatnie 3 ramki dla warstwowania).
    stereo_right_history: VecDeque<f32>,
    direction_markers: VecDeque<DirectionMarker>,
}

impl RadarState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Doprowadza wewnętrzne bufory do rozmiaru bieżącego układu kanałów.
    fn ensure_capacity(&mut self, channels: usize) {
        if self.dots.len() != channels {
            self.dots = (0..channels).map(|_| VecDeque::new()).collect();
            self.ema = vec![0.0; channels];
        }
    }

    /// Aktualizuje stan radaru na podstawie aktualnych poziomów RMS kanałów.
    /// `now` to czas w sekundach (monotoniczny, np. `ctx.input(|i| i.time)`).
    /// Nowe kropki są dodawane co najwyżej co [`SPAWN_INTERVAL_SECS`],
    /// niezależnie od tego, jak często wywoływana jest ta funkcja — dzięki
    /// temu odświeżanie UI (30-60 FPS) jest niezależne od tempa analizy audio.
    /// `channel_rule_energies[kanał][reguła]` to energia w każdej regule
    /// kolorowania (`rules`, patrz [`crate::sound::spectrum::ColorRule`]) per
    /// kanał. Może być krótsza niż `channels` (np. scenariusze testowe, które
    /// jej nie liczą) — brakujące kanały dostają [`FALLBACK_DOT_RGB`].
    /// `decay_secs` (z [`RadarConfig`]) decyduje ile kropek trzeba trzymać
    /// naraz, żeby żadna nie zniknęła, zanim naturalnie wygaśnie — patrz
    /// [`MIN_DOTS_PER_CHANNEL`]/[`MAX_DOTS_PER_CHANNEL`]. `sensitivity`
    /// (`RadarConfig::radar_sensitivity`) to mnożnik RMS przed zamianą na
    /// odległość/rozmiar kropki — patrz pole w `RadarConfig` po co.
    pub fn update(
        &mut self,
        now: f64,
        channels: usize,
        channel_rms: &[f32],
        channel_rule_energies: &[Vec<f32>],
        rules: &[ColorRule],
        decay_secs: f32,
        sensitivity: f32,
        direction_events: &[DirectionEvent],
    ) {
        self.ensure_capacity(channels);

        for event in direction_events {
            self.direction_markers.push_back(DirectionMarker {
                spawned_at: now,
                angle_deg: event.angle_deg,
                intensity: event.intensity,
                color_rgb: event.color_rgb,
            });
            while self.direction_markers.len() > MAX_DIRECTION_MARKERS {
                self.direction_markers.pop_front();
            }
        }

        if now - self.last_spawn < SPAWN_INTERVAL_SECS {
            return;
        }
        self.last_spawn = now;

        // Tyle kropek trzeba trzymać naraz, żeby przy tempie spawnu
        // [`SPAWN_INTERVAL_SECS`] żadna nie została usunięta przed upływem
        // swojego naturalnego czasu zaniku (echo ma być widoczne przez CAŁE
        // `decay_secs`, nie tylko przez ostatnie kilka ramek).
        let max_dots = ((decay_secs as f64 / SPAWN_INTERVAL_SECS).ceil() as usize)
            .clamp(MIN_DOTS_PER_CHANNEL, MAX_DOTS_PER_CHANNEL);

        for (i, dots) in self.dots.iter_mut().enumerate() {
            let energy = (channel_rms.get(i).copied().unwrap_or(0.0) * sensitivity).clamp(0.0, 1.0);
            let role = channel_layout(channels).get(i).copied();

            let energies = channel_rule_energies.get(i).map(Vec::as_slice).unwrap_or(&[]);

            if role == Some(ChannelRole::Lfe) {
                self.lfe_display = self.lfe_display * 0.8 + energy * 0.2;
                self.lfe_color_rgb = resolve_color(rules, energies);
                continue;
            }

            // Stereo L/R wskaźniki (dla 2.0 i większych konfiguracji)
            if role == Some(ChannelRole::FrontLeft) {
                self.stereo_left_display = self.stereo_left_display * 0.8 + energy * 0.2;
                self.stereo_left_color_rgb = resolve_color(rules, energies);
                self.stereo_left_history.push_back(energy);
                while self.stereo_left_history.len() > 3 {
                    self.stereo_left_history.pop_front();
                }
            } else if role == Some(ChannelRole::FrontRight) {
                self.stereo_right_display = self.stereo_right_display * 0.8 + energy * 0.2;
                self.stereo_right_color_rgb = resolve_color(rules, energies);
                self.stereo_right_history.push_back(energy);
                while self.stereo_right_history.len() > 3 {
                    self.stereo_right_history.pop_front();
                }
            }

            let prev_ema = self.ema[i];
            let transient = energy > prev_ema * TRANSIENT_RATIO + 0.05;
            self.ema[i] = prev_ema * 0.9 + energy * 0.1;

            if energy < 0.01 {
                continue;
            }

            let color_rgb = resolve_color(rules, energies);

            dots.push_back(ChannelDot { spawned_at: now, energy, transient, color_rgb });
            while dots.len() > max_dots {
                dots.pop_front();
            }
        }
    }

    /// Usuwa kropki starsze niż `decay_secs` (porządkuje pamięć dla kanałów
    /// nieaktywnych od dawna).
    fn prune(&mut self, now: f64, decay_secs: f32) {
        for dots in &mut self.dots {
            while matches!(dots.front(), Some(d) if now - d.spawned_at > decay_secs as f64) {
                dots.pop_front();
            }
        }
        while matches!(self.direction_markers.front(), Some(m) if now - m.spawned_at > DIRECTION_MARKER_LIFETIME_SECS) {
            self.direction_markers.pop_front();
        }
    }
}

// ── Główna funkcja rysująca ───────────────────────────────────────────────────

const LABELS: [&str; 4] = ["PRZÓD", "PRAWO", "TYŁ", "LEWO"];

/// Rysuje minimalistyczny radar audio: tło, okręgi skali, krzyż kierunków,
/// opcjonalne sektory 7.1, kropki kanałów (z zanikiem w czasie), osobny
/// wskaźnik LFE i niezależne wskaźniki kierunku (jeden na każde wykryte
/// zdarzenie dźwiękowe — onset, patrz [`DirectionMarker`]). `now` to bieżący
/// czas w sekundach (do liczenia zaniku).
pub fn draw_radar<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    cfg: &RadarConfig,
    state: &mut RadarState,
    channels: usize,
    now: f64,
) {
    use crate::sound::devices::VisualizationMode;

    state.prune(now, cfg.decay_secs);

    // Sprawdzamy tryb wyświetlania
    let show_radar = matches!(cfg.visualization_mode, VisualizationMode::OnlyRadar | VisualizationMode::Both);
    let show_bars = matches!(cfg.visualization_mode, VisualizationMode::OnlyBars | VisualizationMode::Both);

    // Jeśli nic się nie wyświetla, wyjdź wcześnie
    if !show_radar && !show_bars {
        return;
    }

    let r = cfg.radius;
    let fg = cfg.fg_alpha;
    let hc = cfg.high_contrast;
    // Mnożnik grubości WSZYSTKICH linii poniżej (okręgi, krzyż, sektory,
    // obrysy) — `1.0` odtwarza dawne, sztywne szerokości.
    let lt = cfg.line_thickness.max(0.0);

    // Paleta: kolorowa (domyślna, zielono-cyjanowa) albo monochromatyczna
    // (wysoki kontrast) — rozróżnienie kropek zawsze przez kąt/rozmiar/jasność,
    // nigdy wyłącznie przez kolor.
    let (ring_rgb, cross_rgb, dot_rgb, bg_rgb) = if hc {
        ((255, 255, 255), (255, 255, 255), (255, 255, 255), (0, 0, 0))
    } else {
        ((0, 255, 157), (0, 255, 157), (0, 220, 220), (0, 8, 16))
    };

    let c_ring   = Color32::from_rgba_unmultiplied(ring_rgb.0, ring_rgb.1, ring_rgb.2, 80);
    let c_inner  = Color32::from_rgba_unmultiplied(ring_rgb.0, ring_rgb.1, ring_rgb.2, 35);
    let c_cross  = Color32::from_rgba_unmultiplied(cross_rgb.0, cross_rgb.1, cross_rgb.2, 50);
    let c_center = Color32::from_rgba_unmultiplied(cross_rgb.0, cross_rgb.1, cross_rgb.2, 200);
    let c_bg     = Color32::from_rgba_unmultiplied(bg_rgb.0, bg_rgb.1, bg_rgb.2, (cfg.bg_alpha * 255.0) as u8);

    let layout = channel_layout(channels);

    if show_radar {
        canvas.circle_filled(center, r, c_bg);

        // 3 koncentryczne okręgi = skala intensywności (słaby/średni/mocny), nie
        // fizyczna odległość.
        for frac in [1.0f32, 0.66, 0.33] {
            let color = if frac == 1.0 { with_alpha(c_ring, fg) } else { with_alpha(c_inner, fg) };
            let width = (if frac == 1.0 { 1.5 } else { 0.8 }) * lt;
            canvas.circle_stroke(center, r * frac, Stroke::new(width, color));
        }

        // Krzyż kierunków.
        let cross = Stroke::new(0.8 * lt, with_alpha(c_cross, fg));
        canvas.line_segment([center - Vec2::new(0.0, r), center + Vec2::new(0.0, r)], cross);
        canvas.line_segment([center - Vec2::new(r, 0.0), center + Vec2::new(r, 0.0)], cross);

        // Bardzo delikatne linie sektorów co 45° — tylko w układzie 7.1, gdzie
        // każdy z 8 kierunków ma osobny kanał.
        if channels == 8 {
            let sector = Stroke::new(0.5 * lt, with_alpha(c_cross, fg * 0.4));
            for k in 0..8 {
                let a = (k as f32) * std::f32::consts::FRAC_PI_4 - std::f32::consts::FRAC_PI_2;
                let p = center + Vec2::new(a.cos(), a.sin()) * r;
                canvas.line_segment([center, p], sector);
            }
        }

        canvas.circle_filled(center, 2.5, with_alpha(c_center, fg));

        // Etykiety kierunków (tryb z etykietami) — bez nich radar jest w pełni minimalny.
        if cfg.show_labels {
            let font = FontId::proportional((r * 0.09).clamp(9.0, 14.0));
            let label_color = with_alpha(c_ring, fg);
            let positions = [
                center - Vec2::new(0.0, r + 12.0), // PRZÓD
                center + Vec2::new(r + 6.0, 0.0),  // PRAWO
                center + Vec2::new(0.0, r + 12.0), // TYŁ
                center - Vec2::new(r + 6.0, 0.0),  // LEWO
            ];
            for (label, pos) in LABELS.iter().zip(positions.iter()) {
                canvas.text(*pos, egui::Align2::CENTER_CENTER, *label, font.clone(), label_color);
            }
        }

        // Kropki kanałów
        for (i, dots) in state.dots.iter().enumerate() {
            let Some(role) = layout.get(i).copied() else { continue };
            let Some(angle_deg) = role.radar_angle_degrees() else { continue }; // LFE ma osobny wskaźnik
            let angle = angle_deg.to_radians() - std::f32::consts::FRAC_PI_2;

            for dot in dots {
                let age = (now - dot.spawned_at) as f32;
                if age < 0.0 || age > cfg.decay_secs {
                    continue;
                }
                let life = 1.0 - (age / cfg.decay_secs).clamp(0.0, 1.0);
                // Zanik przyspiesza pod koniec życia kropki (nie liniowo).
                let alpha_frac = life.powf(1.6);

                let dot_r = (dot.energy * r).max(6.0).min(r);
                let x = center.x + dot_r * angle.cos();
                let y = center.y + dot_r * angle.sin();

                let size = (2.0 + dot.energy * 4.0) * (0.5 + 0.5 * life);
                let alpha = (dot.energy * 180.0 + 50.0) * fg * alpha_frac;
                let pos = Pos2::new(x, y);

                // Kolor koduje "typ" dźwięku wg reguł użytkownika (patrz
                // `resolve_color`, ustalone raz przy powstaniu kropki). W trybie
                // wysokiego kontrastu wyłączamy to i wracamy do jednej barwy, żeby
                // nie polegać wyłącznie na kolorze (rozróżnienie zostaje przez
                // kąt/rozmiar/kształt).
                let rgb = if hc { dot_rgb } else { dot.color_rgb };

                // Styl kropki zmienia się wraz z jej wiekiem, żeby powolne
                // wygasanie dźwięku było czytelne nie tylko przez przezroczystość,
                // ale przez samą formę: świeża kropka ma pełne wypełnienie i
                // delikatną poświatę, w połowie życia traci poświatę, a tuż przed
                // zniknięciem "rozpuszcza się" w pusty, kurczący się pierścień.
                if life > 0.66 {
                    let glow_alpha = (alpha * 0.35).clamp(0.0, 255.0) as u8;
                    let glow_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, glow_alpha);
                    canvas.circle_filled(pos, size * 2.0, glow_color);
                    let color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha.clamp(0.0, 255.0) as u8);
                    canvas.circle_filled(pos, size, color);
                } else if life > 0.33 {
                    let color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha.clamp(0.0, 255.0) as u8);
                    canvas.circle_filled(pos, size, color);
                } else {
                    // Mocniejszy kontur niż wypełnienie miałoby przy tej alpha,
                    // żeby cienki pierścień był wciąż widoczny tuż przed zniknięciem.
                    let ring_alpha = (alpha * 1.6).clamp(0.0, 255.0) as u8;
                    let stroke_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, ring_alpha);
                    canvas.circle_stroke(pos, size.max(1.5), Stroke::new(1.0 * lt, stroke_color));
                }

                // Nagły skok energii: krótki, gasnący pierścień wokół kropki.
                if dot.transient && age < 0.25 {
                    let ring_life = 1.0 - (age / 0.25);
                    let ring_r = size + 3.0 + (0.25 - age) * 40.0;
                    let ring_alpha = (200.0 * ring_life * fg) as u8;
                    let ring_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, ring_alpha);
                    canvas.circle_stroke(Pos2::new(x, y), ring_r, Stroke::new(1.5 * lt, ring_color));
                }
            }
        }

        // Wskaźnik LFE — osobny, poza tarczą
        if layout.contains(&ChannelRole::Lfe) {
            let lfe_rgb = if hc { dot_rgb } else { state.lfe_color_rgb };
            draw_lfe_indicator(canvas, center, r, state.lfe_display, fg, lfe_rgb);
        }

        // Niezależne wskaźniki kierunku
        for marker in &state.direction_markers {
            let age = (now - marker.spawned_at) as f32;
            if age < 0.0 || age > DIRECTION_MARKER_LIFETIME_SECS as f32 {
                continue;
            }
            let life = 1.0 - (age / DIRECTION_MARKER_LIFETIME_SECS as f32).clamp(0.0, 1.0);
            let alpha_frac = life.powf(1.6);

            let beacon_rgb = if hc { (255, 255, 255) } else { marker.color_rgb };
            let angle = marker.angle_deg.to_radians() - std::f32::consts::FRAC_PI_2;
            let dir = Vec2::new(angle.cos(), angle.sin());
            let perp = Vec2::new(-dir.y, dir.x);
            let intensity = marker.intensity.clamp(0.0, 1.0);

            let marker_r = (intensity * r).clamp(10.0, r);
            let pos = center + dir * marker_r;

            crate::dlog!(
                "[marker-pos] angle={:.1}° age={:.2}s life={:.2} x={:.1} y={:.1} (center=({:.1},{:.1}))",
                marker.angle_deg,
                age,
                life,
                pos.x,
                pos.y,
                center.x,
                center.y
            );

            let alpha = ((intensity * 0.6 + 0.4) * 255.0 * fg * alpha_frac).clamp(0.0, 255.0) as u8;
            let beacon_color =
                Color32::from_rgba_unmultiplied(beacon_rgb.0, beacon_rgb.1, beacon_rgb.2, alpha);
            let glow_alpha = ((alpha as f32) * 0.3) as u8;
            let glow_color =
                Color32::from_rgba_unmultiplied(beacon_rgb.0, beacon_rgb.1, beacon_rgb.2, glow_alpha);

            let head_len = 14.0 + intensity * 10.0;
            let head_w = 9.0 + intensity * 6.0;
            let tip = pos + dir * head_len * 0.6;
            let base_center = pos - dir * head_len * 0.4;
            let left = base_center + perp * head_w * 0.5;
            let right = base_center - perp * head_w * 0.5;

            canvas.line_segment([center, base_center], Stroke::new(2.2 * lt, beacon_color));
            canvas.circle_filled(pos, head_w * 1.1, glow_color);
            canvas.triangle_filled([tip, left, right], beacon_color);
            canvas.circle_stroke(pos, head_w * 1.1, Stroke::new(1.0 * lt, beacon_color));
        }
    }

    // Wskaźniki stereo L/R — rysują się gdy show_bars
    if show_bars {
        if layout.contains(&ChannelRole::FrontLeft) || layout.contains(&ChannelRole::FrontRight) {
            draw_stereo_indicators(
                canvas,
                center,
                r,
                &state.stereo_left_history,
                &state.stereo_right_history,
                fg,
            );
        }
    }
}

fn draw_lfe_indicator<C: RadarCanvas>(canvas: &C, center: Pos2, r: f32, level: f32, fg: f32, rgb: (u8, u8, u8)) {
    let w = (r * 0.5).clamp(30.0, 70.0);
    let h = 6.0;
    let top_left = Pos2::new(center.x - w / 2.0, center.y + r + 14.0);
    let bg = Color32::from_rgba_unmultiplied(255, 255, 255, (30.0 * fg) as u8);
    canvas.rect_filled(egui::Rect::from_min_size(top_left, Vec2::new(w, h)), 2.0, bg);

    let fill_w = w * level.clamp(0.0, 1.0);
    let fill = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, (220.0 * fg) as u8);
    canvas.rect_filled(egui::Rect::from_min_size(top_left, Vec2::new(fill_w, h)), 2.0, fill);
}

fn draw_stereo_indicators<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    r: f32,
    left_history: &VecDeque<f32>,
    right_history: &VecDeque<f32>,
    fg: f32,
) {
    let bar_width = (r * 0.12).clamp(8.0, 16.0);
    let bar_height = (r * 1.2).clamp(100.0, 250.0);
    let spacing = (r * 0.4).clamp(30.0, 70.0);

    // Kolory warstw dla L: niebieski -> czerwony -> niebieski
    let left_colors = [
        (0, 100, 200),    // Niebieski (warstwa 0)
        (200, 50, 50),    // Czerwony (warstwa 1)
        (0, 100, 200),    // Niebieski (warstwa 2)
    ];

    // Kolory warstw dla R: zielony -> żółty -> zielony
    let right_colors = [
        (50, 150, 50),    // Zielony (warstwa 0)
        (200, 180, 30),   // Żółty (warstwa 1)
        (50, 150, 50),    // Zielony (warstwa 2)
    ];

    // Lewy słupek (L) — po lewej stronie radaru
    let left_x = center.x - spacing - bar_width;
    let left_top = center.y - bar_height / 2.0;
    let bg = Color32::from_rgba_unmultiplied(255, 255, 255, (30.0 * fg) as u8);
    canvas.rect_filled(
        egui::Rect::from_min_size(Pos2::new(left_x, left_top), Vec2::new(bar_width, bar_height)),
        2.0,
        bg,
    );

    // Rysuj warstwy dla lewego słupka
    let mut accumulated_height = 0.0;
    for (layer_idx, &energy) in left_history.iter().enumerate() {
        let layer_height = bar_height * energy.clamp(0.0, 1.0);
        let (r_color, g_color, b_color) = left_colors[layer_idx % left_colors.len()];
        let color = Color32::from_rgba_unmultiplied(r_color, g_color, b_color, (220.0 * fg) as u8);

        canvas.rect_filled(
            egui::Rect::from_min_size(
                Pos2::new(left_x, left_top + bar_height - accumulated_height - layer_height),
                Vec2::new(bar_width, layer_height),
            ),
            2.0,
            color,
        );
        accumulated_height += layer_height;
    }

    // Prawy słupek (R) — po prawej stronie radaru
    let right_x = center.x + spacing;
    let right_top = center.y - bar_height / 2.0;
    canvas.rect_filled(
        egui::Rect::from_min_size(Pos2::new(right_x, right_top), Vec2::new(bar_width, bar_height)),
        2.0,
        bg,
    );

    // Rysuj warstwy dla prawego słupka
    accumulated_height = 0.0;
    for (layer_idx, &energy) in right_history.iter().enumerate() {
        let layer_height = bar_height * energy.clamp(0.0, 1.0);
        let (r_color, g_color, b_color) = right_colors[layer_idx % right_colors.len()];
        let color = Color32::from_rgba_unmultiplied(r_color, g_color, b_color, (220.0 * fg) as u8);

        canvas.rect_filled(
            egui::Rect::from_min_size(
                Pos2::new(right_x, right_top + bar_height - accumulated_height - layer_height),
                Vec2::new(bar_width, layer_height),
            ),
            2.0,
            color,
        );
        accumulated_height += layer_height;
    }

    // Etykiety L i R
    let font = FontId::proportional((bar_width * 1.5).clamp(8.0, 12.0));
    let label_color = Color32::from_rgba_unmultiplied(255, 255, 255, (180.0 * fg) as u8);
    canvas.text(
        Pos2::new(left_x + bar_width / 2.0, left_top - 12.0),
        egui::Align2::CENTER_BOTTOM,
        "L",
        font.clone(),
        label_color,
    );
    canvas.text(
        Pos2::new(right_x + bar_width / 2.0, right_top - 12.0),
        egui::Align2::CENTER_BOTTOM,
        "R",
        font.clone(),
        label_color,
    );

    // Wskaźnik porównania L vs R — pasek między słupkami
    let left_energy = left_history.back().copied().unwrap_or(0.0);
    let right_energy = right_history.back().copied().unwrap_or(0.0);

    let bar_center_x = center.x;
    let indicator_y = center.y + bar_height / 2.0 + 20.0;
    let indicator_width = spacing * 1.5;
    let indicator_height = 12.0;

    // Tło wskaźnika
    let indicator_bg = Color32::from_rgba_unmultiplied(60, 60, 60, (100.0 * fg) as u8);
    canvas.rect_filled(
        egui::Rect::from_min_size(
            Pos2::new(bar_center_x - indicator_width / 2.0, indicator_y),
            Vec2::new(indicator_width, indicator_height),
        ),
        3.0,
        indicator_bg,
    );

    // Oblicz którą stronę wypełnić
    let energy_diff = (left_energy - right_energy).clamp(-1.0, 1.0);
    let fill_center_offset = (indicator_width / 2.0) * energy_diff;

    // Wypełnienie wskaźnika (lewe lub prawe, w zależności od L vs R)
    if energy_diff > 0.05 {
        // L > R — wypełniamy od lewej
        let fill_width = (indicator_width / 2.0) * (energy_diff / 1.0);
        let left_color = Color32::from_rgba_unmultiplied(0, 100, 200, (220.0 * fg) as u8);
        canvas.rect_filled(
            egui::Rect::from_min_size(
                Pos2::new(bar_center_x - indicator_width / 2.0, indicator_y),
                Vec2::new(fill_width, indicator_height),
            ),
            3.0,
            left_color,
        );
    } else if energy_diff < -0.05 {
        // R > L — wypełniamy od prawej
        let fill_width = (indicator_width / 2.0) * (-energy_diff / 1.0);
        let right_color = Color32::from_rgba_unmultiplied(50, 150, 50, (220.0 * fg) as u8);
        canvas.rect_filled(
            egui::Rect::from_min_size(
                Pos2::new(bar_center_x + indicator_width / 2.0 - fill_width, indicator_y),
                Vec2::new(fill_width, indicator_height),
            ),
            3.0,
            right_color,
        );
    } else {
        // Prawie równe — wypełniamy środek
        let center_fill = 8.0;
        let equal_color = Color32::from_rgba_unmultiplied(150, 150, 150, (220.0 * fg) as u8);
        canvas.rect_filled(
            egui::Rect::from_min_size(
                Pos2::new(bar_center_x - center_fill / 2.0, indicator_y),
                Vec2::new(center_fill, indicator_height),
            ),
            3.0,
            equal_color,
        );
    }
}

// ── Drugi tryb wizualizacji: poświaty na krawędzi ekranu ──────────────────────
//
// Peryferyjna podpowiedź kierunku w stylu "screen edge indicator" ze
// strzelanek (np. wskaźnik obrażeń) — OBOK tarczy radaru, nie zamiast niej.
// W przeciwieństwie do wskaźników na tarczy (przypięte do promienia wokół
// `center`, widoczne tylko gdy patrzy się wprost na mały radar w rogu), ta
// poświata pojawia się na faktycznej krawędzi całego ekranu, więc jest
// czytelna kątem oka niezależnie od tego, gdzie akurat skupiony jest wzrok.
// Wymaga overlay rozciągniętego na cały ekran (patrz `RadarConfig::edge_glow_enabled`
// i `HearRadarApp::ui_overlay_contents`) — bez tego `screen_rect` byłby tylko
// małym kwadratem wokół tarczy, a "krawędź ekranu" nie miałaby sensu.

/// Jak daleko w głąb ekranu sięga najszersza (najbardziej zewnętrzna) warstwa
/// poświaty.
const EDGE_GLOW_MAX_DEPTH: f32 = 160.0;
/// Jak długi (wzdłuż krawędzi) jest najszerszy pas poświaty.
const EDGE_GLOW_MAX_SPAN: f32 = 320.0;
/// Kolejne warstwy poświaty od najbardziej zewnętrznej (największa, najbardziej
/// przezroczysta — efekt miękkiego gasnięcia w głąb ekranu) do najbardziej
/// wewnętrznej (najmniejsza, najintensywniejsza, tuż przy samej krawędzi).
/// `(ułamek rozmiaru, ułamek alfy)`.
const EDGE_GLOW_LAYERS: [(f32, f32); 4] = [(1.0, 0.10), (0.7, 0.22), (0.45, 0.4), (0.22, 0.7)];

/// Zwraca kierunek jednostkowy na ekranie dla danego kąta (konwencja jak w
/// [`draw_radar`]: `0°` = góra ekranu/przód, kąty rosną zgodnie ze wskazówkami
/// zegara).
fn screen_direction(angle_deg: f32) -> Vec2 {
    let angle = angle_deg.to_radians() - std::f32::consts::FRAC_PI_2;
    Vec2::new(angle.cos(), angle.sin())
}

/// Znajduje punkt na obwodzie `screen_rect`, w którym promień wystrzelony ze
/// środka ekranu w kierunku `dir` przecina krawędź — klasyczna metoda
/// przecięcia promień/prostokąt (porównanie parametrów `t` dla obu par
/// krawędzi, mniejszy wygrywa). Zwraca też, czy trafioną krawędzią jest
/// lewa/prawa (pionowa) czy góra/dół (pozioma) — to decyduje o orientacji
/// narysowanego pasa poświaty.
fn edge_intersection(screen_rect: Rect, dir: Vec2) -> (Pos2, bool) {
    let half = screen_rect.size() * 0.5;
    let t_x = if dir.x.abs() > 1e-4 { half.x / dir.x.abs() } else { f32::INFINITY };
    let t_y = if dir.y.abs() > 1e-4 { half.y / dir.y.abs() } else { f32::INFINITY };
    let t = t_x.min(t_y);
    let point = screen_rect.center() + dir * t;
    (point, t_x <= t_y)
}

/// Rysuje jedną, wielowarstwową poświatę przy krawędzi ekranu najbliższej
/// `point` — miękkie gasnięcie w głąb ekranu, symulowane kilkoma nałożonymi,
/// coraz mniejszymi i coraz bardziej nieprzezroczystymi pasami (canvas nie ma
/// prawdziwego gradientu), zamiast jednego twardego prostokąta.
fn draw_edge_glow_at<C: RadarCanvas>(
    canvas: &C,
    screen_rect: Rect,
    point: Pos2,
    vertical_edge: bool,
    rgb: (u8, u8, u8),
    strength: f32,
    fg: f32,
) {
    for &(size_frac, alpha_frac) in &EDGE_GLOW_LAYERS {
        let depth = EDGE_GLOW_MAX_DEPTH * size_frac;
        let span = EDGE_GLOW_MAX_SPAN * size_frac;
        let alpha = (255.0 * alpha_frac * strength * fg).clamp(0.0, 255.0) as u8;
        if alpha == 0 {
            continue;
        }
        let color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha);

        let rect = if vertical_edge {
            let at_left = (point.x - screen_rect.left()).abs() <= (screen_rect.right() - point.x).abs();
            let x0 = if at_left { screen_rect.left() } else { screen_rect.right() - depth };
            Rect::from_min_size(
                Pos2::new(x0, point.y - span * 0.5),
                Vec2::new(depth, span),
            )
        } else {
            let at_top = (point.y - screen_rect.top()).abs() <= (screen_rect.bottom() - point.y).abs();
            let y0 = if at_top { screen_rect.top() } else { screen_rect.bottom() - depth };
            Rect::from_min_size(
                Pos2::new(point.x - span * 0.5, y0),
                Vec2::new(span, depth),
            )
        };
        canvas.rect_filled(rect, depth * 0.25, color);
    }
}

/// Rysuje peryferyjne poświaty kierunku na krawędziach `screen_rect` (patrz
/// doc-komentarz sekcji powyżej) — jedna na każdy aktywny wskaźnik kierunku w
/// `state` ([`DirectionMarker`], ten sam stan co markery na tarczy radaru w
/// [`draw_radar`]), więc oba tryby wizualizacji zawsze pokazują dokładnie te
/// same zdarzenia, tylko w innej formie. Bez efektu, gdy
/// [`RadarConfig::edge_glow_enabled`] jest wyłączone.
pub fn draw_edge_glow_indicators<C: RadarCanvas>(
    canvas: &C,
    screen_rect: Rect,
    state: &RadarState,
    now: f64,
    cfg: &RadarConfig,
) {
    if !cfg.edge_glow_enabled {
        return;
    }
    let hc = cfg.high_contrast;
    let fg = cfg.fg_alpha;

    for marker in &state.direction_markers {
        let age = (now - marker.spawned_at) as f32;
        if age < 0.0 || age > DIRECTION_MARKER_LIFETIME_SECS as f32 {
            continue;
        }
        let life = 1.0 - (age / DIRECTION_MARKER_LIFETIME_SECS as f32).clamp(0.0, 1.0);
        let alpha_frac = life.powf(1.6);
        let intensity = marker.intensity.clamp(0.0, 1.0);
        let strength = (intensity * 0.7 + 0.3) * alpha_frac;

        let rgb = if hc { (255, 255, 255) } else { marker.color_rgb };
        let dir = screen_direction(marker.angle_deg);
        let (point, vertical_edge) = edge_intersection(screen_rect, dir);
        draw_edge_glow_at(canvas, screen_rect, point, vertical_edge, rgb, strength, fg);
    }
}
