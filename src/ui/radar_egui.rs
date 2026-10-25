use crate::sound::devices::{DirectionEvent, QueuedDirectionEvent, QueuedLayerHit, RadarConfig};
use crate::sound::direction::{channel_layout, layout_name, ChannelRole};
use crate::sound::footsteps::StepTracker;
use crate::sound::layers::{Layer, LayerBars};
use crate::sound::spectrum::{resolve_color, ColorRule};
use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use std::collections::VecDeque;
use std::time::Instant;

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
    Sine        = 4,
    SinglePulse = 5,
    /// Dwa DOKŁADNIE przeciwstawne kanały (SL/SR, 180° od siebie), pulsujące
    /// RÓWNOCZEŚNIE z podobną energią — dokładny przypadek, w którym dawny,
    /// pojedynczy sumaryczny wskaźnik kierunku (suma wektorowa wszystkich
    /// kanałów) dawał `sum≈0` i mylącą,
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
        Self::Sine,
        Self::SinglePulse,
        Self::OpposingSources,
        Self::Tone,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off         => "🎙 Live audio",
            Self::Sine        => "∿ Sinusoida (rotacja)",
            Self::SinglePulse => "▲ Pojedynczy impuls (transient)",
            Self::OpposingSources => "⇄ Dwa przeciwstawne źródła naraz",
            Self::Tone        => "🎵 Ton o zadanym Hz (kolor pasma)",
        }
    }

    /// Zwraca symulowany RMS dla każdego z `channels` kanałów, w danej fazie.
    /// `seed` decyduje, który kanał dostaje cykliczny impuls w `SinglePulse` —
    /// ten sam `seed` zawsze da ten sam przebieg, więc łatwiej porównywać
    /// zmiany w renderowaniu.
    pub fn generate(self, channels: usize, phase: f32, seed: u32) -> Vec<f32> {
        (0..channels)
            .map(|i| {
                let t = i as f32 / channels.max(1) as f32;
                match self {
                    Self::Off         => 0.0,
                    Self::Sine        => ((phase - t * std::f32::consts::TAU).sin() * 0.5 + 0.5).clamp(0.0, 1.0),
                    Self::SinglePulse => {
                        // Krótki, cykliczny impuls na jednym kanale na raz — pokazuje
                        // detekcję transientów (większa/jaśniejsza kropka + pierścień).
                        let active = ((phase * 0.5) as usize + seed as usize) % channels;
                        let just_fired = (phase * 8.0).fract() < 0.15;
                        if i == active && just_fired { 1.0 } else { 0.02 }
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

/// Kąt, w którego zasięgu (± tyle stopni od kąta kanału) zdarzenie z detektora zapala
/// pierścień „transient” na tym kanale.
const TRANSIENT_ANGLE_DEG: f32 = 45.0;

#[derive(Clone, Copy, Debug)]
pub struct ChannelDot {
    /// Czas powstania kropki (sekundy, `egui::InputState::time`).
    pub spawned_at: f64,
    /// Energia (RMS) kanału w tej ramce, 0.0..=1.0.
    pub energy: f32,
    /// Czy w tym czasie detektor wykrył zdarzenie na tym kanale (pierścień „transient”).
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

/// Czas życia błysku kroku na łukach L/R i przód/tył. Krótszy niż echo kropek,
/// bo kroki chodzą w rytmie ~2–4 Hz i każdy ma być osobnym, czytelnym impulsem.
const STEP_MARKER_LIFETIME_SECS: f64 = 0.8;

/// Bezpiecznik pamięci dla kroków (jak [`MAX_DIRECTION_MARKERS`]).
const MAX_STEP_MARKERS: usize = 16;

/// Zdarzenie kierunku z chwilą wykrycia na osi UI (sekundy, jak `now`).
#[derive(Clone, Copy, Debug)]
pub struct TimedDirectionEvent {
    pub at: f64,
    pub event: DirectionEvent,
}

/// Zdarzenia z kolejki audio na osi UI. Wiek liczy się jedną miarą dla całej
/// partii, więc zdarzenia z tego samego callbacku dostają identyczną chwilę.
pub fn timed_from_queue(now: f64, queued: &[QueuedDirectionEvent]) -> Vec<TimedDirectionEvent> {
    let batch = Instant::now();
    queued
        .iter()
        .map(|q| TimedDirectionEvent {
            at: now - batch.saturating_duration_since(q.captured_at).as_secs_f64(),
            event: q.event,
        })
        .collect()
}

/// Zdarzenia syntetyczne (scenariusze testowe) pojawiają się od razu, w chwili `now`.
pub fn timed_now(now: f64, events: &[DirectionEvent]) -> Vec<TimedDirectionEvent> {
    events.iter().map(|&event| TimedDirectionEvent { at: now, event }).collect()
}

/// Jeden wykryty krok — patrz [`crate::sound::footsteps::StepTracker`].
#[derive(Clone, Copy, Debug)]
pub struct StepMarker {
    /// Czas wykrycia kroku (sekundy, `egui::InputState::time`).
    pub spawned_at: f64,
    /// Balans L/R uderzenia w stopniach: ujemny = lewa strona, dodatni = prawa
    /// (zakres ±45°, jak w [`crate::sound::direction::stereo_onset_angle_deg`]).
    pub pan_deg: f32,
    /// Siła uderzenia, 0.0..=1.0.
    pub intensity: f32,
    /// Przód (dodatni) / tył (ujemny), −1.0..=1.0. Liczone z nachylenia widma
    /// w chwili uderzenia względem linii bazowej, a nie z ciągłej średniej,
    /// żeby krok nie rozmywał się w tle.
    pub front_back: f32,
}

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
    /// Czy od ostatniej kropki kanału detektor wykrył na nim zdarzenie. Kolejna kropka dostaje
    /// wtedy pierścień „transient”.
    pending_transient: Vec<bool>,
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
    spatial_tilt_baseline: f32,
    spatial_mid_baseline: f32,
    spatial_front_back: f32,
    spatial_height: f32,
    spatial_energy: f32,
    /// Czy linia bazowa nachylenia widma została już ustawiona na pierwszym
    /// poprawnym odczycie (0.0 bywa poprawnym odczytem — tła ciemnego — więc nie
    /// można tego poznać z samej wartości).
    spatial_baseline_ready: bool,
    /// Rytm uderzeń — mówi, które onsety pasują do rytmu kroków.
    step_tracker: StepTracker,
    /// Wykryte kroki, najnowszy na końcu.
    steps: VecDeque<StepMarker>,
    /// Słupki warstw 3+3 (tryb `VisualizationMode::LayerBars`), patrz `sound::layers`.
    pub layer_bars: LayerBars,
}

/// Udział energii pasma `num` w sumie energii pasm `num` i `den`, liczony z
/// reguł pasm (po środku pasma każdej reguły). `None`, gdy obie strony są
/// zerowe, bo wtedy udział nie ma sensu.
fn band_share(rules: &[ColorRule], energies: &[f32], num: (f32, f32), den: (f32, f32)) -> Option<f32> {
    let band_sum = |(lo, hi): (f32, f32)| -> f32 {
        rules
            .iter()
            .zip(energies)
            .filter(|(r, _)| {
                let center = (r.freq_min + r.freq_max) * 0.5;
                center >= lo && center < hi
            })
            .map(|(_, e)| *e)
            .sum()
    };
    let n = band_sum(num);
    let total = n + band_sum(den);
    (total > 1e-6).then(|| n / total)
}

/// Wzmocnienie nachylenia widma na przód/tył i wysokość: ile odchylenia od
/// linii bazowej składa się na pełną długość łuku P/T.
const SPATIAL_GAIN: f32 = 6.0;

/// Udział wysokich pasm (`tilt`) i średnich (`mid`) w kanałach przednich L/R tej
/// ramki, jako pary (suma, liczba kanałów). Surowe, niewygładzone odczyty — dla
/// `update_spatial` i dla pomiaru pojedynczego kroku.
fn front_band_shares(
    channels: usize,
    rules: &[ColorRule],
    channel_rule_energies: &[Vec<f32>],
) -> ((f32, u32), (f32, u32)) {
    let mut tilt_sum = (0.0f32, 0u32);
    let mut mid_sum = (0.0f32, 0u32);
    for (i, role) in channel_layout(channels).iter().enumerate() {
        if !matches!(role, ChannelRole::FrontLeft | ChannelRole::FrontRight) {
            continue;
        }
        let energies = channel_rule_energies.get(i).map(Vec::as_slice).unwrap_or(&[]);
        if let Some(t) = band_share(rules, energies, (4000.0, 20_000.0), (0.0, 1000.0)) {
            tilt_sum.0 += t;
            tilt_sum.1 += 1;
        }
        if let Some(m) = band_share(rules, energies, (4000.0, 8000.0), (1000.0, 4000.0)) {
            mid_sum.0 += m;
            mid_sum.1 += 1;
        }
    }
    (tilt_sum, mid_sum)
}

impl RadarState {
    pub fn new() -> Self {
        Self::default()
    }


    /// Aktualna energia lewego i prawego kanału stereo (0..1), tak jak ją pokazują stare wskaźniki L/R.
    pub fn stereo_now(&self) -> (f32, f32) {
        (
            self.stereo_left_history.back().copied().unwrap_or(0.0),
            self.stereo_right_history.back().copied().unwrap_or(0.0),
        )
    }

    /// Zasila słupki warstw trafieniami z kolejki audio. Czas liczy się tak samo jak dla kropek.
    pub fn record_layer_hits(&mut self, now: f64, queued: &[QueuedLayerHit]) {
        let batch = Instant::now();
        for q in queued {
            let at = now - batch.saturating_duration_since(q.captured_at).as_secs_f64();
            self.layer_bars.record(at, q.hit);
        }
    }

    fn update_spatial(&mut self, tilt: (f32, u32), mid: (f32, u32)) {
        const BASELINE_RATE: f32 = 0.002;

        let energy = (self.stereo_left_display + self.stereo_right_display).min(1.0);
        self.spatial_energy = self.spatial_energy * 0.8 + energy * 0.2;
        if tilt.1 == 0 || mid.1 == 0 {
            self.spatial_front_back *= 0.8;
            self.spatial_height *= 0.8;
            return;
        }

        let tilt = tilt.0 / tilt.1 as f32;
        let mid = mid.0 / mid.1 as f32;
        if !self.spatial_baseline_ready {
            self.spatial_tilt_baseline = tilt;
            self.spatial_mid_baseline = mid;
            self.spatial_baseline_ready = true;
        }
        self.spatial_tilt_baseline += (tilt - self.spatial_tilt_baseline) * BASELINE_RATE;
        self.spatial_mid_baseline += (mid - self.spatial_mid_baseline) * BASELINE_RATE;

        let front_back = ((tilt - self.spatial_tilt_baseline) * SPATIAL_GAIN).clamp(-1.0, 1.0);
        let height = ((mid - self.spatial_mid_baseline) * SPATIAL_GAIN).clamp(-1.0, 1.0);
        self.spatial_front_back = self.spatial_front_back * 0.8 + front_back * 0.2;
        self.spatial_height = self.spatial_height * 0.8 + height * 0.2;
    }

    /// Zapisuje wykryty krok: balans L/R z najmocniejszego onsetu uderzenia oraz
    /// przód/tył z nachylenia widma tego uderzenia (`tilt`, z [`front_band_shares`])
    /// względem linii bazowej. Bez linii bazowej (start) przód/tył zostaje zerem.
    fn push_step(&mut self, now: f64, strongest: &DirectionEvent, tilt: (f32, u32)) {
        let front_back = if tilt.1 == 0 || !self.spatial_baseline_ready {
            0.0
        } else {
            ((tilt.0 / tilt.1 as f32 - self.spatial_tilt_baseline) * SPATIAL_GAIN).clamp(-1.0, 1.0)
        };
        self.steps.push_back(StepMarker {
            spawned_at: now,
            pan_deg: strongest.angle_deg,
            intensity: strongest.intensity.clamp(0.0, 1.0),
            front_back,
        });
        while self.steps.len() > MAX_STEP_MARKERS {
            self.steps.pop_front();
        }
    }

    /// Przód/tył do narysowania: najnowszy krok, wygaszany liniowo w czasie życia,
    /// a gdy żadnego kroku nie ma — ciągły estymator z `update_spatial`.
    fn displayed_front_back(&self, now: f64) -> f32 {
        match self.steps.back() {
            Some(step) => {
                let age = (now - step.spawned_at) as f32;
                step.front_back * (1.0 - age / STEP_MARKER_LIFETIME_SECS as f32).clamp(0.0, 1.0)
            }
            None => self.spatial_front_back,
        }
    }

    /// Doprowadza wewnętrzne bufory do rozmiaru bieżącego układu kanałów.
    fn ensure_capacity(&mut self, channels: usize) {
        if self.dots.len() != channels {
            self.dots = (0..channels).map(|_| VecDeque::new()).collect();
            self.ema = vec![0.0; channels];
            self.pending_transient = vec![false; channels];
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
        direction_events: &[TimedDirectionEvent],
    ) {
        self.ensure_capacity(channels);


        // Nachylenie widma kanałów przednich L/R w TEJ ramce — liczone przed bramką
        // czasu niżej, bo krok może przyjść między kolejnymi spawnami kropek, a jego
        // przód/tył trzeba wtedy mieć pod ręką.
        let (tilt_sum, mid_sum) = front_band_shares(channels, rules, channel_rule_energies);

        for timed in direction_events {
            self.direction_markers.push_back(DirectionMarker {
                spawned_at: timed.at,
                angle_deg: timed.event.angle_deg,
                intensity: timed.event.intensity,
                color_rgb: timed.event.color_rgb,
            });
            while self.direction_markers.len() > MAX_DIRECTION_MARKERS {
                self.direction_markers.pop_front();
            }
        }

        // Zdarzenia z detektora zapalają pierścień „transient” na kanałach, których dotyczą
        // (kąt zdarzenia blisko kąta głośnika). Kropka dostaje go przy najbliższym spawnie.
        let layout = channel_layout(channels);
        for timed in direction_events {
            for (i, role) in layout.iter().enumerate() {
                let Some(channel_angle) = role.radar_angle_degrees() else {
                    continue;
                };
                if (timed.event.angle_deg - channel_angle).abs() <= TRANSIENT_ANGLE_DEG {
                    if let Some(flag) = self.pending_transient.get_mut(i) {
                        *flag = true;
                    }
                }
            }
        }

        // Zdarzenia z jednego callbacku (ta sama chwila) to jedno uderzenie. To rytm
        // uderzeń decyduje, czy jest krokiem.
        for hit in direction_events.chunk_by(|a, b| a.at == b.at) {
            let at = hit[0].at;
            if self.step_tracker.observe_hit(at) {
                let strongest = hit
                    .iter()
                    .map(|timed| &timed.event)
                    .max_by(|a, b| a.intensity.total_cmp(&b.intensity));
                if let Some(strongest) = strongest {
                    self.push_step(at, strongest, tilt_sum);
                }
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

            // Pierścień „transient” z tego samego detektora co wskaźniki kierunku, nie z własnego skoku energii.
            let transient = std::mem::take(&mut self.pending_transient[i]);
            let prev_ema = self.ema[i];
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

        self.update_spatial(tilt_sum, mid_sum);
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
        while matches!(self.steps.front(), Some(s) if now - s.spawned_at > STEP_MARKER_LIFETIME_SECS) {
            self.steps.pop_front();
        }
    }
}

#[cfg(test)]
mod transient_tests {
    use super::*;

    fn timed(at: f64, angle_deg: f32) -> TimedDirectionEvent {
        TimedDirectionEvent {
            at,
            event: DirectionEvent {
                angle_deg,
                intensity: 0.5,
                color_rgb: (0, 0, 0),
            },
        }
    }

    /// Stare łuki L/R pokazują, po której stronie jest więcej dźwięku, nawet bez żadnych zdarzeń
    /// (ciągły dźwięk, np. silnik podczas jazdy).
    #[test]
    fn stereo_history_follows_continuous_sound_without_events() {
        let mut state = RadarState::new();
        for i in 0..300 {
            state.update(i as f64 * 0.016, 2, &[0.4, 0.1], &[vec![], vec![]], &[], 1.0, 1.0, &[]);
        }
        let left = state.stereo_left_history.back().copied().unwrap_or(0.0);
        let right = state.stereo_right_history.back().copied().unwrap_or(0.0);
        assert!(left > right + 0.2, "więcej z lewej: {left} / {right}");
    }

    /// Gdy jedna strona ma wyraźną przewagę, słabsza się chowa. Przy niemal równych obie są widoczne.
    #[test]
    fn weaker_side_is_hidden_only_with_a_clear_lead() {
        assert_eq!(visible_sides(0.6, 0.2, 0.05), [true, false], "lewa przeważa");
        assert_eq!(visible_sides(0.2, 0.6, 0.05), [false, true], "prawa przeważa");
        assert_eq!(visible_sides(0.5, 0.48, 0.05), [true, true], "prawie równo");
        assert_eq!(visible_sides(0.01, 0.02, 0.05), [true, true], "cisza");
    }

    /// Pierścień „transient” dostaje tylko kanał, na którym detektor wykrył zdarzenie.
    #[test]
    fn detector_event_marks_the_matching_channel_only() {
        let mut state = RadarState::new();
        // Zdarzenie z lewej (−45°, przedni lewy). Kanał 0 to FL, kanał 1 to FR.
        state.update(1.0, 2, &[0.5, 0.5], &[vec![], vec![]], &[], 1.0, 1.0, &[timed(1.0, -45.0)]);
        let left = state.dots[0].back().expect("kropka lewego kanału");
        let right = state.dots[1].back().expect("kropka prawego kanału");
        assert!(left.transient, "lewy kanał: zdarzenie z lewej");
        assert!(!right.transient, "prawy kanał bez zdarzenia");
    }
}

// ── Główna funkcja rysująca ───────────────────────────────────────────────────

/// Kolor obrysu dominującego kanału (łuk L/R i głośnik w pierścieniu). Czerwień
/// celowo nie jest w paletach użytkownika (patrz `RadarConfig::stereo_left_color`).
const DOMINANT_OUTLINE_RGB: (u8, u8, u8) = (255, 40, 40);

/// Ile energii (0.0..=1.0) jeden kanał musi przewyższać drugi, żeby zostać
/// uznanym za dominujący i dostać czerwony obrys.
const DOMINANCE_MARGIN: f32 = 0.05;

/// Jak bardzo lewy i prawy kanał (0.0..=1.0) mogą się różnić, żeby uznać je za
/// równe (L=R) i pokazać czerwoną obwódkę przodu.
const BALANCE_TOLERANCE: f32 = 0.03;

/// Poniżej tego poziomu obu kanałów równość L/R się nie liczy (cisza).
const BALANCE_MIN_LEVEL: f32 = 0.05;

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
    state.layer_bars.prune(now, cfg.decay_secs);

    if cfg.visualization_mode == VisualizationMode::LayerBars {
        draw_layer_bars(canvas, center, cfg, state, now);
        return;
    }

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

    let split_layouts = cfg.speaker_layout_view;

    if show_bars {
        if layout.contains(&ChannelRole::FrontLeft) || layout.contains(&ChannelRole::FrontRight) {
            draw_stereo_indicators(
                canvas,
                center,
                r,
                &state.stereo_left_history,
                &state.stereo_right_history,
                cfg.stereo_left_color,
                cfg.stereo_right_color,
                fg,
                lt,
                [true, true],
            );
        }
    }

    if cfg.spatial_audio_view && show_bars {
        draw_spatial_arcs(canvas, center, r, state, now, fg, lt);
        draw_step_flashes(canvas, center, r, state, now, fg, lt);
    }

    if split_layouts && show_radar {
        if channels > 2 {
            draw_speaker_ring(canvas, center, r, layout, &state.ema, fg, lt);
        }
        canvas.text(
            Pos2::new(center.x, center.y - r - 22.0),
            Align2::CENTER_CENTER,
            &layout_name(channels),
            FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(ring_rgb.0, ring_rgb.1, ring_rgb.2, (200.0 * fg) as u8),
        );
    }
}

/// Zestaw 3+3 w stylu łuków L/P: po jednym łuku na stronę (lewy przy 270°, prawy przy 90°)
/// i na warstwę. Warstwy są współśrodkowe jak cebula, a każda para ma własny kolor. Długość
/// łuku rośnie z poziomu tej strony. Zawsze jedna para jest podświetlona. Czerwona obwódka leży
/// na wyższym łuku tej pary, gdy jest aktywny.
fn draw_layer_bars<C: RadarCanvas>(canvas: &C, center: Pos2, cfg: &RadarConfig, state: &RadarState, now: f64) {
    const MAX_HALF_SPAN_DEG: f32 = 70.0;

    let r = cfg.radius;
    let fg = cfg.fg_alpha;
    let thickness = (r * 0.06).clamp(3.0, 9.0) * cfg.line_thickness.max(0.5);
    let inner = r * cfg.layer_arc_inner;
    let spacing = r * cfg.layer_arc_spacing;
    let highlighted = state.layer_bars.highlighted_layer();
    let focus = state.layer_bars.focus_side(now, cfg.decay_secs);

    // Każda warstwa (para L i P) ma swój kolor. Czerwony zostaje na obwódkę.
    let layer_rgb = |layer: Layer| -> (u8, u8, u8) {
        if cfg.high_contrast {
            return (255, 255, 255);
        }
        let c = match layer {
            Layer::Shots => cfg.layer_shots_color,
            Layer::Steps => cfg.layer_steps_color,
            Layer::Rest => cfg.layer_rest_color,
        };
        (c[0], c[1], c[2])
    };
    let rgba = |rgb: (u8, u8, u8), alpha: f32| {
        Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, (alpha * fg).clamp(0.0, 255.0) as u8)
    };

    // Strzały i kroki jako łuki w środku. Reszta jest rysowana starymi wskaźnikami (patrz niżej).
    for layer in [Layer::Shots, Layer::Steps] {
        let radius = inner + layer.index() as f32 * spacing;
        let rgb = layer_rgb(layer);
        let is_highlighted = layer == highlighted;
        // Każda para ma własne wzmocnienie widoczności z ustawień.
        let gain = match layer {
            Layer::Shots => cfg.layer_shots_gain,
            Layer::Steps => cfg.layer_steps_gain,
            Layer::Rest => cfg.layer_rest_gain,
        };
        let heights = [0usize, 1].map(|side| (state.layer_bars.height(side, layer, now, cfg.decay_secs) * gain).clamp(0.0, 1.0));
        let visible = visible_sides(heights[0], heights[1], 0.05);
        for (side, mid_deg) in [(0usize, 270.0_f32), (1, 90.0)] {
            if !visible[side] {
                continue;
            }
            // Ślad całego łuku, żeby było widać zakres. Podświetlona para ma szerszy.
            let track = if is_highlighted { thickness * 2.2 } else { thickness };
            let track_alpha = if is_highlighted { 45.0 } else { 22.0 };
            draw_arc(canvas, center, mid_deg, MAX_HALF_SPAN_DEG, radius, Stroke::new(track, rgba(rgb, track_alpha)));

            let height = heights[side];
            let half_span = MAX_HALF_SPAN_DEG * height;
            if is_highlighted && focus == Some(side) {
                let (ro, go, bo) = DOMINANT_OUTLINE_RGB;
                let outline = Color32::from_rgba_unmultiplied(ro, go, bo, (230.0 * fg).clamp(0.0, 255.0) as u8);
                draw_arc(canvas, center, mid_deg, half_span, radius, Stroke::new(thickness + 3.0, outline));
            }
            draw_arc(canvas, center, mid_deg, half_span, radius, Stroke::new(thickness, rgba(rgb, 120.0 + 135.0 * height)));
        }
    }

    // Reszta: stare wskaźniki L/R i przód/tył wokół tarczy, zasilane ciągłym tłem stereo,
    // więc działają też podczas jazdy, gdy nie ma wyraźnych zdarzeń.
    let lt = cfg.line_thickness.max(0.0);
    // Reszta: stare wskaźniki z własnym wzmocnieniem i kolorem pary.
    let rest_gain = cfg.layer_rest_gain;
    let scale = |history: &VecDeque<f32>| -> VecDeque<f32> {
        history.iter().map(|v| (v * rest_gain).clamp(0.0, 1.0)).collect()
    };
    let left_history = scale(&state.stereo_left_history);
    let right_history = scale(&state.stereo_right_history);
    let left_now = left_history.back().copied().unwrap_or(0.0);
    let right_now = right_history.back().copied().unwrap_or(0.0);
    let (rr, rg, rb) = layer_rgb(Layer::Rest);
    draw_stereo_indicators(
        canvas,
        center,
        r,
        &left_history,
        &right_history,
        [rr, rg, rb],
        [rr, rg, rb],
        fg,
        lt,
        visible_sides(left_now, right_now, 0.05),
    );
    // Przód i tył jak w starym widoku: łuki i odbicia kroków, gdy włączony jest widok przestrzenny.
    draw_spatial_arcs(canvas, center, r, state, now, fg, lt);

    // Legenda: kolor każdej warstwy pod radarem.
    for layer in Layer::ALL {
        let x = center.x + (layer.index() as f32 - 1.0) * r * 0.55;
        canvas.text(
            Pos2::new(x, center.y + r * 1.3),
            Align2::CENTER_CENTER,
            layer.label(),
            FontId::proportional(10.0),
            rgba(layer_rgb(layer), 200.0),
        );
    }
}

/// Tekstowy zrzut tego, co widać na radarze 3+3 w chwili `now`. Liczby są te same, z których
/// rysuje `draw_layer_bars`: wysokości łuków po wzmocnieniu, widoczność stron, podświetlenie,
/// czerwona obwódka i kolory.
pub fn layer_text(state: &RadarState, cfg: &RadarConfig, now: f64) -> String {
    let hex = |c: [u8; 3]| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]);
    let color = |layer: Layer| match layer {
        Layer::Shots => cfg.layer_shots_color,
        Layer::Steps => cfg.layer_steps_color,
        Layer::Rest => cfg.layer_rest_color,
    };
    let gain = |layer: Layer| match layer {
        Layer::Shots => cfg.layer_shots_gain,
        Layer::Steps => cfg.layer_steps_gain,
        Layer::Rest => cfg.layer_rest_gain,
    };
    let highlighted = state.layer_bars.highlighted_layer();
    let focus = state.layer_bars.focus_side(now, cfg.decay_secs);
    let side_name = |side: usize| if side == 0 { "L" } else { "P" };

    let mut out = String::new();
    for layer in [Layer::Shots, Layer::Steps] {
        let h = [0usize, 1].map(|side| {
            (state.layer_bars.height(side, layer, now, cfg.decay_secs) * gain(layer)).clamp(0.0, 1.0)
        });
        let vis = visible_sides(h[0], h[1], 0.05);
        let shown = |side: usize| if vis[side] { "" } else { " (ukryta)" };
        let mark = if layer == highlighted { " <- podświetlona" } else { "" };
        out.push_str(&format!(
            "{:<8} L {:.2}{:<10} P {:.2}{:<10} kolor {}{}\n",
            layer.label(),
            h[0],
            shown(0),
            h[1],
            shown(1),
            hex(color(layer)),
            mark
        ));
        if layer == highlighted {
            if let Some(side) = focus {
                out.push_str(&format!("         czerwona obwódka: {}\n", side_name(side)));
            }
        }
    }

    let (left, right) = state.stereo_now();
    let rest_gain = cfg.layer_rest_gain;
    let (l, r) = ((left * rest_gain).clamp(0.0, 1.0), (right * rest_gain).clamp(0.0, 1.0));
    let vis = visible_sides(l, r, 0.05);
    let shown = |side: usize| if vis[side] { "" } else { " (ukryta)" };
    out.push_str(&format!(
        "{:<8} L {:.2}{:<10} P {:.2}{:<10} kolor {}\n",
        Layer::Rest.label(),
        l,
        shown(0),
        r,
        shown(1),
        hex(color(Layer::Rest))
    ));
    out
}

/// Które strony (lewa, prawa) pokazać. Gdy jedna ma wyraźną przewagę, słabsza się chowa.
/// Przewaga to co najmniej 1,1-krotność drugiej strony. Poniżej `min` obie strony są widoczne.
fn visible_sides(left: f32, right: f32, min: f32) -> [bool; 2] {
    const DOMINANCE_RATIO: f32 = 1.1;
    if left.max(right) < min {
        [true, true]
    } else if left > right * DOMINANCE_RATIO {
        [true, false]
    } else if right > left * DOMINANCE_RATIO {
        [false, true]
    } else {
        [true, true]
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

/// Czy lewy i prawy kanał grają równo — dźwięk stoi na środku, czyli z przodu
/// albo z tyłu. Oba poziomy muszą być słyszalne, żeby cisza nie liczyła się jako „równo”.
fn stereo_balanced(left: f32, right: f32) -> bool {
    left.max(right) >= BALANCE_MIN_LEVEL && (left - right).abs() <= BALANCE_TOLERANCE
}

/// Przód i tył jako dwa łuki (góra/dół) w stylu łuków L/R. Długość = odchylenie
/// przód/tył (z ostatniego kroku, a między krokami — z ciągłego estymatora),
/// jasność = wysokość. Znika w ciszy. Gdy L i P grają równo, złoty łuk „przód”
/// dostaje czerwoną obwódkę.
fn draw_spatial_arcs<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    r: f32,
    state: &RadarState,
    now: f64,
    fg: f32,
    lt: f32,
) {
    // Łuki L/R zajmują 20°–160° i 200°–340°, więc P/T muszą mieścić się w ±20°.
    const MAX_HALF_SPAN_DEG: f32 = 18.0;
    // Najmniejsza połowa rozpiętości obwódki przy L=R, żeby czerwony obrys był
    // widoczny także wtedy, gdy złoty łuk jest krótki albo właśnie zgasł.
    const BALANCE_OUTLINE_MIN_HALF_SPAN_DEG: f32 = 8.0;
    if state.spatial_energy < 0.005 {
        return;
    }

    let thickness = stereo_arc_thickness(r, lt);
    let arc_r = r + 4.0 + thickness * 0.5;
    let brightness = 0.75 + 0.25 * state.spatial_height.clamp(-1.0, 1.0);
    let front_back = state.displayed_front_back(now);
    let front = front_back.max(0.0);
    let back = (-front_back).max(0.0);
    let balanced = stereo_balanced(state.stereo_left_display, state.stereo_right_display);
    let font = FontId::proportional((thickness * 1.8).clamp(10.0, 14.0));

    const SHIFT_PX: f32 = 10.0;
    const FRONT_RGB: (u8, u8, u8) = (255, 225, 0);
    const BACK_RGB: (u8, u8, u8) = (170, 60, 255);
    for (mid_deg, value, label, dy, rgb, outlined) in [
        (0.0_f32, front, "P", -SHIFT_PX, FRONT_RGB, balanced),
        (180.0_f32, back, "T", SHIFT_PX, BACK_RGB, false),
    ] {
        let rgba = |a: f32| {
            let alpha = (a * fg * brightness).clamp(0.0, 255.0) as u8;
            Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha)
        };
        let value = value.clamp(0.0, 1.0);
        let half_span = MAX_HALF_SPAN_DEG * value;
        let c = center + Vec2::new(0.0, dy);
        draw_arc(canvas, c, mid_deg, MAX_HALF_SPAN_DEG, arc_r, Stroke::new(thickness, rgba(60.0)));
        draw_arc(canvas, c, mid_deg, half_span, arc_r, Stroke::new(thickness * 2.2, rgba(80.0 + 80.0 * value)));
        if outlined {
            // Obrys jak dominujący łuk L/R: szerszy czerwony pas pod złotym.
            let (ro, go, bo) = DOMINANT_OUTLINE_RGB;
            let outline = Color32::from_rgba_unmultiplied(ro, go, bo, (230.0 * fg) as u8);
            let outline_span = half_span.max(BALANCE_OUTLINE_MIN_HALF_SPAN_DEG);
            draw_arc(canvas, c, mid_deg, outline_span, arc_r, Stroke::new(thickness + 4.0, outline));
        }
        draw_arc(canvas, c, mid_deg, half_span, arc_r, Stroke::new(thickness, rgba(160.0 + 95.0 * value)));
        let label_pos = c + screen_direction(mid_deg) * (arc_r + thickness + 8.0);
        canvas.text(label_pos, Align2::CENTER_CENTER, label, font.clone(), rgba(180.0 + 75.0 * value));
    }
}

/// Błysk na łukach L/R w chwili kroku: biały łuk po stronie, z której przyszedł
/// krok, gaśnie w ciągu [`STEP_MARKER_LIFETIME_SECS`]. Krok pośrodku (balans ≈ 0)
/// błyska na obu łukach po połowie. Każdy krok to osobny impuls w rytmie chodu.
fn draw_step_flashes<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    r: f32,
    state: &RadarState,
    now: f64,
    fg: f32,
    lt: f32,
) {
    // Ta sama rozpiętość i promień co łuki L/R (patrz `draw_stereo_indicators`).
    const FLASH_MAX_HALF_SPAN_DEG: f32 = 70.0;
    // Kąt głośnika FL/FR — skrajny balans L/R uderzenia (patrz `stereo_onset_angle_deg`).
    const FRONT_SPEAKER_DEG: f32 = 45.0;

    let thickness = stereo_arc_thickness(r, lt);
    let arc_r = r + 4.0 + thickness * 0.5;
    let lifetime = STEP_MARKER_LIFETIME_SECS as f32;
    for step in &state.steps {
        let age = (now - step.spawned_at) as f32;
        if age < 0.0 || age > lifetime {
            continue;
        }
        let alpha_frac = 1.0 - age / lifetime;
        let half_span = FLASH_MAX_HALF_SPAN_DEG * (0.4 + 0.6 * step.intensity);
        let right_share = (step.pan_deg / FRONT_SPEAKER_DEG * 0.5 + 0.5).clamp(0.0, 1.0);
        for (mid_deg, share) in [(270.0_f32, 1.0 - right_share), (90.0_f32, right_share)] {
            let alpha = (255.0 * fg * alpha_frac * share).clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let color = Color32::from_rgba_unmultiplied(255, 255, 255, alpha);
            draw_arc(canvas, center, mid_deg, half_span, arc_r, Stroke::new(thickness * 2.0, color));
        }
    }
}

/// Pierścień kropek w pozycjach głośników 5.1/7.1; jasność każdej = poziom jej kanału.
/// Dominujący głośnik (patrz [`dominant_speaker`]) dostaje czerwony obrys.
fn draw_speaker_ring<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    r: f32,
    layout: &[ChannelRole],
    levels: &[f32],
    fg: f32,
    lt: f32,
) {
    let ring_r = r + 14.0;
    let lt = lt.max(0.5);
    let dominant = dominant_speaker(layout, levels);
    for (i, role) in layout.iter().enumerate() {
        let Some(angle) = role.radar_angle_degrees() else {
            continue;
        };
        let level = levels.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let pos = center + screen_direction(angle) * ring_r;
        let dot_r = 3.5 * lt + 2.0 * level;
        let fill = Color32::from_rgba_unmultiplied(0, 255, 157, ((60.0 + 195.0 * level) * fg) as u8);
        canvas.circle_filled(pos, dot_r, fill);
        canvas.circle_stroke(
            pos,
            dot_r + 1.5,
            Stroke::new(1.0 * lt, Color32::from_rgba_unmultiplied(0, 255, 157, (90.0 * fg) as u8)),
        );
        if dominant == Some(i) {
            let (ro, go, bo) = DOMINANT_OUTLINE_RGB;
            let outline = Color32::from_rgba_unmultiplied(ro, go, bo, (230.0 * fg) as u8);
            canvas.circle_stroke(pos, dot_r + 4.0, Stroke::new(2.0 * lt, outline));
        }
    }
}

/// Indeks dominującego głośnika w pierścieniu 5.1/7.1: ten kanał kierunkowy,
/// którego poziom wyraźnie (o więcej niż [`DOMINANCE_MARGIN`]) przewyższa
/// drugi najgłośniejszy. LFE pomijany (nie ma pozycji). `None`, gdy nikt nie
/// dominuje albo sygnał jest zbyt cichy, żeby go oceniać.
fn dominant_speaker(layout: &[ChannelRole], levels: &[f32]) -> Option<usize> {
    const MIN_LEVEL: f32 = 0.05;
    let mut best: Option<(usize, f32)> = None;
    let mut second = 0.0f32;
    for (i, role) in layout.iter().enumerate() {
        if role.radar_angle_degrees().is_none() {
            continue;
        }
        let level = levels.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        match best {
            Some((_, top)) if level <= top => second = second.max(level),
            Some((_, top)) => {
                second = second.max(top);
                best = Some((i, level));
            }
            None => best = Some((i, level)),
        }
    }
    let (idx, top) = best?;
    (top >= MIN_LEVEL && top - second > DOMINANCE_MARGIN).then_some(idx)
}

fn stereo_arc_thickness(r: f32, lt: f32) -> f32 {
    (r * 0.05).clamp(3.0, 8.0) * lt.max(0.5)
}

fn draw_arc<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    mid_deg: f32,
    half_span: f32,
    radius: f32,
    stroke: Stroke,
) {
    if half_span <= 0.0 {
        return;
    }
    const STEP_DEG: f32 = 3.0;
    let steps = ((2.0 * half_span) / STEP_DEG).ceil().max(1.0) as usize;
    let start = mid_deg - half_span;
    let step = 2.0 * half_span / steps as f32;
    let point_at = |deg: f32| center + screen_direction(deg) * radius;
    let mut prev = point_at(start);
    for i in 1..=steps {
        let next = point_at(start + step * i as f32);
        canvas.line_segment([prev, next], stroke);
        prev = next;
    }
}

/// Łuki L/R na obwodzie tarczy: lewy rozchodzi się od godziny 9, prawy od 3,
/// długość i jasność rosną z energią kanału; cienka kreska to szczyt z historii.
fn draw_stereo_indicators<C: RadarCanvas>(
    canvas: &C,
    center: Pos2,
    r: f32,
    left_history: &VecDeque<f32>,
    right_history: &VecDeque<f32>,
    left_rgb: [u8; 3],
    right_rgb: [u8; 3],
    fg: f32,
    lt: f32,
    visible: [bool; 2],
) {
    const MAX_HALF_SPAN_DEG: f32 = 70.0;

    let thickness = stereo_arc_thickness(r, lt);
    let arc_r = r + 4.0 + thickness * 0.5;

    let point_at = |deg: f32, radius: f32| center + screen_direction(deg) * radius;

    let left_energy = left_history.back().copied().unwrap_or(0.0);
    let right_energy = right_history.back().copied().unwrap_or(0.0);

    let channels = [
        (270.0_f32, left_history, left_rgb, "L", left_energy - right_energy > DOMINANCE_MARGIN),
        (90.0_f32, right_history, right_rgb, "R", right_energy - left_energy > DOMINANCE_MARGIN),
    ];

    for (mid_deg, history, rgb, label, dominant) in channels {
        let side = if mid_deg < 180.0 { 0 } else { 1 };
        if !visible[side] {
            continue;
        }
        let energy = history.back().copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let peak = history.iter().copied().fold(0.0_f32, f32::max).clamp(0.0, 1.0);
        let rgba = |a: f32| Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], (a * fg).clamp(0.0, 255.0) as u8);

        draw_arc(canvas, center, mid_deg, MAX_HALF_SPAN_DEG, arc_r, Stroke::new(thickness, rgba(28.0)));

        let half_span = MAX_HALF_SPAN_DEG * energy;
        draw_arc(canvas, center, mid_deg, half_span, arc_r, Stroke::new(thickness * 2.2, rgba(40.0 + 60.0 * energy)));
        if dominant {
            let (ro, go, bo) = DOMINANT_OUTLINE_RGB;
            let outline = Color32::from_rgba_unmultiplied(ro, go, bo, (230.0 * fg) as u8);
            draw_arc(canvas, center, mid_deg, half_span, arc_r, Stroke::new(thickness + 4.0, outline));
        }
        draw_arc(canvas, center, mid_deg, half_span, arc_r, Stroke::new(thickness, rgba(110.0 + 145.0 * energy)));

        if peak > 0.02 {
            let peak_span = MAX_HALF_SPAN_DEG * peak;
            let tick = Stroke::new(1.5 * lt.max(0.5), rgba(230.0));
            for deg in [mid_deg - peak_span, mid_deg + peak_span] {
                canvas.line_segment(
                    [point_at(deg, arc_r - thickness), point_at(deg, arc_r + thickness)],
                    tick,
                );
            }
        }

        let font = FontId::proportional((thickness * 1.8).clamp(10.0, 14.0));
        canvas.text(
            point_at(mid_deg, arc_r + thickness + 8.0),
            Align2::CENTER_CENTER,
            label,
            font,
            rgba(120.0 + 135.0 * energy),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_speaker_needs_a_clear_gap_over_the_second_loudest() {
        let layout = channel_layout(6); // FL, FR, C, LFE, RL, RR
        // Wyraźnie głośny FL, reszta cicho → FL dominuje (LFE pomijany).
        assert_eq!(dominant_speaker(layout, &[0.6, 0.1, 0.1, 0.9, 0.1, 0.1]), Some(0));
        // Dwa równe, głośne kanały → nikt nie dominuje.
        assert_eq!(dominant_speaker(layout, &[0.5, 0.5, 0.1, 0.9, 0.1, 0.1]), None);
        // Cisza poniżej progu → brak obrysu.
        assert_eq!(dominant_speaker(layout, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn stereo_balanced_needs_equal_levels_and_an_audible_signal() {
        assert!(stereo_balanced(0.60, 0.61), "prawie równe i głośne");
        assert!(!stereo_balanced(0.60, 0.40), "wyraźnie nierówne");
        assert!(!stereo_balanced(0.01, 0.01), "równe, ale cisza — nic nie stoi na środku");
    }

    /// Onset o podanej chwili i kącie, z mocą 0,8 i kolorem czerwonym.
    fn timed(at: f64, angle_deg: f32) -> TimedDirectionEvent {
        TimedDirectionEvent {
            at,
            event: DirectionEvent { angle_deg, intensity: 0.8, color_rgb: (255, 0, 0) },
        }
    }

    #[test]
    fn walking_rhythm_records_steps_and_a_lone_hit_does_not() {
        let mut state = RadarState::new();
        let hit = |at: f64| [timed(at, -30.0)];
        state.update(0.0, 2, &[0.0, 0.0], &[], &[], 2.5, 20.0, &hit(0.0));
        assert!(state.steps.is_empty(), "pierwsze uderzenie nie jest krokiem");
        state.update(0.5, 2, &[0.0, 0.0], &[], &[], 2.5, 20.0, &hit(0.5));
        assert_eq!(state.steps.len(), 1, "drugie uderzenie w rytmie 0,5 s to krok");
        assert!(state.steps[0].pan_deg < 0.0, "balans z lewej strony");
    }

    #[test]
    fn bright_step_over_a_darker_background_is_placed_in_front() {
        let rules = crate::sound::spectrum::default_color_rules();
        let event = DirectionEvent { angle_deg: 0.0, intensity: 0.5, color_rgb: (0, 0, 0) };
        // Kanały L/R z tym samym widmem: wysokie pasma dominują (jasny krok) albo
        // niskie (ciemne tło). Linia bazowa stoi pośrodku.
        let bright = vec![vec![0.0, 0.0, 0.0, 0.5, 0.5]; 2];
        let dark = vec![vec![0.5, 0.5, 0.0, 0.0, 0.0]; 2];

        let mut state = RadarState::new();
        state.spatial_tilt_baseline = 0.5;
        state.spatial_baseline_ready = true;
        let (tilt, _) = front_band_shares(2, &rules, &bright);
        state.push_step(1.0, &event, tilt);
        assert!(state.steps.back().unwrap().front_back > 0.0, "jasny krok = przód");

        let (tilt, _) = front_band_shares(2, &rules, &dark);
        state.push_step(2.0, &event, tilt);
        assert!(state.steps.back().unwrap().front_back < 0.0, "ciemny krok = tył");
    }
}

#[cfg(test)]
mod timed_queue_tests {
    use super::*;

    fn event(angle_deg: f32) -> DirectionEvent {
        DirectionEvent { angle_deg, intensity: 0.8, color_rgb: (255, 0, 0) }
    }

    #[test]
    fn events_from_one_callback_share_one_time() {
        let captured = Instant::now();
        let queued = [
            QueuedDirectionEvent { captured_at: captured, event: event(-30.0) },
            QueuedDirectionEvent { captured_at: captured, event: event(10.0) },
        ];
        let timed = timed_from_queue(5.0, &queued);
        assert_eq!(timed[0].at, timed[1].at, "ta sama chwila dla jednego callbacku");
        assert!(timed[0].at <= 5.0 && timed[0].at > 4.9, "chwila przed `now`: {}", timed[0].at);
    }

    #[test]
    fn events_from_two_callbacks_in_one_frame_count_as_two_hits() {
        // Dwa callbacki w jednej klatce UI, 0,5 s od siebie: to dwa uderzenia i krok.
        let mut state = RadarState::new();
        let batch = [
            TimedDirectionEvent { at: 0.0, event: event(-30.0) },
            TimedDirectionEvent { at: 0.5, event: event(-30.0) },
        ];
        state.update(0.5, 2, &[0.0, 0.0], &[], &[], 2.5, 20.0, &batch);
        assert_eq!(state.steps.len(), 1, "drugi callback w klatce to osobne uderzenie");
    }

    #[test]
    fn timed_now_puts_synthetic_events_at_the_frame_time() {
        let timed = timed_now(3.0, &[event(0.0), event(45.0)]);
        assert!(timed.iter().all(|t| t.at == 3.0));
        assert_eq!(timed[1].event.angle_deg, 45.0);
    }
}
