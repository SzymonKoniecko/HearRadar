use crate::sound::devices::RadarConfig;
use crate::sound::direction::{channel_layout, ChannelRole};
use crate::sound::spectrum::{self, BandEnergies, FrequencyBand};
use egui::{Color32, FontId, Painter, Pos2, Stroke, Vec2};
use std::collections::VecDeque;

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
    /// Prawdziwa sinusoida o zadanej częstotliwości (Hz), puszczona przez
    /// TEN SAM potok analizy co żywe audio (`analyze_channels` +
    /// `spectrum::analyze_channel_bands`) — w przeciwieństwie do pozostałych
    /// scenariuszy, które podają syntetyczny RMS wprost. Służy do
    /// weryfikacji progów kolorowania pasm (200 Hz / 2 kHz) na żywo: przesuń
    /// suwak Hz i patrz, kiedy kropka zmienia kolor. Obsługiwane osobno w
    /// `HearRadarApp::tick_radar_state`, nie przez [`Self::generate`].
    Tone        = 8,
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
            8 => Self::Tone,
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
                    // Obsługiwane osobno w `tick_radar_state` (potrzebuje realnych
                    // próbek, nie samego RMS) — tu tylko niegroźna wartość zapasowa.
                    Self::Tone        => 0.0,
                }
            })
            .collect()
    }

    /// Syntetyczna energia pasm dla scenariuszy testowych: nie liczymy
    /// prawdziwego widma (nie ma prawdziwego sygnału), tylko demonstrujemy
    /// kolorowanie kropek wg pasma — dominujące pasmo powoli krąży po
    /// kanałach w czasie, żeby w podglądzie było widać wszystkie 3 kolory.
    pub fn synthetic_bands(self, channels: usize, phase: f32) -> Vec<BandEnergies> {
        (0..channels)
            .map(|i| {
                let dominant = (i + (phase * 0.4) as usize) % 3;
                let mut e = [0.05f32; 3];
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
    /// Dominujące pasmo częstotliwości tej ramki — decyduje o kolorze kropki
    /// (patrz [`band_color`]). Ustalane raz przy powstaniu kropki, więc kolor
    /// nie "skacze" w trakcie jej zaniku.
    pub band: FrequencyBand,
}

/// Kolor przypisany każdemu pasmu częstotliwości. Na razie na sztywno, ale
/// celowo wydzielone do jednej funkcji: docelowy konfigurator kolorów (reguła
/// parametr → kolor, edytowalna w UI) ma tu podmienić tylko to jedno miejsce
/// — resztą rysowania steruje wyłącznie `dot.band`, więc nic więcej się nie
/// zmieni.
fn band_color(band: FrequencyBand) -> (u8, u8, u8) {
    match band {
        FrequencyBand::Low => (255, 90, 90),   // czerwony: bas / niskie częstotliwości
        FrequencyBand::Mid => (0, 220, 220),   // cyjan: średnie (mowa, kroki)
        FrequencyBand::High => (255, 220, 90), // żółty: wysokie / syczące tony
    }
}

/// Stan radaru utrzymywany między klatkami: historia kropek per kanał,
/// wygładzona energia (do detekcji transientów) i osobny wskaźnik LFE.
#[derive(Default)]
pub struct RadarState {
    dots: Vec<VecDeque<ChannelDot>>,
    ema: Vec<f32>,
    last_spawn: f64,
    /// Wygładzony poziom LFE do rysowania osobnego wskaźnika niskich częstotliwości.
    pub lfe_display: f32,
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
    /// `channel_bands` to energia w 3 pasmach (`[low, mid, high]`) per kanał —
    /// patrz [`crate::sound::spectrum`]. Może być krótsza niż `channels`
    /// (np. scenariusze testowe, które jej nie liczą); brakujące kanały
    /// dostają domyślnie pasmo `Mid`. `decay_secs` (z [`RadarConfig`]) decyduje
    /// ile kropek trzeba trzymać naraz, żeby żadna nie zniknęła, zanim
    /// naturalnie wygaśnie — patrz [`MIN_DOTS_PER_CHANNEL`]/[`MAX_DOTS_PER_CHANNEL`].
    pub fn update(&mut self, now: f64, channels: usize, channel_rms: &[f32], channel_bands: &[BandEnergies], decay_secs: f32) {
        self.ensure_capacity(channels);

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
            let energy = channel_rms.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let role = channel_layout(channels).get(i).copied();

            if role == Some(ChannelRole::Lfe) {
                self.lfe_display = self.lfe_display * 0.8 + energy * 0.2;
                continue;
            }

            let prev_ema = self.ema[i];
            let transient = energy > prev_ema * TRANSIENT_RATIO + 0.05;
            self.ema[i] = prev_ema * 0.9 + energy * 0.1;

            if energy < 0.01 {
                continue;
            }

            let band = channel_bands
                .get(i)
                .map(|&e| spectrum::dominant_band(e))
                .unwrap_or(FrequencyBand::Mid);

            dots.push_back(ChannelDot { spawned_at: now, energy, transient, band });
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
    }
}

// ── Główna funkcja rysująca ───────────────────────────────────────────────────

const LABELS: [&str; 4] = ["PRZÓD", "PRAWO", "TYŁ", "LEWO"];

/// Rysuje minimalistyczny radar audio: tło, okręgi skali, krzyż kierunków,
/// opcjonalne sektory 7.1, kropki kanałów (z zanikiem w czasie) i osobny
/// wskaźnik LFE. `now` to bieżący czas w sekundach (do liczenia zaniku).
pub fn draw_radar(painter: &Painter, center: Pos2, cfg: &RadarConfig, state: &mut RadarState, channels: usize, now: f64) {
    state.prune(now, cfg.decay_secs);

    let r = cfg.radius;
    let fg = cfg.fg_alpha;
    let hc = cfg.high_contrast;

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

    painter.circle_filled(center, r, c_bg);

    // 3 koncentryczne okręgi = skala intensywności (słaby/średni/mocny), nie
    // fizyczna odległość.
    for frac in [1.0f32, 0.66, 0.33] {
        let color = if frac == 1.0 { with_alpha(c_ring, fg) } else { with_alpha(c_inner, fg) };
        let width = if frac == 1.0 { 1.5 } else { 0.8 };
        painter.circle_stroke(center, r * frac, Stroke::new(width, color));
    }

    // Krzyż kierunków.
    let cross = Stroke::new(0.8, with_alpha(c_cross, fg));
    painter.line_segment([center - Vec2::new(0.0, r), center + Vec2::new(0.0, r)], cross);
    painter.line_segment([center - Vec2::new(r, 0.0), center + Vec2::new(r, 0.0)], cross);

    // Bardzo delikatne linie sektorów co 45° — tylko w układzie 7.1, gdzie
    // każdy z 8 kierunków ma osobny kanał.
    if channels == 8 {
        let sector = Stroke::new(0.5, with_alpha(c_cross, fg * 0.4));
        for k in 0..8 {
            let a = (k as f32) * std::f32::consts::FRAC_PI_4 - std::f32::consts::FRAC_PI_2;
            let p = center + Vec2::new(a.cos(), a.sin()) * r;
            painter.line_segment([center, p], sector);
        }
    }

    painter.circle_filled(center, 2.5, with_alpha(c_center, fg));

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
            painter.text(*pos, egui::Align2::CENTER_CENTER, *label, font.clone(), label_color);
        }
    }

    // Kropki kanałów.
    let layout = channel_layout(channels);
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

            // Kolor koduje "typ" dźwięku wg dominującego pasma częstotliwości
            // (bas/średnie/wysokie) — patrz [`band_color`]. W trybie wysokiego
            // kontrastu wyłączamy to i wracamy do jednej barwy, żeby nie
            // polegać wyłącznie na kolorze (rozróżnienie zostaje przez
            // kąt/rozmiar/kształt).
            let rgb = if hc { dot_rgb } else { band_color(dot.band) };

            // Styl kropki zmienia się wraz z jej wiekiem, żeby powolne
            // wygasanie dźwięku było czytelne nie tylko przez przezroczystość,
            // ale przez samą formę: świeża kropka ma pełne wypełnienie i
            // delikatną poświatę, w połowie życia traci poświatę, a tuż przed
            // zniknięciem "rozpuszcza się" w pusty, kurczący się pierścień.
            if life > 0.66 {
                let glow_alpha = (alpha * 0.35).clamp(0.0, 255.0) as u8;
                let glow_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, glow_alpha);
                painter.circle_filled(pos, size * 2.0, glow_color);
                let color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha.clamp(0.0, 255.0) as u8);
                painter.circle_filled(pos, size, color);
            } else if life > 0.33 {
                let color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, alpha.clamp(0.0, 255.0) as u8);
                painter.circle_filled(pos, size, color);
            } else {
                // Mocniejszy kontur niż wypełnienie miałoby przy tej alpha,
                // żeby cienki pierścień był wciąż widoczny tuż przed zniknięciem.
                let ring_alpha = (alpha * 1.6).clamp(0.0, 255.0) as u8;
                let stroke_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, ring_alpha);
                painter.circle_stroke(pos, size.max(1.5), Stroke::new(1.0, stroke_color));
            }

            // Nagły skok energii: krótki, gasnący pierścień wokół kropki.
            if dot.transient && age < 0.25 {
                let ring_life = 1.0 - (age / 0.25);
                let ring_r = size + 3.0 + (0.25 - age) * 40.0;
                let ring_alpha = (200.0 * ring_life * fg) as u8;
                let ring_color = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, ring_alpha);
                painter.circle_stroke(Pos2::new(x, y), ring_r, Stroke::new(1.5, ring_color));
            }
        }
    }

    // Wskaźnik LFE — osobny, poza tarczą, bez przypisanego kierunku. LFE to
    // z definicji bas, więc używa koloru pasma "low" (poza trybem wysokiego
    // kontrastu, gdzie i tak wracamy do jednej barwy).
    if layout.contains(&ChannelRole::Lfe) {
        let lfe_rgb = if hc { dot_rgb } else { band_color(FrequencyBand::Low) };
        draw_lfe_indicator(painter, center, r, state.lfe_display, fg, lfe_rgb);
    }
}

fn draw_lfe_indicator(painter: &Painter, center: Pos2, r: f32, level: f32, fg: f32, rgb: (u8, u8, u8)) {
    let w = (r * 0.5).clamp(30.0, 70.0);
    let h = 6.0;
    let top_left = Pos2::new(center.x - w / 2.0, center.y + r + 14.0);
    let bg = Color32::from_rgba_unmultiplied(255, 255, 255, (30.0 * fg) as u8);
    painter.rect_filled(egui::Rect::from_min_size(top_left, Vec2::new(w, h)), 2.0, bg);

    let fill_w = w * level.clamp(0.0, 1.0);
    let fill = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, (220.0 * fg) as u8);
    painter.rect_filled(egui::Rect::from_min_size(top_left, Vec2::new(fill_w, h)), 2.0, fill);
}
