//! Test kierunków: krótki dźwięk z zadanego kierunku, grany przez wybrane wyjście (słuchawki),
//! i ten sam sygnał przepuszczony przez analizę radaru. Porównujemy trzy kąty: gdzie dźwięk
//! naprawdę był, gdzie użytkownik go usłyszał i gdzie pokazał go radar.
//!
//! Konwencja jak w `direction::ChannelRole::radar_angle_degrees`: `0°` = przód (góra tarczy),
//! kąty dodatnie = prawo, `±180°` = tył.
//!
//! Dwa sygnały testowe. `Level` to sama różnica głośności L/R, czyli to, czym radar dziś
//! wyznacza kierunek dla stereo. `Itd` to sama różnica czasu dotarcia (jedno ucho później, bez
//! różnicy głośności), czyli to, czym gra z HRTF podaje boki. Porównanie pokazuje, czy radar
//! widzi oba.
//!
//! Analiza próby idzie tym samym `devices::update_analysis` co żywe audio, w kawałkach po 10 ms
//! (jak `playback`), więc wynik to to, co radar pokazałby przy tym dźwięku. Osobna instancja
//! `SharedAudioAnalysis` nie zabiera zdarzeń z kolejki, z której rysuje się radar.

use crate::sound::binaural::{DEFAULT_EAR_DISTANCE_M, SPEED_OF_SOUND_M_S};
use crate::sound::channel_test::find_output_device;
use crate::sound::devices::{self, RadarConfig, SharedAudioAnalysis};
use anyhow::{Result, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait, StreamTrait},
};
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_4};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Kierunki testu w konwencji radaru.
pub const DIRECTIONS_DEG: [f32; 8] = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0];

/// Kanały wejścia radaru symulowane, gdy nic akurat nie przechwytuje. `8` to 7.1, jak przy
/// kablu wirtualnym (CABLE Input). Sygnał idzie na FL/FR, reszta kanałów zostaje cicha.
pub const DEFAULT_CHANNELS: usize = 8;

/// Częstotliwość próbkowania toru na żywo (przechwycone audio i mono router).
pub const MONO_RATE: f32 = 48_000.0;

/// Tor dźwięku próby: prosto na wybrane wyjście, albo wpuszczona w analizę na żywo, skąd idzie
/// przez mono router (z jego ustawieniami ucha, wskazówki, tonu, Scouta, zmiany klas i wzmocnienia).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Direct,
    Mono,
}

impl Route {
    pub const ALL: [Route; 2] = [Route::Direct, Route::Mono];

    pub fn label(self) -> &'static str {
        match self {
            Route::Direct => "Bezpośrednio",
            Route::Mono => "Przez mono router",
        }
    }

    /// Nazwa w pliku CSV.
    pub fn key(self) -> &'static str {
        match self {
            Route::Direct => "direct",
            Route::Mono => "mono",
        }
    }
}

/// Odstęp między kawałkami analizy, jak w `playback` i w typowym buforze cpal.
const CHUNK_S: f64 = 0.01;
/// Każda próba i jej dźwięk trwają dokładnie tyle samo: sekundę szumu.
pub const SOUND_S: f32 = 1.0;
/// Krótki narost i zanik, żeby początek i koniec nie trzaskały.
const FADE_S: f32 = 0.01;
/// Cisza wpuszczona tylko do analizy przed próbą. Detektor zdarzeń widzi początek dźwięku dopiero
/// na tle spokojnego poziomu, a słuchawki tej ciszy nie dostają.
const PROBE_WARMUP_S: f32 = 0.3;
const PEAK: f32 = 0.25;
const NOISE_SEED: u32 = 0x9E37_79B9;

/// Sposób, w jaki różnica między uszami trafia do sygnału.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Różnica głośności L/R, czas dotarcia taki sam.
    Level,
    /// Różnica czasu dotarcia (ITD), głośność L i R taka sama.
    Itd,
}

impl Signal {
    pub const ALL: [Signal; 2] = [Signal::Level, Signal::Itd];

    pub fn label(self) -> &'static str {
        match self {
            Signal::Level => "Głośność L/R",
            Signal::Itd => "Czas (ITD)",
        }
    }

    /// Nazwa w pliku CSV.
    pub fn key(self) -> &'static str {
        match self {
            Signal::Level => "level",
            Signal::Itd => "itd",
        }
    }
}

/// Zawija kąt do przedziału `(-180, 180]`.
pub fn wrap_deg(angle: f32) -> f32 {
    (angle + 180.0).rem_euclid(360.0) - 180.0
}

/// Próba jako stereo `[lewy, prawy]` o częstotliwości `rate`: sekunda szumu z kierunkiem.
pub fn burst_stereo(angle_deg: f32, signal: Signal, rate: f32) -> Vec<[f32; 2]> {
    let len = (SOUND_S * rate).round() as usize;
    let mut frames = vec![[0.0f32; 2]; len];

    let theta = angle_deg.to_radians();
    let itd_samples = DEFAULT_EAR_DISTANCE_M * theta.sin() / SPEED_OF_SOUND_M_S * rate;
    let delay = itd_samples.abs().round() as usize;
    let (gain_l, gain_r, late_l, late_r) = match signal {
        Signal::Level => {
            // Stała moc: L² + R² = 1, kąty pośrednie dają pośrednie głośności.
            let phi = (theta.sin() + 1.0) * FRAC_PI_4;
            (phi.cos(), phi.sin(), 0, 0)
        }
        Signal::Itd => {
            // Źródło z prawej: lewe ucho słyszy później (jak w `binaural`).
            let (late_l, late_r) = if itd_samples > 0.0 { (delay, 0) } else { (0, delay) };
            (FRAC_1_SQRT_2, FRAC_1_SQRT_2, late_l, late_r)
        }
    };

    let fade = (FADE_S * rate).max(1.0);
    let mut seed = NOISE_SEED;
    for i in 0..len {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let envelope = (i as f32 / fade).min((len - i) as f32 / fade).min(1.0);
        let sample = noise * envelope * PEAK;
        // Ucho opóźnione kończy się kilka próbek wcześniej, za krótko, żeby było słychać.
        if let Some(frame) = frames.get_mut(i + late_l) {
            frame[0] += gain_l * sample;
        }
        if let Some(frame) = frames.get_mut(i + late_r) {
            frame[1] += gain_r * sample;
        }
    }
    frames
}

/// Stereo rozłożone na `channels` kanałów interleaved: `[L, R, 0, 0, …]`. Mono dostaje średnią.
pub fn to_channels(frames: &[[f32; 2]], channels: usize) -> Vec<f32> {
    let channels = channels.max(1);
    let mut out = Vec::with_capacity(frames.len() * channels);
    for &[l, r] in frames {
        for c in 0..channels {
            out.push(match (channels, c) {
                (1, _) => 0.5 * (l + r),
                (_, 0) => l,
                (_, 1) => r,
                _ => 0.0,
            });
        }
    }
    out
}

/// Wejście analizy dla próby: [`PROBE_WARMUP_S`] ciszy, potem próba, na `channels` kanałach.
pub fn probe_input(frames: &[[f32; 2]], channels: usize, rate: f32) -> Vec<f32> {
    let mut padded = vec![[0.0f32; 2]; (PROBE_WARMUP_S * rate).round() as usize];
    padded.extend_from_slice(frames);
    to_channels(&padded, channels)
}

/// Kąt wskazany myszką względem środka kompasu (`dx`, `dy` w pikselach, `y` rośnie w dół).
pub fn angle_from_pointer(dx: f32, dy: f32) -> f32 {
    dx.atan2(-dy).to_degrees()
}

/// Każdy kierunek `repeats` razy, w kolejności przemieszanej (Fisher–Yates, xorshift od `seed`).
pub fn shuffled(angles: &[f32], repeats: usize, seed: u64) -> Vec<f32> {
    let mut order: Vec<f32> = (0..repeats).flat_map(|_| angles.iter().copied()).collect();
    let mut state = seed.max(1);
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    order
}

/// Wynik radaru dla jednej próby: średnia kątów zdarzeń ważona intensywnością. Dwa równe
/// zdarzenia z obu stron dają środek, pojedyncze zdarzenie daje swój kąt.
pub fn radar_reading(events: &[(f32, f32)]) -> Option<f32> {
    let (mut x, mut y, mut total) = (0.0f32, 0.0f32, 0.0f32);
    for &(angle, weight) in events {
        let w = weight.max(0.0);
        let rad = angle.to_radians();
        x += w * rad.sin();
        y += w * rad.cos();
        total += w;
    }
    (total > f32::EPSILON).then(|| x.atan2(y).to_degrees())
}

/// Próbki (`[lewy, prawy]`) przepuszczone przez analizę radaru. Zwraca zdarzenia
/// `(kąt, intensywność)`, tak jak trafiają na radar. Próbki są podawane w kawałkach po 10 ms.
pub fn measure(
    samples: &[f32],
    channels: usize,
    rate: f32,
    probe: &SharedAudioAnalysis,
    config: &RadarConfig,
) -> Vec<(f32, f32)> {
    let channels = channels.max(1);
    if let Ok(mut g) = probe.lock() {
        g.radar_config = config.clone();
        let _ = g.drain_direction_events();
    }
    let chunk = ((rate as f64 * CHUNK_S).round().max(1.0) as usize) * channels;
    let mut events = Vec::new();
    for part in samples.chunks(chunk) {
        devices::update_analysis(part, channels, rate, probe, None);
        if let Ok(mut g) = probe.lock() {
            events.extend(
                g.drain_direction_events()
                    .into_iter()
                    .map(|queued| (queued.event.angle_deg, queued.event.intensity)),
            );
        }
    }
    events
}

/// Odtwarzana próba. Strumień trzeba trzymać przy życiu, aż próba się skończy.
pub struct Burst {
    pub stream: cpal::Stream,
    /// Stereo z dokładnie tym, co słychać w słuchawkach.
    pub frames: Vec<[f32; 2]>,
    pub rate: u32,
}

/// Odtwarza próbę na urządzeniu wyjścia pasującym do `device_name`, w jego częstotliwości próbkowania.
pub fn play(device_name: &str, angle_deg: f32, signal: Signal) -> Result<Burst> {
    let device = find_output_device(device_name)?;
    let supported = device.default_output_config()?;
    let config: StreamConfig = supported.clone().into();
    let rate = config.sample_rate.0;
    let frames = burst_stereo(angle_deg, signal, rate as f32);
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config, frames.clone())?,
        SampleFormat::I16 => build::<i16>(&device, &config, frames.clone())?,
        fmt => bail!("nieobsługiwany format próbek: {fmt}"),
    };
    stream.play()?;
    Ok(Burst { stream, frames, rate })
}

fn build<T>(device: &cpal::Device, config: &StreamConfig, frames: Vec<[f32; 2]>) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut index = 0usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            for frame in data.chunks_mut(channels) {
                let [l, r] = frames.get(index).copied().unwrap_or([0.0, 0.0]);
                index = index.saturating_add(1);
                for (c, sample) in frame.iter_mut().enumerate() {
                    let value = match (channels, c) {
                        (1, _) => 0.5 * (l + r),
                        (_, 0) => l,
                        (_, 1) => r,
                        _ => 0.0,
                    };
                    *sample = T::from_sample(value);
                }
            }
        },
        |err| eprintln!("Błąd testu kierunków: {err}"),
        None,
    )?;
    Ok(stream)
}

/// Jedna próba: dźwięk był z `expected_deg`, użytkownik wskazał `perceived_deg`, radar pokazał
/// zdarzenia `radar_events` (kąt, intensywność).
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub expected_deg: f32,
    pub perceived_deg: Option<f32>,
    pub radar_events: Vec<(f32, f32)>,
}

impl Trial {
    pub fn radar_deg(&self) -> Option<f32> {
        radar_reading(&self.radar_events)
    }
}

/// Dane sesji dołączane do każdego wiersza CSV.
pub struct RowContext {
    pub unix_ts_ms: u128,
    pub trial_no: usize,
    pub signal: Signal,
    pub route: Route,
    /// `true` przy wskazaniu myszką, `false` przy przycisku kierunku.
    pub pointer: bool,
    pub channels: usize,
    pub sensitivity: f32,
}

pub const CSV_HEADER: &str = "unix_ts_ms,trial,signal,route,answer,channels,sensitivity,expected_deg,perceived_deg,perceived_err_deg,radar_deg,radar_err_deg,radar_events,radar_all\n";

/// Jeden wiersz CSV. Błędy to różnice ze znakiem, w `(-180, 180]`. Brakujące wartości zostają puste.
pub fn csv_row(ctx: &RowContext, trial: &Trial) -> String {
    let num = |v: Option<f32>| v.map_or_else(String::new, |a| format!("{a:.1}"));
    let radar = trial.radar_deg();
    let perceived_err = trial.perceived_deg.map(|p| wrap_deg(p - trial.expected_deg));
    let radar_err = radar.map(|r| wrap_deg(r - trial.expected_deg));
    let all: Vec<String> = trial
        .radar_events
        .iter()
        .map(|(angle, weight)| format!("{angle:.1}:{weight:.2}"))
        .collect();
    format!(
        "{},{},{},{},{},{},{:.2},{:.1},{},{},{},{},{},{}\n",
        ctx.unix_ts_ms,
        ctx.trial_no,
        ctx.signal.key(),
        ctx.route.key(),
        if ctx.pointer { "pointer" } else { "button" },
        ctx.channels,
        ctx.sensitivity,
        trial.expected_deg,
        num(trial.perceived_deg),
        num(perceived_err),
        num(radar),
        num(radar_err),
        trial.radar_events.len(),
        all.join(";"),
    )
}

/// Zbiorcze błędy jednego kierunku (albo wszystkich prób, gdy `expected_deg` to `None`).
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub expected_deg: Option<f32>,
    pub trials: usize,
    /// Próby, w których radar nie pokazał żadnego zdarzenia.
    pub radar_missing: usize,
    /// Średni błąd ze znakiem: dodatni = w prawo od prawdziwego kierunku.
    pub radar_bias_deg: Option<f32>,
    /// Średni błąd bezwzględny (o ile stopni średnio).
    pub radar_mae_deg: Option<f32>,
    pub heard_bias_deg: Option<f32>,
    pub heard_mae_deg: Option<f32>,
}

/// Wiersz dla każdego kierunku testu, a na końcu wiersz zbiorczy „wszystkie”.
pub fn summarize(trials: &[Trial]) -> Vec<Summary> {
    let mut rows: Vec<Summary> = DIRECTIONS_DEG
        .iter()
        .map(|&direction| {
            let group: Vec<&Trial> = trials
                .iter()
                .filter(|t| wrap_deg(t.expected_deg - direction).abs() < 0.5)
                .collect();
            summarize_group(&group, Some(direction))
        })
        .collect();
    let all: Vec<&Trial> = trials.iter().collect();
    rows.push(summarize_group(&all, None));
    rows
}

fn summarize_group(group: &[&Trial], expected_deg: Option<f32>) -> Summary {
    let radar: Vec<f32> = group
        .iter()
        .filter_map(|t| t.radar_deg().map(|r| wrap_deg(r - t.expected_deg)))
        .collect();
    let heard: Vec<f32> = group
        .iter()
        .filter_map(|t| t.perceived_deg.map(|p| wrap_deg(p - t.expected_deg)))
        .collect();
    Summary {
        expected_deg,
        trials: group.len(),
        radar_missing: group.iter().filter(|t| t.radar_deg().is_none()).count(),
        radar_bias_deg: mean(&radar),
        radar_mae_deg: mean(&radar.iter().map(|e| e.abs()).collect::<Vec<_>>()),
        heard_bias_deg: mean(&heard),
        heard_mae_deg: mean(&heard.iter().map(|e| e.abs()).collect::<Vec<_>>()),
    }
}

fn mean(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

/// Wpuszcza próbę w analizę na żywo w czasie rzeczywistym, tak jak przechwycone audio z gry:
/// `live` dostaje ją przez `update_analysis`, więc mono router gra ją ze swoimi ustawieniami, a radar
/// pokazuje kropki. Równolegle `probe` liczy zdarzenia do wyniku. Wątek kończy się po próbie (albo
/// gdy `stop` zostanie ustawione) i zwraca zdarzenia `(kąt, intensywność)`.
pub fn feed_through_radar(
    input: Vec<f32>,
    channels: usize,
    rate: f32,
    live: SharedAudioAnalysis,
    probe: SharedAudioAnalysis,
    config: RadarConfig,
    stop: Arc<AtomicBool>,
) -> JoinHandle<Vec<(f32, f32)>> {
    thread::spawn(move || {
        let channels = channels.max(1);
        let chunk = ((rate as f64 * CHUNK_S).round().max(1.0) as usize) * channels;
        if let Ok(mut g) = probe.lock() {
            g.radar_config = config;
        }
        let started = Instant::now();
        let mut events = Vec::new();
        for (k, part) in input.chunks(chunk).enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // Tempo od startu, a nie suma `sleep`, żeby błędy nie narastały.
            let due = started + Duration::from_secs_f64(k as f64 * CHUNK_S);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }
            devices::update_analysis(part, channels, rate, &live, None);
            devices::update_analysis(part, channels, rate, &probe, None);
            if let Ok(mut g) = probe.lock() {
                events.extend(
                    g.drain_direction_events()
                        .into_iter()
                        .map(|queued| (queued.event.angle_deg, queued.event.intensity)),
                );
            }
        }
        events
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 48_000.0;

    fn energy(frames: &[[f32; 2]], channel: usize) -> f32 {
        frames.iter().map(|f| f[channel] * f[channel]).sum()
    }

    fn first_sound(frames: &[[f32; 2]], channel: usize) -> usize {
        frames.iter().position(|f| f[channel].abs() > 1e-6).unwrap_or(usize::MAX)
    }

    #[test]
    fn every_trial_sound_lasts_exactly_one_second() {
        for signal in Signal::ALL {
            for angle in DIRECTIONS_DEG {
                assert_eq!(burst_stereo(angle, signal, RATE).len(), RATE as usize);
            }
        }
    }

    #[test]
    fn level_signal_sends_a_hard_right_source_to_the_right_ear_only() {
        let frames = burst_stereo(90.0, Signal::Level, RATE);
        assert!(energy(&frames, 0) < 1e-6 * energy(&frames, 1));
    }

    #[test]
    fn level_signal_is_centred_for_a_source_straight_ahead() {
        let frames = burst_stereo(0.0, Signal::Level, RATE);
        let (l, r) = (energy(&frames, 0), energy(&frames, 1));
        assert!((l - r).abs() < 1e-3 * l, "L {l} vs R {r}");
    }

    #[test]
    fn itd_signal_delays_the_far_ear_and_keeps_levels_equal() {
        let frames = burst_stereo(90.0, Signal::Itd, RATE);
        let expected = (DEFAULT_EAR_DISTANCE_M / SPEED_OF_SOUND_M_S * RATE).round() as usize;
        let gap = first_sound(&frames, 0) - first_sound(&frames, 1);
        assert_eq!(gap, expected);
        assert!((energy(&frames, 0) - energy(&frames, 1)).abs() < 1e-3);
    }

    #[test]
    fn to_channels_puts_left_and_right_on_the_front_pair() {
        assert_eq!(to_channels(&[[0.5, -0.25]], 8), vec![0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(to_channels(&[[1.0, 0.0]], 1), vec![0.5]);
    }

    #[test]
    fn pointer_angles_follow_the_radar_convention() {
        assert!(angle_from_pointer(0.0, -10.0).abs() < 1e-3, "przód");
        assert!((angle_from_pointer(10.0, 0.0) - 90.0).abs() < 1e-3, "prawo");
        assert!((angle_from_pointer(0.0, 10.0).abs() - 180.0).abs() < 1e-3, "tył");
        assert!((angle_from_pointer(-10.0, 0.0) + 90.0).abs() < 1e-3, "lewo");
    }

    #[test]
    fn shuffled_keeps_every_direction_the_requested_number_of_times() {
        let order = shuffled(&DIRECTIONS_DEG, 3, 42);
        assert_eq!(order.len(), 24);
        for d in DIRECTIONS_DEG {
            assert_eq!(order.iter().filter(|&&a| a == d).count(), 3);
        }
        assert_ne!(order, shuffled(&DIRECTIONS_DEG, 3, 43), "inny seed, inna kolejność");
    }

    #[test]
    fn radar_reading_is_the_intensity_weighted_mean_of_its_events() {
        assert_eq!(radar_reading(&[]), None);
        assert!((radar_reading(&[(45.0, 1.0)]).unwrap() - 45.0).abs() < 1e-3);
        assert!(radar_reading(&[(-45.0, 1.0), (45.0, 1.0)]).unwrap().abs() < 1e-3, "dwa równe: środek");
        let skewed = radar_reading(&[(-45.0, 0.2), (45.0, 0.8)]).unwrap();
        assert!((skewed - 31.0).abs() < 0.5, "{skewed}");
    }

    #[test]
    fn summary_separates_radar_and_heard_errors() {
        let trials = [
            Trial { expected_deg: 90.0, perceived_deg: Some(90.0), radar_events: vec![(45.0, 1.0)] },
            Trial { expected_deg: 90.0, perceived_deg: Some(60.0), radar_events: vec![] },
        ];
        let rows = summarize(&trials);
        let right = rows.iter().find(|r| r.expected_deg == Some(90.0)).unwrap();
        assert_eq!(right.trials, 2);
        assert_eq!(right.radar_missing, 1);
        assert!((right.radar_bias_deg.unwrap() + 45.0).abs() < 1e-3);
        assert!((right.heard_mae_deg.unwrap() - 15.0).abs() < 1e-3);
        assert_eq!(rows.last().unwrap().expected_deg, None);
        assert_eq!(rows.last().unwrap().trials, 2);
    }

    #[test]
    fn csv_row_has_one_field_per_header_column_and_leaves_missing_readings_empty() {
        let ctx = RowContext {
            unix_ts_ms: 1,
            trial_no: 2,
            signal: Signal::Itd,
            route: Route::Direct,
            pointer: true,
            channels: 8,
            sensitivity: 20.0,
        };
        let trial = Trial { expected_deg: 90.0, perceived_deg: None, radar_events: vec![(45.0, 0.5)] };
        let row = csv_row(&ctx, &trial);
        assert_eq!(row.matches(',').count(), CSV_HEADER.matches(',').count());
        assert_eq!(row, "1,2,itd,direct,pointer,8,20.00,90.0,,,45.0,-45.0,1,45.0:0.50\n");
    }

    #[test]
    fn a_hard_right_level_burst_is_heard_on_the_right_of_the_radar() {
        let probe = SharedAudioAnalysis::new();
        let samples = probe_input(&burst_stereo(90.0, Signal::Level, RATE), DEFAULT_CHANNELS, RATE);
        let events = measure(&samples, DEFAULT_CHANNELS, RATE, &probe, &RadarConfig::default());
        let reading = radar_reading(&events).expect("próba z prawej powinna dać zdarzenie");
        assert!(reading > 0.0, "radar: {reading}");
    }

    #[test]
    fn feeding_through_the_live_analysis_measures_what_measuring_directly_does() {
        let input = probe_input(&burst_stereo(90.0, Signal::Level, RATE), DEFAULT_CHANNELS, RATE);
        let direct = measure(&input, DEFAULT_CHANNELS, RATE, &SharedAudioAnalysis::new(), &RadarConfig::default());
        assert!(!direct.is_empty(), "próba z prawej powinna dać zdarzenie");

        let live = SharedAudioAnalysis::new();
        let worker = feed_through_radar(
            input,
            DEFAULT_CHANNELS,
            RATE,
            live.clone(),
            SharedAudioAnalysis::new(),
            RadarConfig::default(),
            Arc::new(AtomicBool::new(false)),
        );
        let through = worker.join().expect("wątek próby nie powinien paść");

        assert_eq!(through.len(), direct.len());
        for (a, b) in through.iter().zip(&direct) {
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4, "{a:?} vs {b:?}");
        }
        assert_eq!(live.lock().unwrap().channel_count, DEFAULT_CHANNELS);
    }

    /// Raport do ręcznego odczytu: co radar pokazuje dla każdego kierunku i sygnału.
    /// Uruchomienie: `cargo test print_radar_response -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn print_radar_response() {
        for signal in Signal::ALL {
            println!("sygnał: {}", signal.label());
            for &angle in &DIRECTIONS_DEG {
                let probe = SharedAudioAnalysis::new();
                let samples = probe_input(&burst_stereo(angle, signal, RATE), DEFAULT_CHANNELS, RATE);
                let events = measure(&samples, DEFAULT_CHANNELS, RATE, &probe, &RadarConfig::default());
                let reading = radar_reading(&events).map_or("brak".to_string(), |r| format!("{r:+.1}°"));
                let dots: Vec<String> = events.iter().map(|(a, w)| format!("{a:+.0}°/{w:.2}")).collect();
                println!("  {angle:>6.0}°  radar {reading:>8}  kropki [{}]", dots.join(" "));
            }
        }
    }
}
