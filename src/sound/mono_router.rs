//! Mono router — odtwarza w jednej słuchawce (albo obu) zmiksowany sygnał
//! ze wszystkich kanałów, korzystając z TEGO SAMEGO strumienia wejściowego,
//! który już zasila radar. Nie otwiera własnego, niezależnego przechwytywania
//! wejścia — to tylko strumień WYJŚCIOWY, czytający gotowe próbki "ważonego
//! mono" z bufora kołowego w `SharedAudioAnalysis` (patrz
//! `push_stereo_pan`/`read_stereo_pan` w `sound::devices`), do którego pisze
//! `update_analysis` przy każdej ramce audio.
//!
//! Przepływ sygnału:
//!   [urządzenie wejściowe] ──► input callback (`devices::fetch_audio_from_device`)
//!         │
//!         ├──► radar (kropki per-kanał, kierunek — bez zmian)
//!         │
//!         └──► downmix do (left, right) ──► RingBuffer w `SharedAudioAnalysis`
//!                                       │
//!                                 output callback ──► [słuchawki]
//!                                       L              R
//!                                    ┌───────────────────────────┐
//!                         Left  →    │ L+ringmod(R)      0       │
//!                         Right →    │      0       R+ringmod(L) │
//!                         Both  →    │    L+R            L+R     │
//!                                    └───────────────────────────┘
//! W trybach jednousznych (`Left`/`Right`) strona PRZECIWNA do granego ucha
//! nie jest po prostu domiksowana na płasko — dostaje ring modulation (patrz
//! `fill_output_f32`), żeby w jednej słuchawce dało się usłyszeć, że dźwięk
//! oryginalnie był z drugiej strony, zamiast bezpowrotnie tracić informację
//! L/R już na etapie miksu do mono.
//!
//! Wcześniej `mono_router` otwierał WŁASNY, drugi strumień wejściowy na
//! nazwie urządzenia z tej samej listy co radar. Na Windows ta lista
//! (`devices::fetch_output_devices`) zawiera nazwy urządzeń WYJŚCIOWYCH (do
//! przechwytywania pętlą zwrotną WASAPI) — a lookup szukał ich tylko wśród
//! `host.input_devices()`, więc zawsze zawodził i routing w ogóle nie
//! startował (błąd widoczny tylko w konsoli). Ujednolicenie do jednego
//! strumienia wejściowego usuwa ten błąd u źródła, zamiast łatać sam lookup.

use anyhow::{Result, bail};
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::sound::devices::{MonoEar, SharedAudioAnalysis};

/// Częstotliwość (Hz) modulacji pierścieniowej ("brzęczenia") nakładanej na
/// dźwięk z PRZECIWNEJ strony niż aktualne ucho w trybie jednousznym (patrz
/// [`fill_output_f32`]) — na tyle nisko, żeby brzmieć jak wyraźna, chropowata
/// wibracja odróżnialna od czystego dźwięku, a nie po prostu "wyższy ton".
const RING_MOD_FREQ_HZ: f32 = 45.0;

// ── Uchwyt do strumienia wyjściowego ─────────────────────────────────────────

/// Trzymaj ten obiekt przy życiu tak długo, jak ma grać routing.
/// Drop zatrzymuje strumień wyjściowy cpal.
pub struct MonoRouter {
    _output_stream: cpal::Stream,
}

// ── Budowanie strumienia wyjściowego ─────────────────────────────────────────

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let stream_config: StreamConfig = config.clone().into();
    let out_channels = stream_config.channels as usize;
    let sample_rate = stream_config.sample_rate.0 as f32;
    let err_fn = |e| eprintln!("[mono_router] output error: {e}");

    // Wyjście obsługujemy tylko jako F32 — konwertujemy jeśli potrzeba.
    // Sterowanie lewym/prawym uchem przez `analysis.mono_ear`. `mod_phase`
    // to stan oscylatora ring modulation (patrz `fill_output_f32`) — musi
    // przetrwać między kolejnymi wywołaniami callbacku, żeby fala nie
    // "przeskakiwała" fazy i nie trzaskała na granicy buforów.
    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let mut mod_phase = 0.0f32;
            device.build_output_stream(
                &stream_config,
                move |out: &mut [f32], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let (ear, gain, ring_mod_enabled) = read_ear_and_gain(&analysis);
                    fill_output_f32(out, &pan, out_channels, ear, gain, ring_mod_enabled, sample_rate, &mut mod_phase);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut mod_phase = 0.0f32;
            device.build_output_stream(
                &stream_config,
                move |out: &mut [i16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let (ear, gain, ring_mod_enabled) = read_ear_and_gain(&analysis);
                    // Konwersja f32 → i16
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, out_channels, ear, gain, ring_mod_enabled, sample_rate, &mut mod_phase);
                    for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                        *dst = (src * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut mod_phase = 0.0f32;
            device.build_output_stream(
                &stream_config,
                move |out: &mut [u16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let (ear, gain, ring_mod_enabled) = read_ear_and_gain(&analysis);
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, out_channels, ear, gain, ring_mod_enabled, sample_rate, &mut mod_phase);
                    for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                        // f32 (-1..1) → u16 (0..65535), środek = 32768
                        *dst = ((src * 0.5 + 0.5) * u16::MAX as f32)
                            .clamp(0.0, u16::MAX as f32) as u16;
                    }
                },
                err_fn,
                None,
            )?
        }
        fmt => bail!("[mono_router] unsupported output format: {fmt}"),
    };

    Ok(stream)
}

/// Konwertuje decybele na liniowy mnożnik amplitudy (0 dB = 1.0×, +6 dB ≈
/// 2×, -6 dB ≈ 0.5×) — standardowy wzór `10^(dB/20)`.
fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Czyta `mono_ear`, `radar_config.mono_boost_db` i
/// `radar_config.mono_ring_mod_enabled` jednym lockiem i od razu zamienia dB
/// na mnożnik — wołane raz na każdą ramkę callbacku wyjściowego.
fn read_ear_and_gain(analysis: &SharedAudioAnalysis) -> (MonoEar, f32, bool) {
    let (ear, boost_db, ring_mod_enabled) = analysis
        .lock()
        .map(|g| (g.mono_ear, g.radar_config.mono_boost_db, g.radar_config.mono_ring_mod_enabled))
        .unwrap_or((MonoEar::Off, 0.0, true));
    (ear, db_to_linear(boost_db), ring_mod_enabled)
}

// ── Wypełnianie bufora wyjściowego ───────────────────────────────────────────

/// Zapisuje `pan` (pary "ważonego mono" `(left, right)`, patrz
/// [`crate::sound::devices::SharedAudioAnalysis::read_stereo_pan`]) do
/// bufora wyjściowego interleaved, ze wzmocnieniem `gain` (mnożnik liniowy,
/// patrz [`db_to_linear`]) — dociśnięte do `-1.0..=1.0`, żeby duży boost dawał
/// głośniej, a nie zawinięcie/trzaski przy konwersji do i16/u16 niżej w
/// łańcuchu. Kanały poza `out_channels` są zerowane.
///
/// `ear` decyduje który kanał stereo dostaje sygnał — w trybach jednousznych
/// (`Left`/`Right`), gdy `ring_mod_enabled` jest `true`, strona PRZECIWNA do
/// granego ucha nie jest po prostu domiksowywana "na płasko": dostaje ring
/// modulation (mnożenie przez falę sinusoidalną o częstotliwości
/// [`RING_MOD_FREQ_HZ`], stan fazy w `mod_phase`, trzymany przez
/// wywołującego między callbackami), żeby dało się usłyszeć w jednej
/// słuchawce, że dźwięk oryginalnie był z drugiej strony — bez tego cała
/// informacja L/R ginęłaby w trybie jednousznym. `ring_mod_enabled = false`
/// (checkbox w UI, `RadarConfig::mono_ring_mod_enabled`) wraca do starego
/// zachowania: strona przeciwna gra na pełnej głośności, tak samo jak swoja,
/// bez żadnego oznaczenia — dla osób, którym brzęczenie przeszkadza.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fill_output_f32(
    out: &mut [f32],
    pan: &[(f32, f32)],
    out_channels: usize,
    ear: MonoEar,
    gain: f32,
    ring_mod_enabled: bool,
    sample_rate: f32,
    mod_phase: &mut f32,
) {
    let ch = out_channels.max(1);
    let phase_step = RING_MOD_FREQ_HZ / sample_rate.max(1.0);
    for (frame_idx, frame) in out.chunks_exact_mut(ch).enumerate() {
        let (left, right) = pan.get(frame_idx).copied().unwrap_or((0.0, 0.0));
        *mod_phase = (*mod_phase + phase_step).fract();
        let ring_gain = if ring_mod_enabled {
            0.5 + 0.5 * (*mod_phase * std::f32::consts::TAU).sin()
        } else {
            1.0
        };

        for s in frame.iter_mut() {
            *s = 0.0;
        }
        match ear {
            MonoEar::Off => { /* cisza */ }
            MonoEar::Left => {
                frame[0] = ((left + right * ring_gain) * gain).clamp(-1.0, 1.0);
            }
            MonoEar::Right if ch >= 2 => {
                frame[1] = ((right + left * ring_gain) * gain).clamp(-1.0, 1.0);
            }
            MonoEar::Right => {
                // mono device fallback
                frame[0] = ((right + left * ring_gain) * gain).clamp(-1.0, 1.0);
            }
            MonoEar::Both => {
                // Oba uszy już słyszą osobno lewą/prawą stronę (stereo
                // wyjście) — bez ring modulation, bo cue nie jest tu potrzebne.
                let sample = ((left + right) * gain).clamp(-1.0, 1.0);
                frame[0] = sample;
                if ch >= 2 {
                    frame[1] = sample;
                }
            }
        }
    }
}

// ── Publiczne API ─────────────────────────────────────────────────────────────

/// Uruchamia routing mono na wybranym urządzeniu wyjściowym (słuchawki),
/// czytając próbki "ważonego mono" z bufora, do którego pisze strumień
/// wejściowy zasilający radar (patrz komentarz na górze pliku — MUSI on już
/// działać, inaczej `read_stereo_pan` będzie zwracać ciszę).
///
/// - `output_device_name` — nazwa fizycznego urządzenia wyjściowego (np. słuchawki)
/// - `analysis`           — współdzielony stan; musi mieć ustawione `mono_ear`
///
/// Zwraca `MonoRouter`. Drop zatrzymuje strumień.
pub fn start_mono_output(
    output_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<MonoRouter> {
    let host = cpal::default_host();

    let output_device = host
        .output_devices()?
        .find(|d| d.name().map(|n| n == output_device_name).unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("Output device not found: {}", output_device_name))?;

    let output_config = output_device.default_output_config()?;
    crate::dlog!(
        "[mono_router] output: {:?} | {}ch | {}Hz | {:?}",
        output_device_name,
        output_config.channels(),
        output_config.sample_rate().0,
        output_config.sample_format(),
    );

    let output_stream = build_output_stream(&output_device, &output_config, analysis)?;
    output_stream.play()?;

    crate::dlog!("[mono_router] routing active");

    Ok(MonoRouter { _output_stream: output_stream })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kilka klatek stałej pary `(left, right)`, żeby oscylator ring
    /// modulation zdążył przejść przez pełny cykl i test złapał zarówno
    /// szczyt, jak i dolinę fali.
    fn run_frames(pan: &(f32, f32), n: usize, out_channels: usize, ear: MonoEar) -> Vec<f32> {
        run_frames_with_ring_mod(pan, n, out_channels, ear, true)
    }

    fn run_frames_with_ring_mod(
        pan: &(f32, f32),
        n: usize,
        out_channels: usize,
        ear: MonoEar,
        ring_mod_enabled: bool,
    ) -> Vec<f32> {
        let pan_frames = vec![*pan; n];
        let mut out = vec![0.0f32; n * out_channels];
        let mut mod_phase = 0.0f32;
        fill_output_f32(&mut out, &pan_frames, out_channels, ear, 1.0, ring_mod_enabled, 48_000.0, &mut mod_phase);
        out
    }

    #[test]
    fn left_ear_passes_left_side_unmodulated() {
        // Czysty dźwięk z lewej (right=0) — kanał 0 (lewa słuchawka) ma
        // zawsze dokładnie tę samą wartość co wejście, niezależnie od fazy
        // ring modulation (bo modulujemy tylko `right`, a ten jest zerowy).
        let out = run_frames(&(0.7, 0.0), 64, 2, MonoEar::Left);
        for frame in out.chunks_exact(2) {
            assert!((frame[0] - 0.7).abs() < 1e-6);
            assert_eq!(frame[1], 0.0, "prawy kanał wyjścia ma być cichy w trybie Left");
        }
    }

    #[test]
    fn left_ear_ring_modulates_the_right_side_instead_of_silencing_it() {
        // Czysty dźwięk z prawej (left=0) grany w trybie Left: nie może być
        // ani ciszą (informacja by zginęła), ani stałą wartością (nie dałoby
        // się jej odróżnić od dźwięku z lewej) — musi się wahać z ring
        // modulation.
        let out = run_frames(&(0.0, 0.7), 64, 2, MonoEar::Left);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        assert!(left_channel.iter().any(|&s| s.abs() > 1e-6), "sygnał z prawej nie może zniknąć");
        let max = left_channel.iter().cloned().fold(f32::MIN, f32::max);
        let min = left_channel.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max - min > 0.1, "ring modulation powinien wyraźnie wahać amplitudę: min={min} max={max}");
    }

    #[test]
    fn ring_mod_disabled_plays_the_opposite_side_at_full_unmodulated_strength() {
        // Checkbox wyłączony: strona przeciwna wraca do dawnego, prostego
        // zachowania sprzed tej funkcji — pełna głośność, stała wartość, bez
        // wahania — zamiast ring modulation.
        let out = run_frames_with_ring_mod(&(0.0, 0.7), 16, 2, MonoEar::Left, false);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        for &s in &left_channel {
            assert!((s - 0.7).abs() < 1e-6, "bez ring mod sygnał z prawej ma grać niezmieniony: {s}");
        }
    }

    #[test]
    fn both_ears_mode_matches_flat_sum_without_modulation() {
        let out = run_frames(&(0.3, 0.2), 8, 2, MonoEar::Both);
        for frame in out.chunks_exact(2) {
            assert!((frame[0] - 0.5).abs() < 1e-6);
            assert!((frame[1] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn off_produces_silence() {
        let out = run_frames(&(0.9, 0.9), 8, 2, MonoEar::Off);
        assert!(out.iter().all(|&s| s == 0.0));
    }
}
