//! Mono router — odtwarza w jednej słuchawce (albo obu) zmiksowany sygnał
//! ze wszystkich kanałów, korzystając z TEGO SAMEGO strumienia wejściowego,
//! który już zasila radar. Nie otwiera własnego, niezależnego przechwytywania
//! wejścia — to tylko strumień WYJŚCIOWY, czytający gotowe próbki mono z
//! bufora kołowego w `SharedAudioAnalysis` (patrz `push_mono`/`read_mono`
//! w `sound::devices`), do którego pisze `update_analysis` przy każdej
//! ramce audio.
//!
//! Przepływ sygnału:
//!   [urządzenie wejściowe] ──► input callback (`devices::fetch_audio_from_device`)
//!         │
//!         ├──► radar (kropki per-kanał, kierunek — bez zmian)
//!         │
//!         └──► downmix do mono ──► RingBuffer w `SharedAudioAnalysis`
//!                                       │
//!                                 output callback ──► [słuchawki]
//!                                       L  R
//!                                    ┌─────────┐
//!                         Left  →    │ mono  0 │
//!                         Right →    │  0  mono│
//!                         Both  →    │ mono mono│
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
    let err_fn = |e| eprintln!("[mono_router] output error: {e}");

    // Wyjście obsługujemy tylko jako F32 — konwertujemy jeśli potrzeba.
    // Sterowanie lewym/prawym uchem przez `analysis.mono_ear`.
    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |out: &mut [f32], _| {
                let frames = out.len() / out_channels.max(1);
                let mono = analysis.read_mono(frames);
                let ear = analysis.lock().map(|g| g.mono_ear).unwrap_or(MonoEar::Off);
                fill_output_f32(out, &mono, out_channels, ear);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_output_stream(
            &stream_config,
            move |out: &mut [i16], _| {
                let frames = out.len() / out_channels.max(1);
                let mono = analysis.read_mono(frames);
                let ear = analysis.lock().map(|g| g.mono_ear).unwrap_or(MonoEar::Off);
                // Konwersja f32 → i16
                let mut tmp = vec![0.0f32; out.len()];
                fill_output_f32(&mut tmp, &mono, out_channels, ear);
                for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                    *dst = (src * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_output_stream(
            &stream_config,
            move |out: &mut [u16], _| {
                let frames = out.len() / out_channels.max(1);
                let mono = analysis.read_mono(frames);
                let ear = analysis.lock().map(|g| g.mono_ear).unwrap_or(MonoEar::Off);
                let mut tmp = vec![0.0f32; out.len()];
                fill_output_f32(&mut tmp, &mono, out_channels, ear);
                for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                    // f32 (-1..1) → u16 (0..65535), środek = 32768
                    *dst = ((src * 0.5 + 0.5) * u16::MAX as f32)
                        .clamp(0.0, u16::MAX as f32) as u16;
                }
            },
            err_fn,
            None,
        )?,
        fmt => bail!("[mono_router] unsupported output format: {fmt}"),
    };

    Ok(stream)
}

// ── Wypełnianie bufora wyjściowego ───────────────────────────────────────────

/// Zapisuje `mono` do bufora wyjściowego interleaved.
/// Kanały poza `out_channels` są zerowane.
/// `ear` decyduje który kanał stereo dostaje sygnał mono.
#[inline]
fn fill_output_f32(out: &mut [f32], mono: &[f32], out_channels: usize, ear: MonoEar) {
    let ch = out_channels.max(1);
    for (frame_idx, frame) in out.chunks_exact_mut(ch).enumerate() {
        let sample = mono.get(frame_idx).copied().unwrap_or(0.0);
        for s in frame.iter_mut() {
            *s = 0.0;
        }
        match ear {
            MonoEar::Off => { /* cisza */ }
            MonoEar::Left => frame[0] = sample,
            MonoEar::Right if ch >= 2 => frame[1] = sample,
            MonoEar::Right => frame[0] = sample, // mono device fallback
            MonoEar::Both => {
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
/// czytając próbki mono z bufora, do którego pisze strumień wejściowy
/// zasilający radar (patrz komentarz na górze pliku — MUSI on już działać,
/// inaczej `read_mono` będzie zwracać ciszę).
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
    println!(
        "[mono_router] output: {:?} | {}ch | {}Hz | {:?}",
        output_device_name,
        output_config.channels(),
        output_config.sample_rate().0,
        output_config.sample_format(),
    );

    let output_stream = build_output_stream(&output_device, &output_config, analysis)?;
    output_stream.play()?;

    println!("[mono_router] routing active");

    Ok(MonoRouter { _output_stream: output_stream })
}
