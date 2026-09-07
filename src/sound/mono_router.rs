//! Mono router — łączy strumień wejściowy ze strumieniem wyjściowym wewnątrz
//! procesu bez tworzenia wirtualnego urządzenia systemowego.
//!
//! Przepływ sygnału:
//!   [urządzenie wejściowe] ──► input callback
//!         │
//!         ├──► update_analysis()   (kąt + intensywność → radar)
//!         │
//!         └──► RingBuffer<f32>     (próbki mono)
//!                   │
//!             output callback ──► [urządzenie wyjściowe / słuchawki]
//!                                       L  R
//!                                    ┌─────────┐
//!                         Left  →    │ mono  0 │
//!                         Right →    │  0  mono│
//!                         Both  →    │ mono mono│
//!
//! Oba strumienie działają w osobnych wątkach audio sterowanych przez cpal.
//! `RingBuffer` jest lock-free, oparty na `Arc<Mutex<VecDeque>>` —
//! wystarczająco szybki dla buforów ~4096 próbek przy typowych latencjach.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::sound::devices::{MonoEar, SharedAudioAnalysis};
use crate::sound::direction::{self, analyze_channels};

// ── RingBuffer ────────────────────────────────────────────────────────────────

/// Współdzielony bufor kołowy próbek mono między callbackiem wejściowym
/// a callbackiem wyjściowym.
#[derive(Clone)]
struct RingBuffer(Arc<Mutex<VecDeque<f32>>>);

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(VecDeque::with_capacity(capacity))))
    }

    /// Zapisuje próbki do bufora. Jeśli bufor jest pełny, odrzuca najstarsze.
    fn write(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.0.lock() {
            for &s in samples {
                if buf.len() >= buf.capacity() {
                    buf.pop_front();
                }
                buf.push_back(s);
            }
        }
    }

    /// Czyta dokładnie `count` próbek. Zwraca próbki + ile było dostępnych.
    /// Jeśli w buforze jest mniej niż `count`, brakujące miejsca wypełnia ciszą.
    fn read(&self, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        if let Ok(mut buf) = self.0.lock() {
            for _ in 0..count {
                out.push(buf.pop_front().unwrap_or(0.0));
            }
        } else {
            out.resize(count, 0.0);
        }
        out
    }
}

// ── Uchwyty do obu strumieni ─────────────────────────────────────────────────

/// Trzymaj ten obiekt przy życiu tak długo, jak ma działać routing.
/// Drop zatrzymuje oba strumienie cpal.
pub struct MonoRouter {
    _input_stream:  cpal::Stream,
    _output_stream: cpal::Stream,
}

// ── Pomocnik konwersji próbek do f32 ─────────────────────────────────────────

fn to_f32_vec(data: &[i16]) -> Vec<f32> {
    data.iter().map(|&s| direction::i16_to_f32(s)).collect()
}

fn u16_to_f32_vec(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&s| direction::u16_to_f32(s)).collect()
}

// ── Miks N kanałów → mono ────────────────────────────────────────────────────

/// Zinterlacowane próbki `channels`-kanałowe → próbki mono (uśrednione).
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

// ── Callback wejściowy (wspólna logika niezależna od formatu) ─────────────────

fn process_input(
    f32_samples: &[f32],
    channels: usize,
    ring: &RingBuffer,
    analysis: &SharedAudioAnalysis,
) {
    // 1. Analiza kierunku — na wszystkich kanałach
    let stats = analyze_channels(f32_samples, channels);
    let (angle, intensity) = direction::compute_direction(&stats.rms);
    if let Ok(mut lock) = analysis.lock() {
        lock.angle     = angle;
        lock.intensity = intensity;
        lock.peak_l    = stats.peak.first().copied().unwrap_or(0.0);
        lock.peak_r    = stats.peak.get(1).copied().unwrap_or(0.0);
    }

    // 2. Downmix do mono → RingBuffer
    let mono = downmix_to_mono(f32_samples, channels);
    ring.write(&mono);
}

// ── Budowanie strumienia wejściowego ─────────────────────────────────────────

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring: RingBuffer,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let stream_config: StreamConfig = config.clone().into();
    let channels = stream_config.channels as usize;
    let err_fn = |e| eprintln!("[mono_router] input error: {e}");

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| process_input(data, channels, &ring, &analysis),
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let f = to_f32_vec(data);
                process_input(&f, channels, &ring, &analysis);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let f = u16_to_f32_vec(data);
                process_input(&f, channels, &ring, &analysis);
            },
            err_fn,
            None,
        )?,
        fmt => bail!("[mono_router] unsupported input format: {fmt}"),
    };

    Ok(stream)
}

// ── Budowanie strumienia wyjściowego ─────────────────────────────────────────

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring: RingBuffer,
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
                let mono = ring.read(frames);
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
                let mono = ring.read(frames);
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
                let mono = ring.read(frames);
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
            MonoEar::Off  => { /* cisza */ }
            MonoEar::Left  if ch >= 1 => frame[0] = sample,
            MonoEar::Right if ch >= 2 => frame[1] = sample,
            MonoEar::Right             => frame[0] = sample, // mono device fallback
            MonoEar::Left              => frame[0] = sample,
            MonoEar::Both => {
                frame[0] = sample;
                if ch >= 2 { frame[1] = sample; }
            }
        }
    }
}

// ── Publiczne API ─────────────────────────────────────────────────────────────

/// Uruchamia routing mono:
/// - `input_device_name`  — nazwa urządzenia wejściowego (np. BlackHole, mikrofon)
/// - `output_device_name` — nazwa fizycznego urządzenia wyjściowego (np. słuchawki)
/// - `analysis`           — współdzielony stan; musi mieć ustawione `mono_ear`
///
/// Zwraca `MonoRouter`. Drop zatrzymuje oba strumienie.
pub fn start_mono_router(
    input_device_name:  &str,
    output_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<MonoRouter> {
    let host = cpal::default_host();

    // ── Urządzenie wejściowe ──────────────────────────────────────────────
    let input_device = host
        .input_devices()?
        .find(|d| d.name().map(|n| n == input_device_name).unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("Input device not found: {}", input_device_name))?;

    let input_config = input_device.default_input_config()?;
    println!(
        "[mono_router] input:  {:?} | {}ch | {}Hz | {:?}",
        input_device_name,
        input_config.channels(),
        input_config.sample_rate().0,
        input_config.sample_format(),
    );

    // ── Urządzenie wyjściowe ──────────────────────────────────────────────
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

    // ── Współdzielony RingBuffer ──────────────────────────────────────────
    // Pojemność: ~200ms przy 48kHz → 9600 próbek, zaokrąglone do 16384
    let ring = RingBuffer::new(16_384);

    // ── Strumień wejściowy ────────────────────────────────────────────────
    let input_stream = build_input_stream(
        &input_device,
        &input_config,
        ring.clone(),
        analysis.clone(),
    )?;
    input_stream.play()?;

    // ── Strumień wyjściowy ────────────────────────────────────────────────
    let output_stream = build_output_stream(
        &output_device,
        &output_config,
        ring,
        analysis,
    )?;
    output_stream.play()?;

    println!("[mono_router] routing active");

    Ok(MonoRouter {
        _input_stream:  input_stream,
        _output_stream: output_stream,
    })
}
