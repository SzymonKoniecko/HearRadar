//! Sygnał synchronizacji: pół sekundy tonu 1 kHz wysyłanego na wyjście (np. CABLE Input) na starcie
//! nagrania. Ten sam dźwięk trafia do nagrania radaru i do nagrania z NVIDIA, więc na obu plikach
//! da się wskazać ten sam moment i je wyrównać.

use anyhow::{Result, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait, StreamTrait},
};
use std::f32::consts::TAU;

use crate::sound::channel_test::find_output_device;

/// Długość tonu w sekundach.
pub const SYNC_TONE_S: f32 = 0.5;
const SYNC_TONE_HZ: f32 = 1_000.0;
const SYNC_TONE_AMPLITUDE: f32 = 0.5;
/// Krótki narost i zanik, żeby początek i koniec tonu nie trzaskały.
const FADE_S: f32 = 0.01;

/// Próbka tonu o numerze `index` przy częstotliwości próbkowania `rate`. Po `SYNC_TONE_S` cisza.
pub fn sync_tone_sample(index: usize, rate: f32) -> f32 {
    let t = index as f32 / rate;
    if t >= SYNC_TONE_S {
        return 0.0;
    }
    let envelope = (t / FADE_S).min((SYNC_TONE_S - t) / FADE_S).clamp(0.0, 1.0);
    (TAU * SYNC_TONE_HZ * t).sin() * SYNC_TONE_AMPLITUDE * envelope
}

/// Odtwarza ton na urządzeniu wyjścia pasującym do `device_name`. Strumień trzeba trzymać przy życiu,
/// aż ton się skończy. Upuszczenie go wcześniej przerywa odtwarzanie.
pub fn play(device_name: &str) -> Result<cpal::Stream> {
    let device = find_output_device(device_name)?;
    let supported = device.default_output_config()?;
    let config: StreamConfig = supported.clone().into();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config)?,
        SampleFormat::I16 => build::<i16>(&device, &config)?,
        fmt => bail!("nieobsługiwany format próbek: {fmt}"),
    };
    stream.play()?;
    Ok(stream)
}

fn build<T>(device: &cpal::Device, config: &StreamConfig) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let rate = config.sample_rate.0 as f32;
    let mut index = 0usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            for frame in data.chunks_mut(channels) {
                let value = T::from_sample(sync_tone_sample(index, rate));
                index = index.saturating_add(1);
                for sample in frame.iter_mut() {
                    *sample = value;
                }
            }
        },
        |err| eprintln!("Błąd sygnału synchronizacji: {err}"),
        None,
    )?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 48_000.0;

    #[test]
    fn tone_lasts_half_a_second() {
        let tone_samples = (SYNC_TONE_S * RATE) as usize;
        assert!((0..tone_samples).any(|i| sync_tone_sample(i, RATE).abs() > 0.1));
        assert_eq!(sync_tone_sample(tone_samples + 1, RATE), 0.0);
    }

    #[test]
    fn tone_fades_in_and_stays_within_amplitude() {
        assert!(sync_tone_sample(0, RATE).abs() < 1e-6, "początek łagodny");
        let peak = (0..(SYNC_TONE_S * RATE) as usize)
            .map(|i| sync_tone_sample(i, RATE).abs())
            .fold(0.0f32, f32::max);
        assert!(peak <= SYNC_TONE_AMPLITUDE + 1e-6 && peak > 0.4, "szczyt {peak}");
    }
}
