use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::sync::{Arc, Mutex};

use crate::sound::direction::{self, analyze_channels};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AudioAnalysis {
    pub angle: f32,       // -180.0° (tył-lewo) .. 0.0° (przód) .. +180.0° (tył-prawo)
    pub intensity: f32,   // 0.0 .. 1.0 (amplituda / głośność)
    pub peak_l: f32,
    pub peak_r: f32,
}

#[derive(Debug, Clone)]
pub struct SharedAudioAnalysis(pub Arc<Mutex<AudioAnalysis>>);

impl PartialEq for SharedAudioAnalysis {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for SharedAudioAnalysis {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(AudioAnalysis::default())))
    }
}

impl SharedAudioAnalysis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, AudioAnalysis>> {
        self.0.lock()
    }
}


#[derive(Debug, Clone)]
pub struct SoundDevice {
    pub id: usize,
    pub name: String,
    pub is_input: bool,
    pub is_default: bool,
    pub configs: Vec<DeviceConfig>,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub channels: u16,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub sample_format: cpal::SampleFormat,
    pub buffer_size: cpal::SupportedBufferSize,
}

impl SoundDevice {
    pub fn get_device_direction(&self) -> &'static str {
        if self.is_input { "Input" } else { "Output" }
    }

    pub fn get_default_status(&self) -> &'static str {
        if self.is_default { "Yes" } else { "No" }
    }
}

pub fn fetch_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();

    let default_device: Option<String> = host
        .default_output_device()
        .and_then(|device| device.name().ok());

    let mut devices: Vec<SoundDevice> = Vec::new();

    for (index, device) in host.output_devices()?.enumerate() {
        let Ok(device_name) = device.name() else {
            continue;
        };

        let is_default = default_device.as_deref() == Some(device_name.as_str());

        let mut device_configs: Vec<DeviceConfig> = Vec::new();

        match device.supported_output_configs() {
            Ok(supported_configs) => {
                for config in supported_configs {
                    device_configs.push(DeviceConfig {
                        channels: config.channels(),
                        min_sample_rate: config.min_sample_rate().0,
                        max_sample_rate: config.max_sample_rate().0,
                        sample_format: config.sample_format(),
                        buffer_size: *config.buffer_size(),
                    });
                }
            }

            Err(error) => {
                eprintln!(
                    "Could not read supported output configurations for \
                    '{device_name}': {error}"
                );
            }
        }

        devices.push(SoundDevice {
            id: index,
            name: device_name,
            is_input: false,
            is_default,
            configs: device_configs,
        });
    }

    Ok(devices)
}

pub fn fetch_audio_from_device(
    target_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    // 1. Szukamy najpierw w urządzeniach wyjściowych (dla natywnego WASAPI loopback na Windows)
    let (target_device, is_output) = if let Some(dev) = host
        .output_devices()?
        .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
    {
        (dev, true)
    } else if let Some(dev) = host
        .input_devices()?
        .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
    {
        (dev, false)
    } else {
        return Err(anyhow::anyhow!(
            "Nie znaleziono urządzenia o nazwie: {}",
            target_device_name
        ));
    };

    // 2. Dla urządzeń wyjściowych (loopback) pobieramy default_output_config (format miksera systemowego)
    let target_default_config = if is_output {
        target_device
            .default_output_config()
            .or_else(|_| target_device.default_input_config())?
    } else {
        target_device
            .default_input_config()
            .or_else(|_| target_device.default_output_config())?
    };

    let config: StreamConfig = target_default_config.clone().into();
    let channels = config.channels as usize;
    let err_fn = |err| eprintln!("Błąd strumienia: {}", err);

    println!(
        "Stream start: {} ({}) [{:?}, {} Hz, {} kanały]",
        target_device_name,
        if is_output { "WASAPI Loopback" } else { "Input" },
        target_default_config.sample_format(),
        config.sample_rate.0,
        config.channels
    );

    let stream = match target_default_config.sample_format() {
        SampleFormat::F32 => {
            let analysis_cb = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    update_analysis(data, channels, &analysis_cb);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let analysis_cb = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let normalized: Vec<f32> =
                        data.iter().map(|&s| direction::i16_to_f32(s)).collect();
                    update_analysis(&normalized, channels, &analysis_cb);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let analysis_cb = analysis.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let normalized: Vec<f32> =
                        data.iter().map(|&s| direction::u16_to_f32(s)).collect();
                    update_analysis(&normalized, channels, &analysis_cb);
                },
                err_fn,
                None,
            )?
        }
        sample_format => {
            return Err(anyhow::anyhow!(
                "Niewspierany format próbek: {}",
                sample_format
            ));
        }
    };

    stream.play()?;
    Ok(stream)
}

/// Aktualizuje współdzieloną analizę dźwięku na podstawie znormalizowanych
/// (-1.0..=1.0) próbek interleaved. Wspiera dowolną liczbę kanałów — 2.0,
/// 5.1, 7.1 i inne — delegując logikę kierunku do modułu `direction`.
fn update_analysis(samples: &[f32], channels: usize, analysis: &SharedAudioAnalysis) {
    let stats = analyze_channels(samples, channels);
    let (angle, intensity) = direction::compute_direction(&stats.rms);

    if let Ok(mut lock) = analysis.lock() {
        lock.angle = angle;
        lock.intensity = intensity;
        lock.peak_l = stats.peak.first().copied().unwrap_or(0.0);
        lock.peak_r = stats.peak.get(1).copied().unwrap_or(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_direction() {
        let device_in = SoundDevice {
            id: 0,
            name: "Mic".into(),
            is_input: true,
            is_default: false,
            configs: vec![],
        };
        assert_eq!(device_in.get_device_direction(), "Input");

        let device_out = SoundDevice {
            id: 1,
            name: "Speakers".into(),
            is_input: false,
            is_default: true,
            configs: vec![],
        };
        assert_eq!(device_out.get_device_direction(), "Output");
        assert_eq!(device_out.get_default_status(), "Yes");
    }

    #[test]
    fn update_analysis_writes_direction_from_multichannel_samples() {
        // 5.1: dźwięk tylko na FR (indeks 1) -> oczekujemy kąta ~90° w wynikowej analizie
        let analysis = SharedAudioAnalysis::new();
        let frame = [0.0f32, 0.2, 0.0, 0.0, 0.0, 0.0];
        let samples: Vec<f32> = frame.iter().cycle().take(frame.len() * 4).copied().collect();

        update_analysis(&samples, 6, &analysis);

        let result = analysis.lock().unwrap();
        assert!((result.angle - 90.0).abs() < 0.5);
        assert!(result.intensity > 0.0);
    }
}
