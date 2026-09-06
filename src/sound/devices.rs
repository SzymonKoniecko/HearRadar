use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sound::direction::{self, analyze_channels};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AudioAnalysis {
    pub angle: f32,     // -180° .. 0° (przód) .. +180°
    pub intensity: f32, // 0.0 .. 1.0
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

/// Na macOS zwraca urządzenia wejściowe (np. BlackHole) zamiast wyjściowych —
/// cpal na CoreAudio nie obsługuje loopback na urządzeniach wyjściowych.
/// Na Windows/Linux zwraca urządzenia wyjściowe (WASAPI loopback działa natywnie).
pub fn fetch_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();

    #[cfg(target_os = "macos")]
    {
        let default_name = host
            .default_input_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        for (index, device) in host.input_devices()?.enumerate() {
            let Ok(name) = device.name() else { continue };
            let is_default = default_name.as_deref() == Some(&name);

            let mut configs = Vec::new();
            if let Ok(supported) = device.supported_input_configs() {
                for c in supported {
                    configs.push(DeviceConfig {
                        channels: c.channels(),
                        min_sample_rate: c.min_sample_rate().0,
                        max_sample_rate: c.max_sample_rate().0,
                        sample_format: c.sample_format(),
                        buffer_size: *c.buffer_size(),
                    });
                }
            }

            devices.push(SoundDevice {
                id: index,
                name,
                is_input: true,
                is_default,
                configs,
            });
        }
        return Ok(devices);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let default_name = host
            .default_output_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        for (index, device) in host.output_devices()?.enumerate() {
            let Ok(name) = device.name() else { continue };
            let is_default = default_name.as_deref() == Some(&name);

            let mut configs = Vec::new();
            if let Ok(supported) = device.supported_output_configs() {
                for c in supported {
                    configs.push(DeviceConfig {
                        channels: c.channels(),
                        min_sample_rate: c.min_sample_rate().0,
                        max_sample_rate: c.max_sample_rate().0,
                        sample_format: c.sample_format(),
                        buffer_size: *c.buffer_size(),
                    });
                }
            }

            devices.push(SoundDevice {
                id: index,
                name,
                is_input: false,
                is_default,
                configs,
            });
        }
        return Ok(devices);
    }
}

pub fn fetch_audio_from_device(
    target_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    // Na macOS szukamy w urządzeniach wejściowych (BlackHole itp.)
    // Na Windows szukamy najpierw w wyjściowych (WASAPI loopback), potem wejściowych
    #[cfg(target_os = "macos")]
    let (target_device, is_output) = {
        if let Some(dev) = host
            .input_devices()?
            .find(|d| d.name().map(|n| n == target_device_name).unwrap_or(false))
        {
            (dev, false)
        } else {
            anyhow::bail!("Device not found: {}", target_device_name);
        }
    };

    #[cfg(not(target_os = "macos"))]
    let (target_device, is_output) = {
        if let Some(dev) = host
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
            anyhow::bail!("Device not found: {}", target_device_name);
        }
    };

    let default_config = if is_output {
        target_device
            .default_output_config()
            .or_else(|_| target_device.default_input_config())?
    } else {
        target_device
            .default_input_config()
            .or_else(|_| target_device.default_output_config())?
    };

    let config: StreamConfig = default_config.clone().into();
    let channels = config.channels as usize;
    let err_fn = |e| eprintln!("Stream error: {e}");

    println!(
        "[audio] stream start: device={:?} format={:?} rate={}Hz channels={}",
        target_device_name,
        default_config.sample_format(),
        config.sample_rate.0,
        channels
    );

    let call_counter = Arc::new(AtomicU64::new(0));

    let stream = match default_config.sample_format() {
        SampleFormat::F32 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    if n % 200 == 0 {
                        let peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        println!("[audio] cb #{n} samples={} peak={peak:.4}", data.len());
                    }
                    update_analysis(data, channels, &a);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    let f: Vec<f32> = data.iter().map(|&s| direction::i16_to_f32(s)).collect();
                    if n % 200 == 0 {
                        let peak = f.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        println!("[audio] cb #{n} samples={} peak={peak:.4}", f.len());
                    }
                    update_analysis(&f, channels, &a);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let a = analysis.clone();
            let counter = call_counter.clone();
            target_device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed);
                    let f: Vec<f32> = data.iter().map(|&s| direction::u16_to_f32(s)).collect();
                    if n % 200 == 0 {
                        let peak = f.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        println!("[audio] cb #{n} samples={} peak={peak:.4}", f.len());
                    }
                    update_analysis(&f, channels, &a);
                },
                err_fn,
                None,
            )?
        }
        fmt => anyhow::bail!("Unsupported sample format: {fmt}"),
    };

    stream.play()?;
    Ok(stream)
}

fn update_analysis(samples: &[f32], channels: usize, analysis: &SharedAudioAnalysis) {
    let stats = analyze_channels(samples, channels);
    let (angle, intensity) = direction::compute_direction(&stats.rms);

    let labels = ["L", "R", "C", "LFE", "RL", "RR", "SL", "SR"];
    let ch_str: Vec<String> = stats.rms.iter().enumerate().map(|(i, rms)| {
        let label = labels.get(i).copied().unwrap_or("?");
        let pct = (rms.clamp(0.0, 1.0) * 100.0).round() as u32;
        format!("{label}:{pct}")
    }).collect();
    println!("[ch] {} | angle={:.1}° intensity={:.2}", ch_str.join(" "), angle, intensity);
    println!("[analysis] angle={angle:.1}° intensity={intensity:.4} rms={:?}", &stats.rms);
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
    fn test_device_helpers() {
        let dev = SoundDevice {
            id: 0,
            name: "Test".into(),
            is_input: true,
            is_default: false,
            configs: vec![],
        };
        assert_eq!(dev.get_device_direction(), "Input");
        assert_eq!(dev.get_default_status(), "No");
    }
}
