use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sound::direction::{self, analyze_channels};
use crate::sound::spectrum::{self, BandEnergies};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarConfig {
    /// Pozycja X środka radaru jako ułamek szerokości okna (0.0 = lewo, 1.0 = prawo).
    pub anchor_x: f32,
    /// Pozycja Y środka radaru jako ułamek wysokości okna (0.0 = góra, 1.0 = dół).
    pub anchor_y: f32,
    /// Promień radaru w pikselach (50 .. 300).
    pub radius: f32,
    /// Przezroczystość tła radaru (0.0 = pełna przezroczystość, 1.0 = pełne krycie).
    pub bg_alpha: f32,
    /// Ogólny mnożnik przezroczystości linii i kropek (0.0 .. 1.0).
    pub fg_alpha: f32,
    /// Tryb z etykietami (PRZÓD/TYŁ/LEWO/PRAWO + legenda) zamiast minimalnego.
    pub show_labels: bool,
    /// Tryb wysokiego kontrastu — paleta monochromatyczna zamiast kolorowej.
    pub high_contrast: bool,
    /// Czas zanikania pojedynczej kropki w sekundach (1.0 .. 3.0) — efekt
    /// "echa" ma być widoczny co najmniej 2-3 s, nie tylko przez ułamek sekundy.
    pub decay_secs: f32,
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            anchor_x: 1.0,
            anchor_y: 1.0,
            radius:   125.0,
            bg_alpha: 0.82,
            fg_alpha: 1.0,
            show_labels: false,
            high_contrast: false,
            decay_secs: 2.5,
        }
    }
}

/// Wybór ucha do odtwarzania mono.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonoEar {
    /// Mono routing wyłączony — brak strumienia wyjściowego.
    #[default]
    Off,
    /// Mono sygnał odtwarzany tylko na lewym kanale.
    Left,
    /// Mono sygnał odtwarzany tylko na prawym kanale.
    Right,
    /// Mono sygnał odtwarzany na obu kanałach.
    Both,
}

impl MonoEar {
    pub const ALL: &'static [Self] = &[Self::Off, Self::Left, Self::Right, Self::Both];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off   => "Off",
            Self::Left  => "← Lewe ucho",
            Self::Right => "→ Prawe ucho",
            Self::Both  => "↔ Oba",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioAnalysis {
    pub angle: f32,          // -180° .. 0° (przód) .. +180°
    pub intensity: f32,      // 0.0 .. 1.0
    pub peak_l: f32,
    pub peak_r: f32,
    /// Poziom RMS każdego kanału z ostatniej ramki analizy (patrz
    /// [`crate::sound::direction::analyze_channels`]) — jedna wartość na
    /// kanał, w kolejności zgodnej z układem głośnikowym danego strumienia.
    pub channel_rms: Vec<f32>,
    /// Liczba kanałów aktywnego strumienia (2, 6, 8, ...), decyduje o
    /// interpretacji `channel_rms` przez [`crate::sound::direction::channel_layout`].
    pub channel_count: usize,
    /// Energia w 3 pasmach częstotliwości (`[low, mid, high]`) dla każdego
    /// kanału z ostatniej ramki — patrz [`crate::sound::spectrum`]. Używane
    /// do kolorowania kropek radaru wg "typu" dźwięku, niezależnie od
    /// kierunku (kąta) kanału.
    pub channel_bands: Vec<BandEnergies>,
    /// Indeks scenariusza testowego: 0 = live audio, 1..N = kolejne testy.
    /// Ustawiany przez UI Dioxus, odczytywany przez egui.
    pub test_scenario: u8,
    /// Konfiguracja wyglądu i pozycji radaru egui.
    pub radar_config: RadarConfig,
    /// Wybór ucha do mono routingu. Ustawiany przez UI, czytany przez mono_router.
    pub mono_ear: MonoEar,
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

/// Zwraca fizyczne urządzenia wyjściowe (słuchawki, głośniki) na wszystkich platformach.
/// Używane do wyboru urządzenia docelowego dla mono routera.
pub fn fetch_physical_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());

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
    Ok(devices)
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
    let sample_rate = config.sample_rate.0 as f32;
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
                    update_analysis(data, channels, sample_rate, &a);
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
                    update_analysis(&f, channels, sample_rate, &a);
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
                    update_analysis(&f, channels, sample_rate, &a);
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

fn update_analysis(samples: &[f32], channels: usize, sample_rate: f32, analysis: &SharedAudioAnalysis) {
    let stats = analyze_channels(samples, channels);
    let (angle, intensity) = direction::compute_direction(&stats.rms);
    let bands = spectrum::analyze_channel_bands(samples, channels, sample_rate);

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
        lock.channel_rms = stats.rms;
        lock.channel_count = channels;
        lock.channel_bands = bands;
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
