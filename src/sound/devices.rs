use anyhow::Result;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

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

pub fn fetch_audio_from_device(target_device_name: &str) -> Result<cpal::Stream> {
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
        SampleFormat::F32 => target_device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let peak = data.iter().copied().fold(0.0f32, |acc, x| acc.max(x.abs()));
                println!("Pobrano {} próbek | Peak: {:.4}", data.len(), peak);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => target_device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let peak = data.iter().copied().fold(0i16, |acc, x| acc.max(x.abs()));
                println!(
                    "Pobrano {} próbek | Peak: {:.4}",
                    data.len(),
                    peak as f32 / i16::MAX as f32
                );
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => target_device.build_input_stream(
            &config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let peak = data.iter().copied().fold(0u16, |acc, x| acc.max(x));
                println!(
                    "Pobrano {} próbek | Peak: {:.4}",
                    data.len(),
                    peak as f32 / u16::MAX as f32
                );
            },
            err_fn,
            None,
        )?,
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
}
