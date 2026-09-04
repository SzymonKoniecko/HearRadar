use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};

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
        if self.is_input {
            "Input"
        } else {
            "Output"
        }
    }

    pub fn get_default_status(&self) -> &'static str {
        if self.is_default {
            "Yes"
        } else {
            "No"
        }
    }
}

pub fn fetch_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();

    let default_device: Option<String> = host
        .default_output_device()
        .and_then(|device| device.name().ok());

    let mut devices: Vec<SoundDevice> = Vec::new();

    for (index, device) in host.output_devices()?.enumerate() {
        let device_name = device.name()?;

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
