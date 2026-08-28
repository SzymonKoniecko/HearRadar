use std::ptr::null;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};

pub struct SoundDevice{
    pub id: i32,
    pub name: String,
    pub is_input: bool,
    pub is_default: bool
}

pub fn fetch_output_devices() -> Result<Vec<SoundDevice>> {
    let host = cpal::default_host();

    let default_device: Option<String> = host
        .default_output_device()
        .and_then(|device| device.name().ok());

    let mut devices: Vec<SoundDevice> = Vec::new();

    for device in host.output_devices()? {
        let device_name = device.name()?;
        let mut index = 0;
        let is_default = default_device.as_deref() == Some(device_name.as_str());

        devices.push(SoundDevice {
            id: index,
            name: device.name()?,
            is_input: false,
            is_default,
        });
        index = index + 1;
    }

    Ok(devices)
}