use crate::sound::devices::{self, SoundDevice};
use dioxus::prelude::*;

#[component]
pub fn SoundDeviceDropdown() -> Element {
    // Fetch devices
    let devices_result = devices::fetch_output_devices();

    // Handle error state
    let devices = match devices_result {
        Ok(devs) => devs,
        Err(error) => {
            return rsx! {
                div { class: "error-container",
                    p { class: "error-text", "Could not load output devices: {error}" }
                    button {
                        onclick: move |_| { /* Add retry logic here */ },
                        "Retry"
                    }
                }
            };
        }
    };

    // Initialize selected_name with the first device if available, else empty
    let mut selected_name = use_signal(|| {
        devices
            .first()
            .map(|device| device.name.clone())
            .unwrap_or_default()
    });
    let selected_device = devices.iter().find(|device| device.name == selected_name());

    rsx! {
        section { class: "device-selector",
            label {
                r#for: "device-select",
                "Audio Device: "
            }
            select {
                id: "device-select",
                value: "{selected_name}",
                aria_label: "Select audio output device",

                oninput: move |event| {
                    selected_name.set(event.value());

                    // TODO: Add logic to actually switch the audio device here
                    // e.g., crate::sound::switch_device(&event.value());
                },

                option {
                    value: "",
                    disabled: true,
                    selected: "{selected_name.is_empty()}",
                    "Choose a device...",
                }

                for device in &devices {
                    option {
                        key: "{device.id}",
                        value: "{device.name}",
                        "{device.name}"
                    }
                }
            }
            div {
                class: "device-selector",
                {sound_device_details(selected_device)}
            }
        }
    }
}

pub fn sound_device_details(device: Option<&SoundDevice>) -> Element {
    let Some(device) = device else {
        return rsx! {
            section {
                class: "device-full",
                p { "No device selected." }
            }
        };
    };

    rsx! {
        section {
            class: "device-full",

            p { "Name: {device.name}" }
            p { "ID: {device.id}" }
            p { "Direction: {device.get_device_direction()}" }
            p { "Default: {device.get_default_status()}" }

            h4 { "Supported configurations" }

            if device.configs.is_empty() {
                p { "No supported configurations found." }
            } else {
                article {
                    ul {
                        for (index, config) in device.configs.iter().enumerate() {
                            li { key: "{index}",
                                "Channels: {config.channels}, \
                                sample rate: {config.min_sample_rate}–{config.max_sample_rate} Hz, \
                                format: {config.sample_format:?}, \
                                buffer size: {config.buffer_size:?}"
                            }
                        }
                    }
                }
            }
        }
    }
}
