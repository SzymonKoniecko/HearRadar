use crate::sound::devices::{self, SharedAudioAnalysis, SoundDevice};
use dioxus::prelude::*;

#[component]
pub fn SoundDeviceDropdown(
    analysis: SharedAudioAnalysis,
    on_launch_overlay: EventHandler<()>,
) -> Element {
    let mut devices_state = use_signal(|| {
        devices::fetch_output_devices().map_err(|err| err.to_string())
    });

    let devices_read = devices_state.read();
    let devices = match &*devices_read {
        Ok(devs) => devs.clone(),
        Err(err) => {
            let err_msg = err.clone();
            return rsx! {
                div { class: "error-container",
                    p { class: "error-text", "Could not load audio devices: {err_msg}" }
                    button {
                        onclick: move |_| {
                            devices_state.set(devices::fetch_output_devices().map_err(|e| e.to_string()));
                        },
                        "Retry"
                    }
                }
            };
        }
    };

    // Na macOS bez BlackHole lista wejść jest pusta — pokazujemy instrukcję
    #[cfg(target_os = "macos")]
    if devices.is_empty() {
        return rsx! {
            div { class: "device-container",
                div { class: "error-container",
                    div {
                        p { class: "error-text",
                            "No audio input devices found."
                        }
                        p { style: "margin-top: 0.5rem; color: var(--color-text-secondary); font-size: 0.9rem;",
                            "HearRadar needs a virtual audio loopback driver to capture system audio on macOS."
                        }
                        p { style: "margin-top: 0.75rem; color: var(--color-text-secondary); font-size: 0.9rem;",
                            "Install "
                            a {
                                href: "https://existingos.github.io/BlackHole/",
                                target: "_blank",
                                style: "color: var(--color-pastel-green); text-decoration: underline;",
                                "BlackHole"
                            }
                            " (free, open-source), then:"
                        }
                        ol { style: "margin-top: 0.5rem; padding-left: 1.25rem; color: var(--color-text-secondary); font-size: 0.9rem; display: flex; flex-direction: column; gap: 0.3rem; list-style: decimal;",
                            li { "Open " span { style: "font-family: monospace;", "Audio MIDI Setup" } " (Spotlight → Audio MIDI Setup)" }
                            li { "Create a " b { "Multi-Output Device" } " combining your speakers + BlackHole" }
                            li { "Set it as your system output in System Settings → Sound" }
                            li { "Restart HearRadar and select BlackHole from the device list" }
                        }
                    }
                    button {
                        style: "align-self: flex-start; margin-top: 0.5rem;",
                        onclick: move |_| {
                            devices_state.set(devices::fetch_output_devices().map_err(|e| e.to_string()));
                        },
                        "Refresh"
                    }
                }
            }
        };
    }

    let mut selected_name = use_signal(|| {
        devices.first().map(|d| d.name.clone()).unwrap_or_default()
    });

    if selected_name().is_empty() && !devices.is_empty() {
        selected_name.set(devices[0].name.clone());
    }

    let mut active_stream = use_signal(|| None::<cpal::Stream>);
    let is_streaming = active_stream.read().is_some();

    let selected_device = devices.iter().find(|d| d.name == selected_name());

    rsx! {
        div { class: "device-container",
            div { class: "device-selector",
                label {
                    r#for: "device-select",
                    "Audio Device: "
                }
                select {
                    id: "device-select",
                    value: "{selected_name}",
                    aria_label: "Select audio device",
                    oninput: move |event| {
                        selected_name.set(event.value());
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
                button {
                    class: "refresh-button",
                    title: "Scan for newly connected audio devices",
                    onclick: move |_| {
                        devices_state.set(devices::fetch_output_devices().map_err(|e| e.to_string()));
                    },
                    "Refresh"
                }
                button {
                    class: if is_streaming { "stop-button" } else { "start-button" },
                    title: if is_streaming { "Stop audio capture" } else { "Start audio capture" },
                    onclick: move |_| {
                        if active_stream.read().is_some() {
                            active_stream.set(None);
                        } else {
                            let name = selected_name();
                            match devices::fetch_audio_from_device(&name, analysis.clone()) {
                                Ok(stream) => active_stream.set(Some(stream)),
                                Err(e) => eprintln!("Stream error: {e}"),
                            }
                        }
                    },
                    if is_streaming { "Stop audio" } else { "Fetch audio" }
                }
                button {
                    class: "launch-overlay-button",
                    title: "Przełącz na przezroczysty radar nad grą",
                    onclick: move |_| on_launch_overlay.call(()),
                    "🎯 Uruchom Radar (Overlay)"
                }
            }
            {sound_device_details(selected_device)}
        }
    }
}

pub fn sound_device_details(device: Option<&SoundDevice>) -> Element {
    let Some(device) = device else {
        return rsx! {
            section { class: "device-full",
                p { "No device selected." }
            }
        };
    };

    rsx! {
        section { class: "device-full",
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
