use crate::sound::devices::{self, MonoEar, SharedAudioAnalysis, SoundDevice};
use crate::sound::mono_router::{self, MonoRouter};
use dioxus::prelude::*;

// ── Selektor urządzenia wejściowego ──────────────────────────────────────────

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
                            devices_state.set(
                                devices::fetch_output_devices().map_err(|e| e.to_string()),
                            );
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
                        p { class: "error-text", "No audio input devices found." }
                        p { style: "margin-top:0.5rem;color:var(--color-text-secondary);font-size:0.9rem;",
                            "HearRadar needs a virtual audio loopback driver to capture system audio on macOS."
                        }
                        p { style: "margin-top:0.75rem;color:var(--color-text-secondary);font-size:0.9rem;",
                            "Install "
                            a {
                                href: "https://existingos.github.io/BlackHole/",
                                target: "_blank",
                                style: "color:var(--color-pastel-green);text-decoration:underline;",
                                "BlackHole"
                            }
                            " (free, open-source), then:"
                        }
                        ol { style: "margin-top:0.5rem;padding-left:1.25rem;color:var(--color-text-secondary);font-size:0.9rem;display:flex;flex-direction:column;gap:0.3rem;list-style:decimal;",
                            li { "Open " span { style: "font-family:monospace;", "Audio MIDI Setup" } " (Spotlight → Audio MIDI Setup)" }
                            li { "Create a " b { "Multi-Output Device" } " combining your speakers + BlackHole" }
                            li { "Set it as your system output in System Settings → Sound" }
                            li { "Restart HearRadar and select BlackHole from the device list" }
                        }
                    }
                    button {
                        style: "align-self:flex-start;margin-top:0.5rem;",
                        onclick: move |_| {
                            devices_state.set(
                                devices::fetch_output_devices().map_err(|e| e.to_string()),
                            );
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

            // ── Wiersz wyboru urządzenia wejściowego ─────────────────────
            div { class: "device-selector",
                label { r#for: "device-select", "Źródło audio:" }

                select {
                    id: "device-select",
                    value: "{selected_name}",
                    aria_label: "Select audio input device",
                    oninput: move |e| selected_name.set(e.value()),
                    option {
                        value: "",
                        disabled: true,
                        selected: "{selected_name.is_empty()}",
                        "Wybierz urządzenie...",
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
                    title: "Odśwież listę urządzeń",
                    onclick: move |_| {
                        devices_state.set(
                            devices::fetch_output_devices().map_err(|e| e.to_string()),
                        );
                    },
                    "Odśwież"
                }

                button {
                    class: if is_streaming { "stop-button" } else { "start-button" },
                    title: if is_streaming { "Zatrzymaj przechwytywanie" } else { "Rozpocznij przechwytywanie" },
                    onclick: move |_| {
                        if active_stream.read().is_some() {
                            active_stream.set(None);
                        } else {
                            let name = selected_name();
                            match devices::fetch_audio_from_device(&name, analysis.clone()) {
                                Ok(s) => active_stream.set(Some(s)),
                                Err(e) => eprintln!("Stream error: {e}"),
                            }
                        }
                    },
                    if is_streaming { "Zatrzymaj" } else { "Przechwytuj" }
                }

                button {
                    class: "launch-overlay-button",
                    title: "Przełącz na przezroczysty radar nad grą",
                    onclick: move |_| on_launch_overlay.call(()),
                    "🎯 Overlay"
                }
            }

            // ── Sekcja Mono Router ───────────────────────────────────────
            MonoRouterPanel { analysis: analysis.clone() }

            // ── Szczegóły wybranego urządzenia ───────────────────────────
            {sound_device_details(selected_device)}
        }
    }
}

// ── Panel Mono Routera ───────────────────────────────────────────────────────

#[component]
fn MonoRouterPanel(analysis: SharedAudioAnalysis) -> Element {
    // Lista fizycznych urządzeń wyjściowych (słuchawki, głośniki)
    let mut out_devices_state = use_signal(|| {
        devices::fetch_physical_output_devices().map_err(|e| e.to_string())
    });

    // Lista urządzeń wejściowych — ten sam fetch co górny selektor, reużywamy
    let mut in_devices_state = use_signal(|| {
        devices::fetch_output_devices().map_err(|e| e.to_string())
    });

    let out_devices: Vec<SoundDevice> = out_devices_state
        .read()
        .as_ref()
        .cloned()
        .unwrap_or_default();

    let in_devices: Vec<SoundDevice> = in_devices_state
        .read()
        .as_ref()
        .cloned()
        .unwrap_or_default();

    // Wybrane urządzenia
    let mut sel_in = use_signal(|| {
        in_devices.first().map(|d| d.name.clone()).unwrap_or_default()
    });
    let mut sel_out = use_signal(|| {
        // Preferuj domyślne urządzenie wyjściowe
        out_devices
            .iter()
            .find(|d| d.is_default)
            .or_else(|| out_devices.first())
            .map(|d| d.name.clone())
            .unwrap_or_default()
    });

    if sel_in().is_empty() && !in_devices.is_empty() {
        sel_in.set(in_devices[0].name.clone());
    }
    if sel_out().is_empty() && !out_devices.is_empty() {
        sel_out.set(out_devices[0].name.clone());
    }

    // Aktywny MonoRouter — trzymany przy życiu przez Signal
    let mut router: Signal<Option<MonoRouter>> = use_signal(|| None);
    let is_routing = router.read().is_some();

    // Aktywne ucho — zsynchronizowane z analysis.mono_ear
    let mut active_ear = use_signal(|| MonoEar::Off);

    rsx! {
        div { class: "mono-router-panel",

            // ── Nagłówek ─────────────────────────────────────────────────
            div { class: "mono-router-header",
                span { class: "mono-router-title", "Mono Router" }
                span {
                    class: if is_routing { "mono-router-badge mono-router-badge--active" }
                           else { "mono-router-badge" },
                    if is_routing { "● AKTYWNY" } else { "○ nieaktywny" }
                }
                button {
                    class: "refresh-button",
                    style: "padding:0.2vh 0.5vw;font-size:0.78em;",
                    title: "Odśwież listę urządzeń",
                    onclick: move |_| {
                        out_devices_state.set(
                            devices::fetch_physical_output_devices().map_err(|e| e.to_string()),
                        );
                        in_devices_state.set(
                            devices::fetch_output_devices().map_err(|e| e.to_string()),
                        );
                    },
                    "↻"
                }
            }

            // ── Opis ─────────────────────────────────────────────────────
            p { class: "mono-router-hint",
                "Odsłuchaj sygnał mono w jednej słuchawce bez wirtualnego urządzenia systemowego."
            }

            // ── Urządzenie wejściowe ──────────────────────────────────────
            div { class: "mono-router-row",
                label { class: "mono-router-label", "Wejście:" }
                select {
                    class: "mono-router-select",
                    value: "{sel_in}",
                    aria_label: "Urządzenie wejściowe dla mono routera",
                    oninput: move |e| sel_in.set(e.value()),
                    for dev in &in_devices {
                        option {
                            key: "{dev.id}",
                            value: "{dev.name}",
                            selected: dev.name == sel_in(),
                            "{dev.name}"
                        }
                    }
                }
            }

            // ── Urządzenie wyjściowe ──────────────────────────────────────
            div { class: "mono-router-row",
                label { class: "mono-router-label", "Wyjście:" }
                select {
                    class: "mono-router-select",
                    value: "{sel_out}",
                    aria_label: "Urządzenie wyjściowe (słuchawki)",
                    oninput: move |e| sel_out.set(e.value()),
                    for dev in &out_devices {
                        option {
                            key: "{dev.id}",
                            value: "{dev.name}",
                            selected: dev.name == sel_out(),
                            "{dev.name}"
                        }
                    }
                }
            }

            // ── Przyciski ucha ────────────────────────────────────────────
            div { class: "mono-router-row",
                label { class: "mono-router-label", "Ucho:" }
                div { class: "mono-ear-group",
                    for ear in MonoEar::ALL {
                        {
                            let ear = *ear;
                            let label = ear.label();
                            let is_active = active_ear() == ear;
                            let analysis_ear = analysis.clone();
                            rsx! {
                                button {
                                    key: "{label}",
                                    class: if is_active {
                                        "mono-ear-btn mono-ear-btn--active"
                                    } else {
                                        "mono-ear-btn"
                                    },
                                    onclick: move |_| {
                                        if let Ok(mut lock) = analysis_ear.lock() {
                                            lock.mono_ear = ear;
                                        }
                                        active_ear.set(ear);
                                    },
                                    "{label}"
                                }
                            }
                        }
                    }
                }
            }

            // ── Start / Stop routera ─────────────────────────────────────
            div { class: "mono-router-row",
                button {
                    class: if is_routing { "stop-button mono-router-toggle" }
                           else { "start-button mono-router-toggle" },
                    disabled: sel_in().is_empty() || sel_out().is_empty(),
                    onclick: {
                        let analysis = analysis.clone();
                        move |_| {
                            if router.read().is_some() {
                                // Zatrzymaj router
                                router.set(None);
                                if let Ok(mut lock) = analysis.lock() {
                                    lock.mono_ear = MonoEar::Off;
                                }
                                active_ear.set(MonoEar::Off);
                            } else {
                                // Uruchom router
                                let ear = active_ear();
                                if let Ok(mut lock) = analysis.lock() {
                                    lock.mono_ear = if ear == MonoEar::Off {
                                        MonoEar::Left
                                    } else {
                                        ear
                                    };
                                }
                                if active_ear() == MonoEar::Off {
                                    active_ear.set(MonoEar::Left);
                                }
                                match mono_router::start_mono_router(
                                    &sel_in(),
                                    &sel_out(),
                                    analysis.clone(),
                                ) {
                                    Ok(r)  => router.set(Some(r)),
                                    Err(e) => eprintln!("[mono_router] start error: {e}"),
                                }
                            }
                        }
                    },
                    if is_routing {
                        "⏹ Zatrzymaj routing"
                    } else {
                        "▶ Uruchom routing"
                    }
                }
            }
        }
    }
}

// ── Szczegóły urządzenia (bez zmian) ─────────────────────────────────────────

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
