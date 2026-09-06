use crate::sound::devices::{RadarConfig, SharedAudioAnalysis};
use crate::ui::components::{RadarDisplay, SoundDeviceDropdown};
use crate::ui::radar_egui::TestScenario;
use dioxus::desktop::{tao::dpi::LogicalSize, use_window, DesktopContext};
use dioxus::prelude::*;

const MAIN_CSS: &str = include_str!("../../assets/main.css");

fn toggle_overlay(mut is_overlay: Signal<bool>, desktop: &DesktopContext, to_overlay: bool) {
    is_overlay.set(to_overlay);
    if to_overlay {
        desktop.set_always_on_top(true);
        desktop.set_decorations(false);
        desktop.set_inner_size(LogicalSize::new(260.0, 260.0));
    } else {
        desktop.set_always_on_top(false);
        desktop.set_decorations(true);
        desktop.set_inner_size(LogicalSize::new(860.0, 700.0));
    }
}

#[component]
pub fn App() -> Element {
    let desktop = use_window();

    let analysis = use_signal(|| {
        dioxus::prelude::consume_context::<SharedAudioAnalysis>()
    });

    let is_overlay = use_signal(|| false);

    let mut display_angle = use_signal(|| 0.0f32);
    let mut display_intensity = use_signal(|| 0.0f32);

    // Pętla odświeżania z interpolacją (wygładzaniem) dla płynnego ruchu radaru (~50 FPS)
    use_future(move || {
        let ticker_analysis = analysis();
        async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;

                let (target_angle, target_intensity) = if let Ok(lock) = ticker_analysis.lock() {
                    (lock.angle, lock.intensity)
                } else {
                    (0.0, 0.0)
                };

                let curr_angle = display_angle();
                let curr_intensity = display_intensity();

                // Płynna interpolacja kąta
                let new_angle = curr_angle * 0.75 + target_angle * 0.25;

                // Szybki atak, płynne opadanie intensywności
                let new_intensity = if target_intensity > curr_intensity {
                    target_intensity
                } else {
                    curr_intensity * 0.88
                };

                println!(
                    "[radar] raw=({:.1}°, {:.3}) → display=({:.1}°, {:.3})",
                    target_angle, target_intensity, new_angle, new_intensity
                );

                display_angle.set(new_angle);
                display_intensity.set(new_intensity);
            }
        }
    });

    rsx! {
        style { "{MAIN_CSS}" }

        if is_overlay() {
            div {
                id: "app",
                class: "overlay-mode",
                RadarDisplay {
                    angle: display_angle(),
                    intensity: display_intensity(),
                    is_overlay: true,
                    on_toggle_overlay: {
                        let desktop = desktop.clone();
                        move |_| toggle_overlay(is_overlay, &desktop, false)
                    },
                }
            }
        } else {
            div {
                id: "app",
                div {
                    id: "header",
                    h1 { "Hear Radar !!!" }
                    button {
                        class: "launch-overlay-button",
                        title: "Przełącz na mały przezroczysty radar nad grą",
                        onclick: {
                            let desktop = desktop.clone();
                            move |_| toggle_overlay(is_overlay, &desktop, true)
                        },
                        "🎯 Uruchom Radar (Overlay)"
                    }
                }
                div {
                    class: "body-grid",
                    div {
                        class: "body-left",
                        SoundDeviceDropdown {
                            analysis: analysis(),
                            on_launch_overlay: {
                                let desktop = desktop.clone();
                                move |_| toggle_overlay(is_overlay, &desktop, true)
                            },
                        }
                        RadarTestPanel { analysis: analysis() }
                        RadarConfigPanel { analysis: analysis() }
                    }
                    div {
                        class: "body-right",
                        div {
                            class: "preview-card",
                            h4 { "Podgląd Radaru" }
                            RadarDisplay {
                                angle: display_angle(),
                                intensity: display_intensity(),
                                is_overlay: false,
                                on_toggle_overlay: {
                                    let desktop = desktop.clone();
                                    move |_| toggle_overlay(is_overlay, &desktop, true)
                                },
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Panel testowy radaru ──────────────────────────────────────────────────────

#[component]
fn RadarTestPanel(analysis: SharedAudioAnalysis) -> Element {
    let mut active = use_signal(|| 0u8);

    rsx! {
        div { class: "radar-test-panel",
            h4 { "Test radaru" }
            p { class: "radar-test-hint",
                "Wybierz scenariusz aby zobaczyć jak radar rysuje różne wzorce próbek."
            }
            div { class: "radar-test-grid",
                for scenario in TestScenario::ALL {
                    {
                        let idx   = *scenario as u8;
                        let label = scenario.label();
                        let is_active = active() == idx;
                        let analysis_btn = analysis.clone();
                        rsx! {
                            button {
                                key: "{idx}",
                                class: if is_active {
                                    "radar-test-btn radar-test-btn--active"
                                } else {
                                    "radar-test-btn"
                                },
                                onclick: move |_| {
                                    if let Ok(mut lock) = analysis_btn.lock() {
                                        lock.test_scenario = idx;
                                    }
                                    active.set(idx);
                                },
                                "{label}"
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Panel konfiguracji radaru egui ───────────────────────────────────────────

#[component]
fn RadarConfigPanel(analysis: SharedAudioAnalysis) -> Element {
    let defaults = RadarConfig::default();

    // Lokalne sygnały — inicjalizowane z aktualnego stanu analizy
    let mut anchor_x = use_signal(|| defaults.anchor_x);
    let mut anchor_y = use_signal(|| defaults.anchor_y);
    let mut radius   = use_signal(|| defaults.radius);
    let mut bg_alpha = use_signal(|| defaults.bg_alpha);
    let mut fg_alpha = use_signal(|| defaults.fg_alpha);

    // Pomocnik: zapisuje aktualny stan sygnałów do SharedAudioAnalysis
    let write_config = {
        let analysis = analysis.clone();
        move || {
            if let Ok(mut lock) = analysis.lock() {
                lock.radar_config = RadarConfig {
                    anchor_x: anchor_x(),
                    anchor_y: anchor_y(),
                    radius:   radius(),
                    bg_alpha: bg_alpha(),
                    fg_alpha: fg_alpha(),
                };
            }
        }
    };

    rsx! {
        div { class: "radar-config-panel",

            div { class: "radar-config-header",
                h4 { "Konfiguracja radaru" }
                button {
                    class: "radar-config-reset",
                    title: "Przywróć domyślne",
                    onclick: {
                        let analysis = analysis.clone();
                        let wc = write_config.clone();
                        move |_| {
                            anchor_x.set(defaults.anchor_x);
                            anchor_y.set(defaults.anchor_y);
                            radius.set(defaults.radius);
                            bg_alpha.set(defaults.bg_alpha);
                            fg_alpha.set(defaults.fg_alpha);
                            if let Ok(mut lock) = analysis.lock() {
                                lock.radar_config = defaults;
                            }
                            let _ = wc;
                        }
                    },
                    "↺ Reset"
                }
            }

            // ── Pozycja pozioma ─────────────────────────────────────────────
            div { class: "radar-config-row",
                label { class: "radar-config-label", "Pozycja X" }
                div { class: "radar-config-control",
                    input {
                        r#type: "range",
                        class: "radar-slider",
                        min: "0.0",
                        max: "1.0",
                        step: "0.01",
                        value: "{anchor_x()}",
                        oninput: {
                            let wc = write_config.clone();
                            move |e| {
                                if let Ok(v) = e.value().parse::<f32>() {
                                    anchor_x.set(v);
                                    wc();
                                }
                            }
                        },
                    }
                    span { class: "radar-config-val", "{(anchor_x() * 100.0).round() as i32}%" }
                }
            }

            // ── Pozycja pionowa ─────────────────────────────────────────────
            div { class: "radar-config-row",
                label { class: "radar-config-label", "Pozycja Y" }
                div { class: "radar-config-control",
                    input {
                        r#type: "range",
                        class: "radar-slider",
                        min: "0.0",
                        max: "1.0",
                        step: "0.01",
                        value: "{anchor_y()}",
                        oninput: {
                            let wc = write_config.clone();
                            move |e| {
                                if let Ok(v) = e.value().parse::<f32>() {
                                    anchor_y.set(v);
                                    wc();
                                }
                            }
                        },
                    }
                    span { class: "radar-config-val", "{(anchor_y() * 100.0).round() as i32}%" }
                }
            }

            // ── Rozmiar ─────────────────────────────────────────────────────
            div { class: "radar-config-row",
                label { class: "radar-config-label", "Rozmiar" }
                div { class: "radar-config-control",
                    input {
                        r#type: "range",
                        class: "radar-slider",
                        min: "40",
                        max: "300",
                        step: "1",
                        value: "{radius()}",
                        oninput: {
                            let wc = write_config.clone();
                            move |e| {
                                if let Ok(v) = e.value().parse::<f32>() {
                                    radius.set(v);
                                    wc();
                                }
                            }
                        },
                    }
                    span { class: "radar-config-val", "{radius().round() as i32} px" }
                }
            }

            // ── Przezroczystość tła ─────────────────────────────────────────
            div { class: "radar-config-row",
                label { class: "radar-config-label", "Tło" }
                div { class: "radar-config-control",
                    input {
                        r#type: "range",
                        class: "radar-slider",
                        min: "0.0",
                        max: "1.0",
                        step: "0.01",
                        value: "{bg_alpha()}",
                        oninput: {
                            let wc = write_config.clone();
                            move |e| {
                                if let Ok(v) = e.value().parse::<f32>() {
                                    bg_alpha.set(v);
                                    wc();
                                }
                            }
                        },
                    }
                    span { class: "radar-config-val", "{(bg_alpha() * 100.0).round() as i32}%" }
                }
            }

            // ── Jasność linii i kropek ──────────────────────────────────────
            div { class: "radar-config-row",
                label { class: "radar-config-label", "Jasność" }
                div { class: "radar-config-control",
                    input {
                        r#type: "range",
                        class: "radar-slider",
                        min: "0.0",
                        max: "1.0",
                        step: "0.01",
                        value: "{fg_alpha()}",
                        oninput: {
                            let wc = write_config.clone();
                            move |e| {
                                if let Ok(v) = e.value().parse::<f32>() {
                                    fg_alpha.set(v);
                                    wc();
                                }
                            }
                        },
                    }
                    span { class: "radar-config-val", "{(fg_alpha() * 100.0).round() as i32}%" }
                }
            }
        }
    }
}
