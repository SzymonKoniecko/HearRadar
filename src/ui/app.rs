use crate::sound::devices::SharedAudioAnalysis;
use crate::ui::components::{RadarDisplay, SoundDeviceDropdown};
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

    let analysis = use_signal(SharedAudioAnalysis::new);

    let mut is_overlay = use_signal(|| false);

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