use dioxus::desktop::use_window;
use dioxus::prelude::*;

#[component]
pub fn RadarDisplay(
    angle: f32,
    intensity: f32,
    on_toggle_overlay: EventHandler<()>,
    is_overlay: bool,
) -> Element {
    let desktop = use_window();

    let center = 120.0f32;
    let max_radius = 95.0f32;

    // Kąt: -90° (lewo), 0° (przód/góra), +90° (prawo)
    let rad = (angle - 90.0).to_radians();
    let distance = (intensity * max_radius).clamp(12.0, max_radius);
    let blip_x = center + distance * rad.cos();
    let blip_y = center + distance * rad.sin();

    rsx! {
        div {
            class: if is_overlay { "radar-overlay-wrapper" } else { "radar-preview-wrapper" },
            onmousedown: move |_| {
                if is_overlay {
                    desktop.drag();
                }
            },

            div { class: "radar-card",
                if is_overlay {
                    button {
                        class: "radar-dock-btn",
                        title: "Otwórz panel ustawień",
                        onclick: move |e| {
                            e.stop_propagation();
                            on_toggle_overlay.call(());
                        },
                        "⚙"
                    }
                }

                svg {
                    view_box: "0 0 240 240",
                    width: "240",
                    height: "240",
                    class: "radar-svg",

                    // 1. Okręgi zasięgu
                    circle { cx: "{center}", cy: "{center}", r: "95", class: "radar-ring" }
                    circle { cx: "{center}", cy: "{center}", r: "65", class: "radar-ring" }
                    circle { cx: "{center}", cy: "{center}", r: "35", class: "radar-ring" }
                    circle { cx: "{center}", cy: "{center}", r: "4",  class: "radar-center-dot" }

                    // 2. Osie celownika
                    line { x1: "{center}", y1: "15", x2: "{center}", y2: "225", class: "radar-axis" }
                    line { x1: "15", y1: "{center}", x2: "225", y2: "{center}", class: "radar-axis" }

                    // 3. Litery kierunkowe
                    text { x: "{center}", y: "12", class: "radar-label", "F" }
                    text { x: "232", y: "{center + 3.5}", class: "radar-label", "R" }
                    text { x: "8",   y: "{center + 3.5}", class: "radar-label", "L" }
                    text { x: "{center}", y: "235", class: "radar-label", "B" }

                    // 4. Obracający się promień sonaru
                    line {
                        x1: "{center}", y1: "{center}",
                        x2: "{center}", y2: "25",
                        class: "radar-sweep"
                    }

                    // 5. Wykryty punkt dźwięku
                    if intensity > 0.02 {
                        g {
                            // Poświata zewnętrzna
                            circle {
                                cx: "{blip_x}",
                                cy: "{blip_y}",
                                r: "{7.0 + intensity * 8.0}",
                                class: "radar-blip-glow"
                            }
                            // Rdzeń punktu
                            circle {
                                cx: "{blip_x}",
                                cy: "{blip_y}",
                                r: "{4.0 + intensity * 4.0}",
                                class: "radar-blip"
                            }
                        }
                    }
                }

                if is_overlay {
                    div { class: "radar-drag-hint", "przeciągnij radar" }
                }
            }
        }
    }
}
