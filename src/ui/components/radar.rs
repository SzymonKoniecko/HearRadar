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
    let radius = 100.0f32;

    // Kąt: 0° = przód (góra), +90° = prawo, -90° = lewo
    let rad = (angle - 90.0).to_radians();
    let distance = (intensity * radius).clamp(10.0, radius);
    let blip_x = center + distance * rad.cos();
    let blip_y = center + distance * rad.sin();

    rsx! {
        div {
            style: "position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;",
            onmousedown: move |_| {
                if is_overlay {
                    desktop.drag();
                }
            },

            // Przycisk powrotu (tylko overlay)
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
                style: "width: 100%; height: 100%; max-width: 100%; max-height: 100%;",

                // Pierścień zewnętrzny (100%)
                circle {
                    cx: "{center}", cy: "{center}", r: "{radius}",
                    fill: "none",
                    stroke: "rgba(0,255,157,0.5)",
                    stroke_width: "1.5",
                }

                // Pierścień środkowy (66%)
                circle {
                    cx: "{center}", cy: "{center}", r: "{radius * 0.66}",
                    fill: "none",
                    stroke: "rgba(0,255,157,0.25)",
                    stroke_width: "1",
                }

                // Pierścień wewnętrzny (33%)
                circle {
                    cx: "{center}", cy: "{center}", r: "{radius * 0.33}",
                    fill: "none",
                    stroke: "rgba(0,255,157,0.25)",
                    stroke_width: "1",
                }

                // Środek
                circle {
                    cx: "{center}", cy: "{center}", r: "2.5",
                    fill: "rgba(0,255,157,0.8)",
                }

                // Blip dźwiękowy
                if intensity > 0.02 {
                    circle {
                        cx: "{blip_x}", cy: "{blip_y}",
                        r: "6",
                        fill: "#ff0055",
                        stroke: "rgba(255,255,255,0.6)",
                        stroke_width: "1.5",
                    }
                }
            }
        }
    }
}
