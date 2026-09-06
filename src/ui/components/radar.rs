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

    // Kąt: -90° (lewo), 0° (przód/góra), +90° (prawo)
    let rad = (angle - 90.0).to_radians();
    let distance = (intensity * radius).clamp(15.0, radius);
    let blip_x = center + distance * rad.cos();
    let blip_y = center + distance * rad.sin();

    // Kropki kierunkowe rozmieszczone co 30° na obwodzie okręgu (12 pozycji, jak na tarczy zegara)
    let direction_dots: Vec<(f32, f32)> = (0..12)
        .map(|i| {
            let a = (i as f32 * 30.0 - 90.0).to_radians();
            (center + radius * a.cos(), center + radius * a.sin())
        })
        .collect();

    rsx! {
        div {
            onmousedown: move |_| {
                if is_overlay {
                    desktop.drag();
                }
            },

            if is_overlay {
                button {
                    title: "Otwórz panel ustawień",
                    style: "position: absolute; top: 4px; right: 4px; z-index: 10;",
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

                // Surowy okrąg zasięgu radaru
                circle {
                    cx: "{center}",
                    cy: "{center}",
                    r: "{radius}",
                    fill: "none",
                    stroke: "white",
                    stroke_width: "1.5",
                }

                // Środek radaru
                circle {
                    cx: "{center}",
                    cy: "{center}",
                    r: "2",
                    fill: "white",
                }

                // Kropki oznaczające kierunki na obwodzie
                for (dx , dy) in direction_dots.iter() {
                    circle {
                        cx: "{dx}",
                        cy: "{dy}",
                        r: "2.5",
                        fill: "gray",
                    }
                }

                // Wykryty punkt dźwięku - kropka w kierunku i odległości zależnej od intensywności
                if intensity > 0.02 {
                    circle {
                        cx: "{blip_x}",
                        cy: "{blip_y}",
                        r: "{5.0 + intensity * 5.0}",
                        fill: "red",
                    }
                }
            }
        }
    }
}