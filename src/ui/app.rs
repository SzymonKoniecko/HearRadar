use crate::ui::components::SoundDeviceDropdown;
use dioxus::prelude::*;

const MAIN_CSS: &str = include_str!("../../assets/main.css");

#[component]
pub fn App() -> Element {
    rsx! {
        style { "{MAIN_CSS}" }
        div { id: "app",
            div { id: "header", h1 { "Hear Radar !!!" } }
            div {
                class: "body",
                SoundDeviceDropdown {}
            }
        }
    }
}
