use crate::ui::components::SoundDeviceDropdown;
use dioxus::prelude::*;

static MAIN_CSS: Asset = asset!("assets/main.css");

#[component]
pub fn App() -> Element {
    rsx! {
        document::Stylesheet {
            href: MAIN_CSS
        }
        div { id: "app",
            div { id: "header", h1 { "Hear Radar !!!" } }
            div {
                class: "body",
                SoundDeviceDropdown {}
            }
        }
    }
}
