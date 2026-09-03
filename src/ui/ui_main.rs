use crate::sound;
use crate::ui;
use dioxus::prelude::*;

static MAIN_CSS: Asset = asset!("src/ui/assets/main.css");

#[component]
pub fn app() -> Element {
    rsx! {
        document::Stylesheet {
            href: MAIN_CSS
        }
        div { id:"app",
            div { id: "header", h1{"Hear Radar !!!"} }
            div{
                class:"body",
                ui::devices_component::ui_soundDeviceDropdown {}
            }
    }
    }
}
