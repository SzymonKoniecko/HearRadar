use dioxus::prelude::*;
use crate::sound;
use crate::ui;

#[component]
pub fn app() -> Element {
    rsx! {
        h1 { text_align: "center", "Hear Radar !!!" }
        ui::devices_dropdown_component::soundDeviceDropdown {}
    }
}