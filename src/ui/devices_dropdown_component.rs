use dioxus::prelude::*;
use crate::sound::devices;

#[component]
pub fn soundDeviceDropdown() -> Element{
    let devices = match devices::fetch_output_devices(){
        Ok(devices) => devices,
        Err(error) => {
            return rsx! {
                p { "Could not load output devices: {error}" }
            };
        }
    };

    let mut selected_name = use_signal(String::new);

    rsx! {
        h3 { "Audio device selector" }
        select {
            value: "{selected_name}",

            oninput: move |event| {
                selected_name.set(event.value());
            },

            option {
                value: "",
                disabled: true,
                "Choose a device...",
            }

            for device in &devices {
                option {
                    value: "{device.name}",
                    "{device.name}"
                }
            }
        }
        p {
            "Selected: {selected_name}"
        }
    }
}