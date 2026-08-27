use dioxus::prelude::*;

mod ui;

fn main() {
    dioxus::launch(ui::ui_component::app);
}
