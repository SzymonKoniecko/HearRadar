use dioxus::prelude::*;

mod ui;
mod sound;
fn main() {
    dioxus::launch(ui::ui_main::app);
}
