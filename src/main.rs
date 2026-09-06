use dioxus::desktop::{Config, WindowBuilder, tao::dpi::LogicalSize};
use dioxus::prelude::*;

mod sound;
mod ui;

fn main() {
    let window = WindowBuilder::new()
        .with_title("Hear Radar")
        .with_transparent(true)
        .with_inner_size(LogicalSize::new(860.0, 700.0));

    LaunchBuilder::desktop()
        .with_cfg(Config::new().with_window(window))
        .launch(ui::App);
}
