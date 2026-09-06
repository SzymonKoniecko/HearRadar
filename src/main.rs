use dioxus::desktop::{Config, WindowBuilder, tao::dpi::LogicalSize};
use dioxus::prelude::*;

mod sound;
mod ui;

use sound::devices::SharedAudioAnalysis;

fn main() {
    // Współdzielony stan analizy audio — ten sam Arc przekazywany do obu okien
    let analysis = SharedAudioAnalysis::new();
    let analysis_egui = analysis.clone();
    let analysis_dioxus = analysis.clone();

    // Okno radaru egui w osobnym wątku systemowym
    std::thread::spawn(move || {
        ui::launch_radar_window(analysis_egui);
    });

    // Główne okno Dioxus: konfiguracja, wybór urządzenia audio
    // SharedAudioAnalysis wstrzykiwany przez with_context, pobierany w App przez consume_context
    let window = WindowBuilder::new()
        .with_title("Hear Radar")
        .with_transparent(true)
        .with_inner_size(LogicalSize::new(860.0, 700.0));

    LaunchBuilder::desktop()
        .with_cfg(Config::new().with_window(window))
        .with_context(analysis_dioxus)
        .launch(ui::App);
}
