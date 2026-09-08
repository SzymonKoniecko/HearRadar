mod config_store;
mod debug_log;
mod sound;
mod ui;

use sound::devices::SharedAudioAnalysis;
use ui::HearRadarApp;

fn main() {
    // Zapis do pliku wyłączony (patrz `debug_log::init`) — narósł do dziesiątek
    // tysięcy linii w kilka minut i nie był już potrzebny. `dlog!` w
    // `sound::devices`/`mono_router` bez wywołania `init()` samoistnie cichnie
    // do samego `println!` na konsolę (patrz `debug_log::log_line`) — odkomentuj
    // poniższe wywołanie, żeby wrócić do dublowania logów do `config/debug.log`.
    // debug_log::init();

    let analysis = SharedAudioAnalysis::new();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Hear Radar")
            .with_transparent(true)
            .with_inner_size([860.0, 700.0]),
        ..Default::default()
    };

    // Jeden `eframe::run_native` na wątku głównym — jedyny natywny event
    // loop w procesie. Okno overlay z radarem jest renderowane jako drugi
    // "viewport" tej samej aplikacji (patrz `ui::app`), a nie jako osobne
    // okno w osobnym wątku, co wcześniej łamało wymóg macOS/winit "UI
    // event loop musi działać na wątku głównym".
    eframe::run_native(
        "HearRadar",
        options,
        Box::new(|_cc| Ok(Box::new(HearRadarApp::new(analysis)))),
    )
    .expect("eframe failed to start");
}
