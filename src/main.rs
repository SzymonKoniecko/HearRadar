mod config_store;
mod debug_log;
mod sound;
mod ui;

use sound::devices::SharedAudioAnalysis;
use ui::HearRadarApp;

fn main() {
    // Dubluje logi diagnostyczne (patrz `dlog!` w `sound::devices`/`mono_router`)
    // do pliku `config/debug.log` — patrz `debug_log::init`. Był tu wcześniej
    // wyłączony, bo `[ch]`/`[analysis]` logowane bezwarunkowo co callback audio
    // (~100x/s) narosły do dziesiątek tysięcy linii w kilka minut; te dwa logi
    // są teraz przerzedzone do ~co 2s (patrz `devices::ANALYSIS_LOG_COUNTER`),
    // więc plik znowu rośnie w rozsądnym tempie. Bez wywołania `init()` `dlog!`
    // jest CAŁKOWICIE ciche (żaden `println!` — patrz doc-komentarz
    // `debug_log`), więc bez tej linii nie ma żadnej diagnostyki, nawet na konsoli.
    debug_log::init();

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
