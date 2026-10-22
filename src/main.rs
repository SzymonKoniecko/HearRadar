mod config_store;
mod debug_log;
mod sound;
mod ui;

use cpal::traits::{DeviceTrait, HostTrait};
use sound::devices::SharedAudioAnalysis;
use std::time::{Duration, Instant};
use ui::HearRadarApp;

/// Tryb bez GUI: `radar --listen-cable [--seconds N]` nasłuchuje urządzenia
/// „Cable In 16” i wypisuje co pół sekundy poziomy RMS wszystkich kanałów.
fn listen_cable(seconds: u64) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let name = host
        .output_devices()?
        .filter_map(|d| d.name().ok())
        .find(|n| n.to_lowercase().contains("cable in 16"))
        .ok_or_else(|| anyhow::anyhow!("nie znaleziono urządzenia 'Cable In 16'"))?;
    println!("Nasłuchuję: {name}");

    let analysis = SharedAudioAnalysis::new();
    let _stream = sound::devices::fetch_audio_from_device(&name, analysis.clone())?;

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) {
        std::thread::sleep(Duration::from_millis(500));
        if let Ok(guard) = analysis.lock() {
            let rms: Vec<String> = guard.channel_rms.iter().map(|v| format!("{v:.4}")).collect();
            println!(
                "t={:.1}s ch={} rms=[{}]",
                start.elapsed().as_secs_f32(),
                guard.channel_count,
                rms.join(", ")
            );
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--listen-cable") {
        let seconds = args
            .iter()
            .position(|a| a == "--seconds")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        if let Err(err) = listen_cable(seconds) {
            eprintln!("Błąd: {err}");
            std::process::exit(1);
        }
        return;
    }

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
