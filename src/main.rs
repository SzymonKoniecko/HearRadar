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

/// Wspólna obsługa `--recordSample` i `--recordVideo`. `pos` to indeks nazwy komendy.
/// Opcje: --device, --game, --scene, --mode, --output, --spatial, --game-volume,
/// --windows-volume, --seconds; dla wideo także --ffmpeg, --window, --fps.
fn record_command(args: &[String], pos: usize, with_video: bool) {
    let command = if with_video { "--recordVideo" } else { "--recordSample" };
    let Some(name) = args.get(pos + 1) else {
        eprintln!("Użycie: radar {command} <nazwa sesji> [--device <urządzenie>] [--game <gra>] ...");
        std::process::exit(2);
    };
    check_option_values(
        args,
        &[
            "--device", "--game", "--scene", "--mode", "--output", "--spatial", "--game-volume",
            "--windows-volume", "--seconds", "--fps", "--ffmpeg", "--window",
        ],
    );
    let device = option_value(args, "--device").unwrap_or("CABLE Input");
    let meta = sound::recorder::SessionMeta {
        game: option_value(args, "--game").map(str::to_string),
        scene: option_value(args, "--scene").map(str::to_string),
        mode: option_value(args, "--mode").map(str::to_string),
        output: option_value(args, "--output").map(str::to_string),
        spatial_sound: option_value(args, "--spatial").map(str::to_string),
        game_volume: option_value(args, "--game-volume").map(str::to_string),
        windows_volume: option_value(args, "--windows-volume").map(str::to_string),
    };
    let duration = option_value(args, "--seconds").map(|s| match s.parse::<u64>() {
        Ok(n) => Duration::from_secs(n),
        Err(_) => {
            eprintln!("Błąd: --seconds wymaga liczby całkowitej");
            std::process::exit(2);
        }
    });
    let video = with_video.then(|| {
        let fps = option_value(args, "--fps").map_or(30, |s| match s.parse::<u32>() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("Błąd: --fps wymaga liczby całkowitej");
                std::process::exit(2);
            }
        });
        sound::video::VideoOptions {
            ffmpeg: option_value(args, "--ffmpeg").unwrap_or("ffmpeg").to_string(),
            source: match option_value(args, "--window") {
                Some(title) => sound::video::VideoSource::Window(title.to_string()),
                None => sound::video::VideoSource::Desktop,
            },
            fps,
        }
    });
    if let Err(err) = sound::recorder::run(name, device, meta, duration, video) {
        eprintln!("Błąd: {err}");
        std::process::exit(1);
    }
}

/// Sprawdza, czy każda z podanych opcji ma wartość. Bez wartości `option_value`
/// zwraca `None` i opcja zostaje po cichu pominięta, np. `--ffmpeg` bez ścieżki.
fn check_option_values(args: &[String], flags: &[&str]) {
    for flag in flags {
        if let Some(pos) = args.iter().position(|a| a == flag) {
            match args.get(pos + 1) {
                Some(value) if !value.starts_with("--") => {}
                _ => {
                    eprintln!("Błąd: {flag} wymaga wartości");
                    std::process::exit(2);
                }
            }
        }
    }
}

/// Liczba z dwoma miejscami po przecinku albo `-`, gdy wartości nie ma.
fn fmt_opt(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"))
}

/// Wartość `--onset` w sekundach. Domyślnie 0,2 s, czyli zdarzenie w klipie z `hearanalytics`.
fn onset_seconds(args: &[String]) -> f64 {
    match option_value(args, "--onset") {
        Some(s) => s.parse::<f64>().unwrap_or_else(|_| {
            eprintln!("Błąd: --onset wymaga liczby");
            std::process::exit(2);
        }),
        None => 0.2,
    }
}

/// Wartość opcji `flag` (np. `--game "Nazwa"`), jeśli została podana.
fn option_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
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

    // Tryb bez GUI: `radar --channel-test <nazwa urządzenia>` gra sinus po kolei
    // na każdym kanale i sprawdza, który indeks dostaje sygnał (patrz `sound::channel_test`).
    if let Some(pos) = args.iter().position(|a| a == "--channel-test") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --channel-test <nazwa urządzenia>");
            std::process::exit(2);
        };
        if let Err(err) = sound::channel_test::run(name) {
            eprintln!("Błąd: {err}");
            std::process::exit(1);
        }
        return;
    }

    // Tryb bez GUI: `radar --analyze-clips <folder> [--onset S]` liczy kierunek (ITD)
    // dla każdego pliku WAV w folderze. Zdarzenie w klipie jest w chwili --onset (domyślnie 0,2 s).
    if let Some(pos) = args.iter().position(|a| a == "--analyze-clips") {
        let Some(folder) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --analyze-clips <folder> [--onset S]");
            std::process::exit(2);
        };
        let onset = onset_seconds(&args);
        match sound::clip_analysis::analyze_folder(std::path::Path::new(folder), onset) {
            Ok(reports) => {
                println!("plik\tazymut_deg\tpewnosc\tild_db");
                for r in &reports {
                    println!(
                        "{}\t{}\t{}\t{}",
                        r.file,
                        fmt_opt(r.azimuth_deg),
                        fmt_opt(r.confidence),
                        fmt_opt(r.ild_db)
                    );
                }
                let with_angle = reports.iter().filter(|r| r.azimuth_deg.is_some()).count();
                println!("\nplików: {}, z kierunkiem: {}", reports.len(), with_angle);
            }
            Err(err) => {
                eprintln!("Błąd: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Tryb bez GUI: `radar --radar-dump <wav> [--every S]` pokazuje tekstowo, co radar 3+3 wyświetliłby
    // w czasie: wysokości łuków, widoczność stron, podświetlenie i kolory.
    if let Some(pos) = args.iter().position(|a| a == "--radar-dump") {
        let Some(file) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --radar-dump <plik.wav> [--every S]");
            std::process::exit(2);
        };
        let every = option_value(&args, "--every").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.05);
        let steps_gain = option_value(&args, "--steps-gain").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
        match sound::playback::radar_dump(std::path::Path::new(file), every, steps_gain) {
            Ok(text) => print!("{text}"),
            Err(err) => {
                eprintln!("Błąd: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Tryb bez GUI: `radar --classify-session <sesja lub audio.wav>` przechodzi przez całe nagranie
    // tym samym potokiem co radar na żywo i wypisuje decyzje warstw. Do porównania z `events.csv`.
    if let Some(pos) = args.iter().position(|a| a == "--classify-session") {
        let Some(source) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --classify-session <sesja lub audio.wav>");
            std::process::exit(2);
        };
        let source = std::path::Path::new(source);
        let wav = if source.is_dir() {
            source.join("audio.wav")
        } else {
            source.to_path_buf()
        };
        match sound::playback::classify_recording(&wav) {
            Ok(hits) => {
                println!("czas_s\tpoziom_dbfs\tjasnosc_db\tklasa\todstep_s\twarstwa\tpan\tja");
                for h in &hits {
                    println!(
                        "{:.3}\t{:.1}\t{}\t{}\t{}\t{}\t{:+.2}\t{}",
                        h.onset_s,
                        h.peak_dbfs,
                        fmt_opt(h.brightness_db),
                        h.class.map_or_else(|| "-".to_string(), |c| format!("{c:?}")),
                        fmt_opt(h.gap_s),
                        h.layer.label(),
                        h.pan,
                        h.is_own_candidate()
                    );
                }
                println!("\nzdarzeń: {}", hits.len());
            }
            Err(err) => {
                eprintln!("Błąd: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Tryb bez GUI: `radar --classify-clips <folder> [--onset S]` przypisuje każdemu klipowi
    // klasę A–D i warstwę (strzały, kroki, reszta) tą samą drogą co radar na żywo.
    // Do porównania z `events.csv` z Pythona, kolumna `klasa_auto`.
    if let Some(pos) = args.iter().position(|a| a == "--classify-clips") {
        let Some(folder) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --classify-clips <folder> [--onset S]");
            std::process::exit(2);
        };
        let onset = onset_seconds(&args);
        match sound::clip_analysis::classify_folder(std::path::Path::new(folder), onset) {
            Ok(rows) => {
                println!("plik\tpoziom_dbfs\tjasnosc_db\tklasa\twarstwa");
                for (file, hit) in &rows {
                    match hit {
                        Some(h) => println!(
                            "{file}\t{:.1}\t{}\t{}\t{}",
                            h.peak_dbfs,
                            fmt_opt(h.brightness_db),
                            h.class.map_or_else(|| "-".to_string(), |c| format!("{c:?}")),
                            h.layer.label()
                        ),
                        None => println!("{file}\t-\t-\t-\t-"),
                    }
                }
                println!("\nklipów: {}", rows.len());
            }
            Err(err) => {
                eprintln!("Błąd: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Tryb bez GUI: `radar --recordSample <nazwa sesji>` nagrywa dwa pierwsze kanały
    // urządzenia do WAV z metadanymi, a `radar --recordVideo <nazwa sesji>` dodatkowo
    // obraz przez ffmpeg (patrz `sound::recorder` i `sound::video`).
    if args.iter().any(|a| a == "--record") {
        eprintln!("Komenda --record została zmieniona: użyj --recordSample (dźwięk) albo --recordVideo (dźwięk + wideo).");
        std::process::exit(2);
    }
    if let Some(pos) = args.iter().position(|a| a == "--recordSample") {
        record_command(&args, pos, false);
        return;
    }
    if let Some(pos) = args.iter().position(|a| a == "--recordVideo") {
        record_command(&args, pos, true);
        return;
    }

    // Tryb bez GUI: `radar --monitor <nazwa urządzenia> [--seconds N]` pokazuje na
    // żywo, które kanały urządzenia niosą dźwięk (do sprawdzenia routingu gry).
    if let Some(pos) = args.iter().position(|a| a == "--monitor") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("Użycie: radar --monitor <nazwa urządzenia> [--seconds N]");
            std::process::exit(2);
        };
        let seconds = args
            .iter()
            .position(|a| a == "--seconds")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        if let Err(err) = sound::channel_test::monitor(name, Duration::from_secs(seconds)) {
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
