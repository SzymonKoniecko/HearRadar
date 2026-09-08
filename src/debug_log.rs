//! Dublowanie `println!`-owych logów diagnostycznych do pliku, obok konsoli —
//! żeby dało się przesłać komuś log z sesji (np. do debugowania zachowania
//! radaru) bez ręcznego kopiowania z terminala. Nie zastępuje `println!`
//! wszędzie — tylko w miejscach, które i tak już logują stan audio (patrz
//! `sound::devices`, `sound::mono_router`), przez makro [`dlog!`].

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static LOG_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();

/// Plik logu leży obok `radar_config.json` (patrz `config_store::config_dir`)
/// — jedno, znane miejsce na wszystkie zapisywane dane aplikacji, niezależnie
/// od katalogu roboczego, z którego uruchomiono `cargo run`/gotową binarkę.
fn log_file_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("debug.log")
}

/// Otwiera (append, tworzy jeśli brak) plik logu. Wołane raz, na starcie
/// aplikacji (patrz `main.rs`) — dopóki nie zostanie wywołane, [`log_line`]
/// jest cichym no-opem (np. w testach jednostkowych, które nie wołają `init`).
pub fn init() {
    let path = log_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = OpenOptions::new().create(true).append(true).open(&path).ok();
    // `OnceLock::set` ignoruje drugie wywołanie zamiast panikować — `init()`
    // wołane więcej niż raz (np. przez pomyłkę) nie jest błędem.
    let _ = LOG_FILE.set(Mutex::new(file));
    log_line(&format!(
        "\n=== HearRadar start (unix_ts={}) ===",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
}

/// Dopisuje jedną linię do pliku logu (flush od razu — to log diagnostyczny,
/// ma przetrwać nawet twardy crash procesu, nie tylko czyste zamknięcie).
pub fn log_line(line: &str) {
    let Some(mutex) = LOG_FILE.get() else { return };
    let Ok(mut guard) = mutex.lock() else { return };
    let Some(file) = guard.as_mut() else { return };
    let _ = writeln!(file, "{line}");
    let _ = file.flush();
}

/// Jak `println!`, ale dodatkowo dopisuje tę samą linię do pliku logu (patrz
/// [`log_line`]) — używane w miejscach, gdzie dziennik z sesji potrzebny jest
/// do debugowania zachowania radaru bez terminala pod ręką.
#[macro_export]
macro_rules! dlog {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        $crate::debug_log::log_line(&line);
    }};
}
