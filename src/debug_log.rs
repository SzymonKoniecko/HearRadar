//! Dublowanie `println!`-owych logów diagnostycznych do pliku, obok konsoli —
//! żeby dało się przesłać komuś log z sesji (np. do debugowania zachowania
//! radaru) bez ręcznego kopiowania z terminala. Nie zastępuje `println!`
//! wszędzie — tylko w miejscach, które i tak już logują stan audio (patrz
//! `sound::devices`, `sound::mono_router`), przez makro [`dlog!`].
//!
//! [`dlog!`] jest wołane m.in. z callbacku audio (`sound::devices::update_analysis`)
//! — CO RAMKĘ (~10ms), bezwarunkowo. Zapis do pliku z jawnym `flush()` (patrz
//! niżej) potrafi się zablokować na kilka-kilkanaście ms (dysk, antywirus
//! skanujący plik) — na wątku czasu rzeczywistego audio to wystarczy, żeby
//! przekroczyć deadline bufora i wywołać słyszalne przycięcie w dźwięku
//! przekierowanym przez `mono_router`. Dlatego faktyczny zapis na dysk
//! przeniesiony jest na OSOBNY wątek roboczy (patrz [`init`]): [`log_line`]
//! na wątku wołającym (audio albo inny) robi tylko tanie, nieblokujące
//! wysłanie linii przez kanał — samo `write`/`flush` dzieje się asynchronicznie,
//! poza ścieżką czasu rzeczywistego.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock};

static LOG_SENDER: OnceLock<Mutex<Sender<String>>> = OnceLock::new();

/// Plik logu leży obok `radar_config.json` (patrz `config_store::config_dir`)
/// — jedno, znane miejsce na wszystkie zapisywane dane aplikacji, niezależnie
/// od katalogu roboczego, z którego uruchomiono `cargo run`/gotową binarkę.
fn log_file_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("debug.log")
}

/// Otwiera (append, tworzy jeśli brak) plik logu i startuje wątek roboczy,
/// który jako jedyny go dotyka. Wołane raz, na starcie aplikacji (patrz
/// `main.rs`) — dopóki nie zostanie wywołane, [`log_line`] jest cichym
/// no-opem (np. w testach jednostkowych, które nie wołają `init`).
pub fn init() {
    let path = log_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path).ok();

    let (tx, rx) = mpsc::channel::<String>();
    // Jedyny wątek, który kiedykolwiek pisze do `file` — reszta aplikacji
    // (w tym wątek audio) tylko wrzuca linie do kanału (patrz [`log_line`]).
    std::thread::spawn(move || {
        for line in rx {
            if let Some(f) = file.as_mut() {
                let _ = writeln!(f, "{line}");
                let _ = f.flush();
            }
        }
    });

    // `OnceLock::set` ignoruje drugie wywołanie (upuszczając `tx` z tej
    // drugiej próby, co od razu i cicho kończy jej wątek roboczy) zamiast
    // panikować — `init()` wołane więcej niż raz nie jest błędem.
    let _ = LOG_SENDER.set(Mutex::new(tx));
    log_line(&format!(
        "\n=== HearRadar start (unix_ts={}) ===",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
}

/// Wysyła jedną linię do zapisania przez wątek roboczy (patrz [`init`]) —
/// tanie, nieblokujące (samo `Sender::send` na kanale, bez dotykania dysku),
/// więc bezpieczne do wołania z wątku czasu rzeczywistego audio.
pub fn log_line(line: &str) {
    let Some(mutex) = LOG_SENDER.get() else { return };
    let Ok(sender) = mutex.lock() else { return };
    let _ = sender.send(line.to_string());
}

/// Zapisuje sformatowaną linię do pliku logu (patrz [`log_line`]) — używane
/// w miejscach, gdzie dziennik z sesji potrzebny jest do debugowania
/// zachowania radaru bez terminala pod ręką.
///
/// CELOWO nie wypisuje nic na konsolę (`println!`/`eprintln!`) — na Windows
/// zaznaczenie tekstu w oknie konsoli (tryb "Szybkiej Edycji") wstrzymuje
/// zapis do stdout/stderr KAŻDEGO procesu, dopóki użytkownik nie odklika
/// zaznaczenia; wołane z wątku czasu rzeczywistego audio (patrz
/// `sound::devices::update_analysis`) byłoby to tym samym zagrożeniem
/// przycięciami, co niegdysiejszy synchroniczny zapis na dysk (patrz
/// doc-komentarz modułu). [`log_line`] samo w sobie już jest bezpieczne
/// (tylko `send` na kanale) — ten brak `println!` to dodatkowe zabezpieczenie
/// na wypadek, gdyby ktoś kiedyś chciał je z powrotem dopisać.
#[macro_export]
macro_rules! dlog {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        $crate::debug_log::log_line(&line);
    }};
}
