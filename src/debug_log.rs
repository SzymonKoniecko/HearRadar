//! Makro [`dlog!`] do logów diagnostycznych.
//!
//! Zapis do pliku (`init`, `log_file_path`) został usunięty — nic go nie
//! wywoływało. [`log_line`] nadal wysyła linię kanałem do wątku, ale nikt
//! już tego wątku nie uruchamia, więc obecnie każde `dlog!` tylko formatuje
//! tekst i kończy się bez efektu. Jeśli logi nie są już potrzebne, cały moduł
//! i wywołania `dlog!` można usunąć.
//!
//! Zasady z poprzedniej wersji nadal obowiązują: nic nie drukujemy na konsolę
//! z wątku audio (zaznaczenie tekstu w konsoli Windows wstrzymuje zapis do
//! stdout) i nie zapisujemy na dysk w ścieżce czasu rzeczywistego.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

static LOG_SENDER: OnceLock<Mutex<Sender<String>>> = OnceLock::new();

/// Wysyła jedną linię kanałem do wątku zapisu (patrz doc-komentarz modułu) —
/// tanie, nieblokujące, bezpieczne do wołania z wątku czasu rzeczywistego audio.
/// Bez ustawionego kanału jest cichym no-opem.
pub fn log_line(line: &str) {
    let Some(mutex) = LOG_SENDER.get() else { return };
    let Ok(sender) = mutex.lock() else { return };
    let _ = sender.send(line.to_string());
}

/// Formatuje linię diagnostyczną i przekazuje ją do [`log_line`].
#[macro_export]
macro_rules! dlog {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        $crate::debug_log::log_line(&line);
    }};
}
