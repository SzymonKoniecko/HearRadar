//! Logger próbek klasyfikacji koloru do ręcznej analizy — pozwala zebrać,
//! jakie kolory (albo brak markera w ogóle) dostają realne dźwięki z gry, żeby
//! offline (w Excelu/arkuszu) dostroić granice i wagi [`crate::sound::spectrum::ColorRule`]
//! w configu. Włączane checkboxem w [`crate::ui::app::HearRadarApp::ui_color_rules_panel`].
//!
//! Każde włączenie loggera otwiera NOWY plik CSV (kolejny numer w katalogu
//! configu, patrz [`crate::config_store::config_dir`]) zamiast doklejać do
//! jednego, rosnącego bez końca pliku — dzięki temu każda sesja logowania da
//! się analizować osobno.
//!
//! [`SampleLogger::log_sample`] jest wołane z callbacku audio
//! (`sound::devices::update_analysis`), do kilku razy na ramkę (jeden raz na
//! słyszalny kanał) — zapis CSV z jawnym `flush()` na tym wątku miałby ten sam
//! problem co niegdyś `crate::dlog!` (patrz `debug_log`): potrafi się
//! zablokować na tyle długo, żeby wywołać słyszalne przycięcie w dźwięku
//! przekierowanym przez `mono_router`. Dlatego, tak samo jak tam, faktyczny
//! zapis pliku dzieje się na OSOBNYM wątku roboczym — `log_sample` na wątku
//! audio robi tylko tanie, nieblokujące wysłanie już złożonego wiersza przez
//! kanał.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;

enum LoggerCommand {
    Enable,
    Disable,
    Row(SampleRow),
}

const FILE_PREFIX: &str = "sample_log_";
const FILE_SUFFIX: &str = ".csv";
const CSV_HEADER: &str =
    "unix_ts_ms,channel_index,rms,scaled_energy,angle_deg,sensitivity,marker_spawned,rule_index,rule_label,color_hex,color_r,color_g,color_b\n";

/// Jedna sklasyfikowana próbka (jeden kanał, jedna ramka analizy) do zapisu.
pub struct SampleRow {
    pub unix_ts_ms: u128,
    pub channel_index: usize,
    pub rms: f32,
    pub scaled_energy: f32,
    /// Płynny kąt kierunku (patrz [`crate::sound::direction::onset_angle_deg`])
    /// liczony dla TEJ próbki niezależnie od tego, czy przebiła próg onsetu —
    /// bez tego z logu nie da się odtworzyć, z jakiego kierunku dźwięk
    /// faktycznie przyszedł, tylko sam poziom i kolor.
    pub angle_deg: f32,
    /// Wartość suwaka "Czułość radaru" w chwili próbki — wpływa zarówno na
    /// `scaled_energy`, jak i na próg onsetu, więc bez niej `marker_spawned`
    /// nie da się zinterpretować offline (ta sama `rms` przy innej czułości
    /// mogłaby, albo nie mogłaby, wygenerować marker).
    pub sensitivity: f32,
    /// Czy ta próbka realnie wygenerowała marker na radarze (przebiła próg
    /// onsetu) — `false` przy słyszalnym, ale niezaklasyfikowanym dźwięku.
    pub marker_spawned: bool,
    /// Indeks reguły w `radar_config.color_rules`, która faktycznie
    /// dostarczyła kolor (patrz [`crate::sound::spectrum::resolve_color`]) —
    /// `None`, gdy kolor jest fallbackiem (lista reguł pusta).
    pub rule_index: Option<usize>,
    pub rule_label: Option<String>,
    /// Kolor, jaki próbka FAKTYCZNIE dostała (ten sam RGB co na markerze).
    pub color: (u8, u8, u8),
}

#[derive(Debug)]
pub struct SampleLogger {
    /// Odzwierciedla stan NATYCHMIAST po `set_enabled` (patrz tam) — czytane
    /// z wątku audio na każdą ramkę, więc musi być tanie i nieblokujące;
    /// faktyczne otwarcie/zamknięcie pliku dogania to asynchronicznie na
    /// wątku roboczym (patrz [`LoggerCommand`]).
    enabled: AtomicBool,
    tx: Mutex<Sender<LoggerCommand>>,
}

impl SampleLogger {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<LoggerCommand>();
        // Jedyny wątek, który kiedykolwiek dotyka pliku CSV — audio tylko
        // wysyła gotowe wiersze/polecenia przez kanał (patrz `log_sample`/
        // `set_enabled`), bez czekania na zapis/flush na dysk.
        std::thread::spawn(move || {
            let mut file: Option<File> = None;
            for cmd in rx {
                match cmd {
                    LoggerCommand::Enable => file = open_next_log_file(),
                    LoggerCommand::Disable => file = None,
                    LoggerCommand::Row(row) => {
                        let Some(f) = file.as_mut() else { continue };
                        let (r, g, b) = row.color;
                        let line = format!(
                            "{},{},{:.4},{:.4},{:.1},{:.4},{},{},{},#{r:02x}{g:02x}{b:02x},{r},{g},{b}\n",
                            row.unix_ts_ms,
                            row.channel_index,
                            row.rms,
                            row.scaled_energy,
                            row.angle_deg,
                            row.sensitivity,
                            row.marker_spawned,
                            row.rule_index.map(|i| i.to_string()).unwrap_or_default(),
                            row.rule_label.as_deref().unwrap_or("").replace(',', ";"),
                        );
                        let _ = f.write_all(line.as_bytes());
                        let _ = f.flush();
                    }
                }
            }
        });
        Self { enabled: AtomicBool::new(false), tx: Mutex::new(tx) }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Włącza/wyłącza logowanie. Włączenie ZAWSZE otwiera świeży plik —
    /// nawet jeśli logger był już włączony — żeby kolejne kliknięcie
    /// checkboxa w UI zawsze dawało nową, czystą sesję logu do analizy.
    /// Samo otwarcie/zamknięcie pliku dzieje się na wątku roboczym (patrz
    /// [`SampleLogger::new`]); `enabled` (czytane przez [`is_enabled`])
    /// ustawiane jest tu, od razu, więc UI odzwierciedla stan natychmiast.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        let Ok(tx) = self.tx.lock() else { return };
        let _ = tx.send(if enabled { LoggerCommand::Enable } else { LoggerCommand::Disable });
    }

    /// Wysyła jedną próbkę do zapisania przez wątek roboczy (no-op, gdy
    /// logger wyłączony). Tanie, nieblokujące — bezpieczne do wołania z
    /// wątku czasu rzeczywistego audio (patrz doc-komentarz modułu).
    pub fn log_sample(&self, row: SampleRow) {
        if !self.is_enabled() {
            return;
        }
        let Ok(tx) = self.tx.lock() else { return };
        let _ = tx.send(LoggerCommand::Row(row));
    }
}

impl Default for SampleLogger {
    fn default() -> Self {
        Self::new()
    }
}

fn open_next_log_file() -> Option<File> {
    let dir = crate::config_store::config_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{FILE_PREFIX}{:04}{FILE_SUFFIX}", next_file_number(&dir)));
    let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(path).ok()?;
    file.write_all(CSV_HEADER.as_bytes()).ok()?;
    Some(file)
}

/// Najmniejszy numer WIĘKSZY niż jakikolwiek `sample_log_NNNN.csv` już
/// obecny w katalogu configu — kolejne sesje logowania nigdy nie nadpisują
/// poprzednich, nawet po restarcie aplikacji.
fn next_file_number(dir: &Path) -> u32 {
    let mut max_n = 0u32;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if let Some(n) = name
                .to_str()
                .and_then(|s| s.strip_prefix(FILE_PREFIX))
                .and_then(|s| s.strip_suffix(FILE_SUFFIX))
                .and_then(|s| s.parse::<u32>().ok())
            {
                max_n = max_n.max(n);
            }
        }
    }
    max_n + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_file_number_starts_at_one_for_empty_dir() {
        let dir = std::env::temp_dir().join(format!("hearradar_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(next_file_number(&dir), 1);
        std::fs::write(dir.join("sample_log_0001.csv"), "").unwrap();
        std::fs::write(dir.join("sample_log_0007.csv"), "").unwrap();
        std::fs::write(dir.join("not_a_log.csv"), "").unwrap();
        assert_eq!(next_file_number(&dir), 8);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn logger_is_disabled_by_default_and_toggles() {
        let logger = SampleLogger::new();
        assert!(!logger.is_enabled());
    }
}
