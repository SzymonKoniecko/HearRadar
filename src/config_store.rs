//! Zapisywanie/wczytywanie konfiguracji radaru (`RadarConfig`, w tym reguły
//! kolorowania) do lokalnego pliku JSON, żeby ustawienia przetrwały
//! restart aplikacji. Zapisujemy zwykły, czytelny dla człowieka JSON (nie
//! format binarny), więc plik można też ręcznie edytować albo wysłać
//! komuś innemu jako gotowy zestaw reguł.
//!
//! Plik trzymany jest W REPO (katalog `config/` obok `Cargo.toml`), nie w
//! systemowym katalogu ustawień użytkownika — celowo, żeby konfigurację dało
//! się łatwo znaleźć, wersjonować albo skopiować razem z projektem zamiast
//! szukać jej w `~/Library/Application Support` czy `%APPDATA%`.

use crate::sound::devices::RadarConfig;
use anyhow::{Context, Result};
use std::path::PathBuf;

const CONFIG_FILE_NAME: &str = "radar_config.json";

/// Katalog `config/` w korzeniu repozytorium. `CARGO_MANIFEST_DIR` jest
/// wypalane w binarkę w czasie kompilacji i wskazuje katalog z `Cargo.toml`
/// — działa niezależnie od tego, z jakiego katalogu roboczego uruchomiono
/// `cargo run`/gotową binarkę.
fn config_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config")
}

/// Pełna ścieżka do pliku z zapisaną konfiguracją radaru.
pub fn config_file_path() -> PathBuf {
    config_dir().join(CONFIG_FILE_NAME)
}

/// Zapisuje konfigurację jako czytelny (pretty-printed) JSON. Tworzy katalog
/// konfiguracyjny, jeśli jeszcze nie istnieje. Zwraca ścieżkę zapisanego pliku.
pub fn save(cfg: &RadarConfig) -> Result<PathBuf> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("nie udało się utworzyć katalogu {}", dir.display()))?;

    let path = config_file_path();
    let json = serde_json::to_string_pretty(cfg).context("serializacja konfiguracji do JSON")?;
    std::fs::write(&path, json)
        .with_context(|| format!("nie udało się zapisać pliku {}", path.display()))?;
    Ok(path)
}

/// Wczytuje wcześniej zapisaną konfigurację. Zwraca błąd, jeśli plik nie
/// istnieje albo ma nieprawidłowy format — wywołujący decyduje, czy w takim
/// wypadku użyć `RadarConfig::default()`.
pub fn load() -> Result<RadarConfig> {
    let path = config_file_path();
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("nie udało się odczytać pliku {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("nieprawidłowy JSON w {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let cfg = RadarConfig {
            radius: 200.0,
            decay_secs: 2.0,
            ..RadarConfig::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RadarConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn save_then_load_round_trips_on_the_real_filesystem() {
        let cfg = RadarConfig {
            radius: 77.0,
            high_contrast: true,
            ..RadarConfig::default()
        };
        let path = save(&cfg).expect("save should succeed");
        assert!(path.exists());

        let loaded = load().expect("load should succeed right after save");
        assert_eq!(cfg, loaded);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_file_path_lives_in_repo_config_dir() {
        let path = config_file_path();
        assert_eq!(path.file_name().unwrap(), CONFIG_FILE_NAME);
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "config");
        assert!(
            path.starts_with(env!("CARGO_MANIFEST_DIR")),
            "plik konfiguracyjny powinien być w repo, nie w systemowym katalogu ustawień"
        );
    }
}
