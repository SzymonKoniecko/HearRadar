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
pub(crate) fn config_dir() -> PathBuf {
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
    use std::sync::Mutex;

    /// `save`/`load` operują na JEDNYM ustalonym pliku w repo (celowo, patrz
    /// komentarz modułu), więc testy, które faktycznie piszą/czytają z
    /// dysku, muszą się wykluczać — `cargo test` domyślnie odpala testy
    /// równolegle, a bez tej blokady dwa takie testy naprzemiennie nadpisują
    /// sobie ten sam plik (obserwowane jako losowe niepowodzenia asercji).
    static FILE_TEST_LOCK: Mutex<()> = Mutex::new(());

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
        let _guard = FILE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    fn loading_config_saved_before_overlay_click_through_existed_defaults_to_true() {
        // Symuluje plik zapisany starszą wersją aplikacji, sprzed dodania
        // `overlay_click_through` do `RadarConfig` — brakujące pole musi
        // dostać `true` (patrz `#[serde(default = "default_overlay_click_through")]`),
        // inaczej wczytanie starego configu niechcący zablokowałoby klikanie
        // w grę pod overlayem.
        let json = r#"{
            "anchor_x": 1.0,
            "anchor_y": 1.0,
            "radius": 125.0,
            "bg_alpha": 0.82,
            "fg_alpha": 1.0,
            "show_labels": false,
            "high_contrast": false,
            "decay_secs": 2.5,
            "color_rules": []
        }"#;
        let cfg: RadarConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.overlay_click_through);
    }

    #[test]
    fn applying_a_built_in_color_profile_survives_save_and_load() {
        // To dokładnie ścieżka z UI: użytkownik klika profil w
        // `ui_color_profile_picker`, co nadpisuje `color_rules` i woła
        // `write_radar_config()` — a ta od teraz od razu zapisuje na dysk
        // (patrz `HearRadarApp::write_radar_config`), więc nowy profil musi
        // przetrwać zapis/wczytanie identycznie jak reszta konfiguracji.
        let _guard = FILE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let profile = crate::sound::color_profiles::built_in_profiles()
            .first()
            .expect("at least one built-in profile should exist");

        let cfg = RadarConfig {
            color_rules: (profile.build)(),
            ..RadarConfig::default()
        };
        assert!(!cfg.color_rules.is_empty(), "profile should produce at least one rule");

        let path = save(&cfg).expect("save should succeed");
        let loaded = load().expect("load should succeed right after save");
        assert_eq!(cfg.color_rules, loaded.color_rules);

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
