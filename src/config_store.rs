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
//!
//! Plik zapisuje się tylko na wyraźne życzenie (przycisk „Zapisz” w UI), nie
//! przy każdej zmianie. Obok można trzymać nazwane profile, czyli pełne kopie
//! konfiguracji w `config/profile/` (patrz [`profile_dir`]).
//!
//! Testy nie dotykają tego pliku: zapis i odczyt działają na ścieżce podanej
//! przez wywołującego (`save_to`/`load_from`), a testy podają plik tymczasowy.

use crate::sound::devices::RadarConfig;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

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

/// Zapisuje konfigurację do pliku w `config/`. Zwraca ścieżkę zapisanego pliku.
pub fn save(cfg: &RadarConfig) -> Result<PathBuf> {
    let path = config_file_path();
    save_to(&path, cfg)?;
    Ok(path)
}

/// Zapisuje konfigurację jako czytelny (pretty-printed) JSON pod podaną ścieżką.
/// Tworzy katalog nadrzędny, jeśli jeszcze nie istnieje.
pub fn save_to(path: &Path, cfg: &RadarConfig) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("nie udało się utworzyć katalogu {}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg).context("serializacja konfiguracji do JSON")?;
    std::fs::write(path, json)
        .with_context(|| format!("nie udało się zapisać pliku {}", path.display()))?;
    Ok(())
}

/// Wczytuje wcześniej zapisaną konfigurację z `config/`. Zwraca błąd, jeśli plik
/// nie istnieje albo ma nieprawidłowy format — wywołujący decyduje, czy w takim
/// wypadku użyć `RadarConfig::default()`.
pub fn load() -> Result<RadarConfig> {
    load_from(&config_file_path())
}

/// Wczytuje konfigurację z podanej ścieżki. Błędy jak w [`load`].
///
/// Plik zapisany starszą wersją aplikacji jest od razu przeliczany do bieżącej skali poziomu mono
/// (patrz [`RadarConfig::migrate_mono_level`]), żeby głośność w słuchawce nie skoczyła, oraz do
/// osobnych kolorów stron warstw (patrz [`RadarConfig::migrate_layer_side_colors`]), żeby łuki
/// warstw miały dotychczasowe kolory. Na dysk trafia to przy następnym zapisie konfiguracji.
pub fn load_from(path: &Path) -> Result<RadarConfig> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("nie udało się odczytać pliku {}", path.display()))?;
    let mut cfg: RadarConfig =
        serde_json::from_str(&data).with_context(|| format!("nieprawidłowy JSON w {}", path.display()))?;
    cfg.migrate_mono_level();
    cfg.migrate_layer_side_colors();
    Ok(cfg)
}

// ── Profile: nazwane kopie całej konfiguracji ────────────────────────────────

/// Podkatalog `config/` z profilami. Każdy profil to pełna konfiguracja (jak
/// `radar_config.json`) w pliku `<nazwa>.json`. Profil nie wczytuje się przy
/// starcie: przy starcie zawsze obowiązuje `radar_config.json`.
const PROFILE_DIR_NAME: &str = "profile";

/// Katalog z profilami (`config/profile/`).
pub fn profile_dir() -> PathBuf {
    config_dir().join(PROFILE_DIR_NAME)
}

/// Nazwy profili (bez `.json`) z `config/profile/`, alfabetycznie. Brak
/// katalogu to po prostu brak profili.
pub fn list_profiles() -> Vec<String> {
    list_profiles_in(&profile_dir())
}

/// Jak [`list_profiles`], dla podanego katalogu.
pub fn list_profiles_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("json")))
        .filter_map(|path| path.file_stem().and_then(|stem| stem.to_str()).map(str::to_owned))
        .collect();
    names.sort_by_key(|name| name.to_lowercase());
    names
}

/// Nazwy zarezerwowane przez Windows (nie da się utworzyć takiego pliku).
const RESERVED_NAMES: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1",
    "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Nazwa profilu nadająca się na nazwę pliku: bez końcówki `.json`, znaki
/// niedozwolone w nazwach plików (`\ / : * ? " < > |` i sterujące) zamienione
/// na `_`, bez spacji i kropek na końcach. Błąd, gdy nic nie zostaje albo
/// nazwa jest zarezerwowana przez Windows.
pub fn sanitize_profile_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    let without_ext = match trimmed.len().checked_sub(5) {
        Some(cut) if trimmed.is_char_boundary(cut) && trimmed[cut..].eq_ignore_ascii_case(".json") => &trimmed[..cut],
        _ => trimmed,
    };
    let cleaned: String = without_ext
        .chars()
        .map(|c| if c.is_control() || r#"\/:*?"<>|"#.contains(c) { '_' } else { c })
        .collect();
    let cleaned = cleaned.trim_matches(|c: char| c == '.' || c.is_whitespace()).to_owned();
    if cleaned.is_empty() {
        anyhow::bail!("podaj nazwę profilu");
    }
    if RESERVED_NAMES.iter().any(|reserved| reserved.eq_ignore_ascii_case(&cleaned)) {
        anyhow::bail!("„{cleaned}” to nazwa zarezerwowana przez Windows, wybierz inną");
    }
    Ok(cleaned)
}

/// Ścieżka pliku profilu o nazwie `name` (po [`sanitize_profile_name`]).
pub fn profile_path(name: &str) -> Result<PathBuf> {
    profile_path_in(&profile_dir(), name)
}

/// Jak [`profile_path`], dla podanego katalogu.
pub fn profile_path_in(dir: &Path, name: &str) -> Result<PathBuf> {
    Ok(dir.join(format!("{}.json", sanitize_profile_name(name)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plik tymczasowy o nazwie zależnej od testu i procesu. Testy zapisu nie
    /// mogą używać `config/radar_config.json`, bo to dane użytkownika.
    fn temp_config_path(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hearradar_{test_name}_{}.json", std::process::id()))
    }

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
    fn save_then_load_round_trips_on_disk() {
        let path = temp_config_path("round_trip");
        let cfg = RadarConfig {
            radius: 77.0,
            high_contrast: true,
            ..RadarConfig::default()
        };
        save_to(&path, &cfg).expect("save should succeed");
        assert!(path.exists());

        let loaded = load_from(&path).expect("load should succeed right after save");
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
        // `ui_color_profile_picker`, co nadpisuje `color_rules`, a po kliknięciu
        // „Zapisz” trafia na dysk, więc nowy profil musi przetrwać
        // zapis/wczytanie identycznie jak reszta konfiguracji.
        let path = temp_config_path("color_profile");
        let profile = crate::sound::color_profiles::built_in_profiles()
            .first()
            .expect("at least one built-in profile should exist");

        let cfg = RadarConfig {
            color_rules: (profile.build)(),
            ..RadarConfig::default()
        };
        assert!(!cfg.color_rules.is_empty(), "profile should produce at least one rule");

        save_to(&path, &cfg).expect("save should succeed");
        let loaded = load_from(&path).expect("load should succeed right after save");
        assert_eq!(cfg.color_rules, loaded.color_rules);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loading_an_old_config_lowers_mono_boost_once() {
        // Plik sprzed skali stereo w buforze mono (bez `mono_level_version`), jak
        // `config/radar_config.json` użytkownika z +22 dB.
        let path = temp_config_path("mono_level");
        let mut value =
            serde_json::to_value(RadarConfig { mono_boost_db: 22.0, ..RadarConfig::default() }).unwrap();
        value.as_object_mut().unwrap().remove("mono_level_version");
        std::fs::write(&path, value.to_string()).unwrap();

        let loaded = load_from(&path).expect("stary plik powinien się wczytać");
        assert!((loaded.mono_boost_db - (22.0 - 12.04)).abs() < 0.01, "boost po migracji: {}", loaded.mono_boost_db);

        // Po zapisie plik ma już bieżącą wersję, więc kolejne wczytanie niczego nie odejmuje.
        save_to(&path, &loaded).expect("save should succeed");
        let again = load_from(&path).expect("load should succeed right after save");
        assert_eq!(again.mono_boost_db, loaded.mono_boost_db);

        let _ = std::fs::remove_file(&path);
    }

    /// Pola kolorów stron warstw, których nie ma w pliku sprzed tej funkcji.
    const SIDE_COLOR_KEYS: [&str; 6] = [
        "layer_shots_left_color",
        "layer_shots_right_color",
        "layer_steps_left_color",
        "layer_steps_right_color",
        "layer_rest_left_color",
        "layer_rest_right_color",
    ];

    #[test]
    fn loading_a_config_from_before_side_colors_keeps_layer_colors_on_both_sides() {
        use crate::sound::devices::VisualizationMode;
        use crate::sound::layers::Layer;
        // Stary plik: własne kolory warstw, w głównych polach i w wyglądzie trybu, bez pól stron i wersji.
        let path = temp_config_path("layer_side_colors");
        let mut old = RadarConfig {
            layer_shots_color: [10, 20, 30],
            layer_steps_color: [40, 50, 60],
            layer_rest_color: [70, 80, 90],
            ..RadarConfig::default()
        };
        let mut appearance = old.appearance();
        appearance.layer_steps_color = [1, 2, 3];
        old.mode_appearance.insert(VisualizationMode::LayerBars22, appearance);
        let mut value = serde_json::to_value(&old).unwrap();
        let object = value.as_object_mut().unwrap();
        assert!(object.remove("layer_colors_version").is_some());
        for key in SIDE_COLOR_KEYS {
            assert!(object.remove(key).is_some(), "{key}");
            assert!(object["mode_appearance"]["LayerBars22"].as_object_mut().unwrap().remove(key).is_some(), "{key}");
        }
        std::fs::write(&path, value.to_string()).unwrap();

        let loaded = load_from(&path).expect("stary plik powinien się wczytać");
        assert_eq!(loaded.layer_side_colors(Layer::Shots), [[10, 20, 30]; 2]);
        assert_eq!(loaded.layer_side_colors(Layer::Steps), [[40, 50, 60]; 2]);
        assert_eq!(loaded.layer_side_colors(Layer::Rest), [[70, 80, 90]; 2]);
        let saved = loaded.mode_appearance[&VisualizationMode::LayerBars22];
        assert_eq!([saved.layer_steps_left_color, saved.layer_steps_right_color], [[1, 2, 3]; 2]);
        assert_eq!([saved.layer_shots_left_color, saved.layer_shots_right_color], [[10, 20, 30]; 2]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn side_colors_survive_save_and_load() {
        use crate::sound::layers::Layer;
        // Wybór z UI: wszystko po lewej niebieskie, po prawej fioletowe. Ma przetrwać restart.
        let path = temp_config_path("side_colors_round_trip");
        let mut cfg = RadarConfig::default();
        cfg.set_layer_side_color_for_all(0, [0, 150, 255]);
        cfg.set_layer_side_color_for_all(1, [180, 40, 200]);
        cfg.remember_appearance();
        save_to(&path, &cfg).expect("save should succeed");

        let loaded = load_from(&path).expect("load should succeed right after save");
        assert_eq!(loaded, cfg);
        for layer in Layer::ALL {
            assert_eq!(loaded.layer_side_colors(layer), [[0, 150, 255], [180, 40, 200]], "{layer:?}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn profile_names_become_safe_file_names() {
        assert_eq!(sanitize_profile_name("  grzebień 13 ms ").unwrap(), "grzebień 13 ms");
        assert_eq!(sanitize_profile_name("kroki.JSON").unwrap(), "kroki");
        assert_eq!(sanitize_profile_name("a/b\\c:d*e?f\"g<h>i|j").unwrap(), "a_b_c_d_e_f_g_h_i_j");
        assert_eq!(sanitize_profile_name("koniec. ").unwrap(), "koniec");
        assert!(sanitize_profile_name("   ").is_err());
        assert!(sanitize_profile_name(".json").is_err());
        assert!(sanitize_profile_name("nul").is_err());
        assert!(sanitize_profile_name("ż.json").is_ok(), "końcówka po wielobajtowym znaku");
    }

    #[test]
    fn profiles_are_listed_by_name_and_round_trip() {
        let dir = std::env::temp_dir().join(format!("hearradar_profiles_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(list_profiles_in(&dir).is_empty(), "brak katalogu = brak profili");

        let combo = RadarConfig { mono_cue_style: crate::sound::devices::MonoCueStyle::DirectionalComb, ..RadarConfig::default() };
        save_to(&profile_path_in(&dir, "Zestaw b").unwrap(), &combo).unwrap();
        save_to(&profile_path_in(&dir, "a.json").unwrap(), &RadarConfig::default()).unwrap();
        std::fs::write(dir.join("notatka.txt"), "nie profil").unwrap();

        assert_eq!(list_profiles_in(&dir), ["a", "Zestaw b"]);
        assert_eq!(load_from(&profile_path_in(&dir, "Zestaw b").unwrap()).unwrap(), combo);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profiles_live_next_to_the_main_config() {
        assert_eq!(profile_dir().parent().unwrap(), config_dir());
        assert_eq!(profile_path("x").unwrap(), profile_dir().join("x.json"));
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
