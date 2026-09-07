//! Uruchamia `ml/info.py` (osobny proces Pythona) i parsuje jego JSON, żeby
//! pokazać w UI diagnostykę środowiska ML — czy Python/TensorFlow/
//! onnxruntime są zainstalowane i czy model (`assets/ml/yamnet.onnx`) jest
//! już wyeksportowany. To JEDYNE miejsce w aplikacji, które odpala proces
//! Pythona w runtime — poza tym Python jest używany wyłącznie offline do
//! eksportu modelu (patrz `ml/README.md`); sama klasyfikacja dźwięku
//! (`sound::ml_classifier`) działa w Rust przez ONNX Runtime i nie wymaga
//! Pythona.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::process::Command;

/// Wynik `ml/info.py` — patrz ten plik po dokładny format JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct PythonEnvInfo {
    pub python_version: String,
    pub tensorflow_installed: bool,
    pub tensorflow_version: Option<String>,
    pub onnxruntime_installed: bool,
    pub onnxruntime_version: Option<String>,
    pub model_exists: bool,
    pub model_size_mb: Option<f64>,
    pub mapping_exists: bool,
    pub category_count: Option<usize>,
    /// Kategoria wykryta przez `ml/info.py` po faktycznym uruchomieniu
    /// klasyfikacji (przez onnxruntime, na wyeksportowanym modelu) na
    /// testowym tonie [`TEST_FREQ_HZ`] — `None`, jeśli onnxruntime/model są
    /// niedostępne (patrz `test_error` wtedy albo brak `model_exists`).
    /// `#[serde(default)]`: starsze `ml/info.py` sprzed tej funkcji nie
    /// zwracały tych pól wcale.
    #[serde(default)]
    pub test_category: Option<String>,
    #[serde(default)]
    pub test_confidence: Option<f32>,
    /// Komunikat błędu, jeśli klasyfikacja testowa się nie powiodła mimo że
    /// onnxruntime i model są dostępne (np. niezgodna liczba klas).
    #[serde(default)]
    pub test_error: Option<String>,
}

/// Ten sam sygnał testowy (częstotliwość, czas trwania, częstotliwość
/// próbkowania), co generuje `ml/info.py` (`TEST_FREQ_HZ`/`TEST_SAMPLE_RATE`)
/// — dzięki temu wynik Rusta i Pythona dla "tego samego dźwięku" da się
/// bezpośrednio porównać w UI (patrz `ui_python_info_test` w `src/ui/app.rs`).
pub const TEST_FREQ_HZ: f32 = 1000.0;
pub const TEST_SAMPLE_RATE: u32 = 16_000;
const TEST_AMPLITUDE: f32 = 0.5;

/// Generuje ten sam ton testowy co `ml/info.py::generate_test_waveform()`.
pub fn generate_test_waveform() -> Vec<f32> {
    (0..TEST_SAMPLE_RATE)
        .map(|i| TEST_AMPLITUDE * (std::f32::consts::TAU * TEST_FREQ_HZ * (i as f32 / TEST_SAMPLE_RATE as f32)).sin())
        .collect()
}

fn ml_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ml")
}

/// Preferuje interpreter z `ml/.venv` (ma zainstalowane zależności z
/// `ml/requirements.txt`, patrz `ml/bootstrap.sh`/`ml/bootstrap.ps1`) —
/// spada na systemowy interpreter w PATH, jeśli venv jeszcze nie istnieje
/// (np. nikt nie uruchamiał jeszcze cold startu). W tym drugim przypadku
/// `info.py` po prostu zgłosi TensorFlow/onnxruntime jako niezainstalowane,
/// zamiast się wywalić. Układ katalogu venv różni się między platformami
/// (`bin/python3` na macOS/Linux, `Scripts\python.exe` na Windows — patrz
/// `venv` w dokumentacji Pythona), więc sprawdzamy oba warianty.
fn python_executable() -> PathBuf {
    let venv_dir = ml_dir().join(".venv");
    let venv_python_unix = venv_dir.join("bin").join("python3");
    let venv_python_windows = venv_dir.join("Scripts").join("python.exe");

    if venv_python_unix.exists() {
        venv_python_unix
    } else if venv_python_windows.exists() {
        venv_python_windows
    } else if cfg!(windows) {
        // Windows zwykle instaluje tylko `python`, nie `python3` w PATH.
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

/// Uruchamia `ml/info.py` i parsuje jego JSON. Zwraca czytelny błąd (np.
/// brak Pythona w PATH, albo skrypt zakończony niezerowym kodem) zamiast
/// panikować — to tylko diagnostyka na żądanie użytkownika, nie coś
/// krytycznego dla działania reszty aplikacji.
pub fn fetch_python_env_info() -> Result<PythonEnvInfo> {
    let script = ml_dir().join("info.py");
    anyhow::ensure!(script.exists(), "brak {} — repo jest niekompletne?", script.display());

    let output = Command::new(python_executable())
        .arg(&script)
        .output()
        .context("nie udało się uruchomić Pythona (sprawdź, czy `python3` jest w PATH)")?;

    anyhow::ensure!(
        output.status.success(),
        "ml/info.py zakończyło się błędem:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).context("nieprawidłowy JSON z ml/info.py")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_info_json() {
        let json = r#"{
            "python_version": "3.11.4",
            "tensorflow_installed": true,
            "tensorflow_version": "2.16.2",
            "onnxruntime_installed": true,
            "onnxruntime_version": "1.19.2",
            "model_exists": true,
            "model_size_mb": 16.1,
            "mapping_exists": true,
            "category_count": 8
        }"#;
        let info: PythonEnvInfo = serde_json::from_str(json).expect("powinno się sparsować");
        assert_eq!(info.python_version, "3.11.4");
        assert!(info.tensorflow_installed);
        assert_eq!(info.tensorflow_version.as_deref(), Some("2.16.2"));
        assert_eq!(info.category_count, Some(8));
    }

    #[test]
    fn parses_missing_model_as_nulls_not_errors() {
        let json = r#"{
            "python_version": "3.9.6",
            "tensorflow_installed": false,
            "tensorflow_version": null,
            "onnxruntime_installed": false,
            "onnxruntime_version": null,
            "model_exists": false,
            "model_size_mb": null,
            "mapping_exists": false,
            "category_count": null
        }"#;
        let info: PythonEnvInfo = serde_json::from_str(json).expect("powinno się sparsować");
        assert!(!info.model_exists);
        assert!(info.model_size_mb.is_none());
    }

    #[test]
    fn parses_test_classification_fields() {
        let json = r#"{
            "python_version": "3.9.6",
            "tensorflow_installed": true,
            "tensorflow_version": "2.16.2",
            "onnxruntime_installed": true,
            "onnxruntime_version": "1.19.2",
            "model_exists": true,
            "model_size_mb": 16.1,
            "mapping_exists": true,
            "category_count": 8,
            "test_category": "Inne",
            "test_confidence": 0.8798195,
            "test_error": null
        }"#;
        let info: PythonEnvInfo = serde_json::from_str(json).expect("powinno się sparsować");
        assert_eq!(info.test_category.as_deref(), Some("Inne"));
        assert!(info.test_confidence.unwrap() > 0.8);
        assert!(info.test_error.is_none());
    }

    #[test]
    fn old_info_json_without_test_fields_still_parses() {
        // Starsze `ml/info.py` (sprzed dodania self-testu) nie zwracało tych
        // pól wcale — `#[serde(default)]` musi na to pozwolić.
        let json = r#"{
            "python_version": "3.9.6",
            "tensorflow_installed": false,
            "tensorflow_version": null,
            "onnxruntime_installed": false,
            "onnxruntime_version": null,
            "model_exists": false,
            "model_size_mb": null,
            "mapping_exists": false,
            "category_count": null
        }"#;
        let info: PythonEnvInfo = serde_json::from_str(json).expect("powinno się sparsować mimo brakujących pól");
        assert!(info.test_category.is_none());
    }

    #[test]
    fn generate_test_waveform_has_expected_length_and_range() {
        let waveform = generate_test_waveform();
        assert_eq!(waveform.len(), TEST_SAMPLE_RATE as usize);
        for &s in &waveform {
            assert!(s.abs() <= TEST_AMPLITUDE + 1e-6, "próbka {s} poza oczekiwaną amplitudą");
        }
    }

    #[test]
    fn fetch_returns_readable_error_or_valid_info() {
        // Nie zakładamy, że Python/venv są dostępne w środowisku testowym —
        // sprawdzamy tylko, że funkcja nigdy nie panikuje i zawsze zwraca
        // albo sensowną strukturę, albo czytelny błąd.
        match fetch_python_env_info() {
            Ok(info) => assert!(!info.python_version.is_empty()),
            Err(e) => assert!(!e.to_string().is_empty()),
        }
    }

    /// Sprawdza, że Python (onnxruntime, przez `ml/info.py`) i Rust (`ort`,
    /// przez `SoundClassifier`) dają TĘ SAMĄ kategorię dla tego samego
    /// testowego tonu — to jest właściwy sens tego testu diagnostycznego w
    /// UI, nie tylko sprawdzenie że pliki istnieją. `#[ignore]`, bo wymaga
    /// wyeksportowanego modelu (`./ml/bootstrap.sh`) i onnxruntime w `ml/.venv`.
    #[test]
    #[ignore]
    fn python_and_rust_agree_on_test_tone_category() {
        use crate::sound::ml_classifier::SoundClassifier;

        let info = fetch_python_env_info().expect("uruchom ./ml/bootstrap.sh przed tym testem");
        let py_category = info.test_category.clone().unwrap_or_else(|| {
            panic!("ml/info.py nie zwróciło test_category (błąd: {:?})", info.test_error)
        });

        let mut classifier =
            SoundClassifier::load(&SoundClassifier::default_model_dir()).expect("model powinien być wyeksportowany");
        let waveform = generate_test_waveform();
        let results = classifier.classify(&waveform).expect("inferencja nie powinna zwrócić błędu");
        let rust_category = &results.first().expect("oczekiwano co najmniej jednego wyniku").category;

        assert_eq!(
            &py_category, rust_category,
            "Python i Rust powinny się zgadzać co do kategorii dla tego samego sygnału"
        );
        println!("Zgodność potwierdzona: obie strony dają kategorię \"{py_category}\".");
    }
}
