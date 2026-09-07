//! Klasyfikacja dźwięku (ML) — inferencja ONNX na modelu YAMNet
//! wyeksportowanym narzędziami z `ml/` (patrz `ml/README.md`).
//!
//! To osobna, EKSPERYMENTALNA warstwa nad istniejącym radarem pasmowym
//! (`spectrum.rs`), nie jego zamiennik: działa na wolniejszych oknach ~1 s
//! (patrz [`WINDOW_SECS`]), podczas gdy radar rysuje kropki co 20-40 ms z
//! prostej energii pasma. Model NIE jest wymagany do działania reszty
//! aplikacji — [`SoundClassifier::load`] zwraca czytelny błąd zamiast
//! panikować, gdy plików brak, a wywołujący (`devices.rs`) po prostu nie
//! uruchamia wątku klasyfikatora.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::sound::devices::SharedAudioAnalysis;
use crate::sound::direction;

/// YAMNet oczekuje mono, 16 kHz — mel-spektrogram jest częścią jego grafu
/// (patrz `ml/export_yamnet_onnx.py`), więc Rust karmi go surowym PCM.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Długość okna klasyfikacji w sekundach — jeden przebieg inferencji na
/// tyle próbek naraz. YAMNet dzieli to wewnętrznie na klatki ~0.96 s;
/// krótsze okno daje mu za mało kontekstu do sensownej klasyfikacji.
pub const WINDOW_SECS: f32 = 1.0;

/// Zawartość `category_mapping.json` wygenerowanego przez
/// `ml/export_mapping.py` — patrz ten plik po dokładny format.
#[derive(Debug, Deserialize)]
struct CategoryMappingFile {
    /// `categories[i]` = kategoria dla klasy AudioSet o indeksie `i`
    /// (521 wpisów, w kolejności zgodnej z wyjściem modelu).
    categories: Vec<String>,
    /// Wszystkie możliwe wartości występujące w `categories`, w stałej
    /// kolejności — używane do zbudowania domyślnej palety kolorów w UI.
    unique_categories: Vec<String>,
}

/// Wynik klasyfikacji jednego okna audio dla jednej kategorii — już
/// zsumowany z wielu klas AudioSet wg mapowania, nie surowy wynik modelu.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub category: String,
    pub confidence: f32,
}

/// Najsilniejsza kategoria z ostatniego okna klasyfikacji, wraz z kierunkiem
/// (jeśli dało się go policzyć — patrz [`crate::sound::direction::compute_direction`])
/// i chwilą powstania (do liczenia zaniku "odznaki" w UI, patrz `radar_egui.rs`).
/// `Instant`, nie znacznik czasu egui — wątek klasyfikatora działa poza
/// pętlą UI i nie ma dostępu do zegara `egui::Context`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationEvent {
    pub category: String,
    pub confidence: f32,
    pub angle: Option<f32>,
    pub spawned_at: Instant,
}

/// Domyślne kolory dla kategorii z `ml/class_mapping.py::all_categories()`.
/// MUSZĄ używać dokładnie tych samych polskich nazw co Python — dopasowanie
/// kolor↔kategoria w UI odbywa się przez porównanie stringów z
/// `category_mapping.json`. Zmieniasz nazwy kategorii w Pythonie? Zaktualizuj
/// też tę listę.
pub fn default_category_colors() -> HashMap<String, [u8; 3]> {
    [
        ("Strzały/wybuchy", [255, 90, 90]),
        ("Kroki", [80, 220, 90]),
        ("Mowa", [0, 200, 220]),
        ("Pojazdy", [180, 40, 200]),
        ("Mechaniczne/przeładowanie", [255, 200, 60]),
        ("Muzyka/otoczenie", [120, 160, 255]),
        ("Cisza", [140, 140, 140]),
        ("Inne", [255, 255, 255]),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// Miksuje przeplecione próbki multi-channel do mono (średnia kanałów per
/// klatka). Dla `channels <= 1` zwraca próbki bez zmian.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Prosty resampler liniowy — wystarczający do klasyfikacji (nie do
/// odsłuchu). Nie filtruje aliasingu przy downsamplingu, co dla mowy/efektów
/// dźwiękowych z gry jest akceptowalnym kompromisem na pierwszą wersję (patrz
/// `ml/README.md`) — dokładniejszy resampler (np. `rubato`) to potencjalne
/// ulepszenie, jeśli jakość klasyfikacji tego wymaga.
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let last = input.len() - 1;

    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = (src_pos.floor() as usize).min(last);
            let frac = (src_pos - idx as f64) as f32;
            let a = input[idx];
            let b = input[(idx + 1).min(last)];
            a + (b - a) * frac
        })
        .collect()
}

pub struct SoundClassifier {
    session: Session,
    categories: Vec<String>,
    pub unique_categories: Vec<String>,
}

impl SoundClassifier {
    /// Domyślna lokalizacja plików modelu: `assets/ml/` w repo (patrz
    /// `ml/bootstrap.sh`) — ścieżka wypalana w binarkę w czasie kompilacji,
    /// tak samo jak w `crate::config_store::config_dir`.
    pub fn default_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets").join("ml")
    }

    /// Wczytuje model (`yamnet.onnx`) i mapowanie klas
    /// (`category_mapping.json`) z podanego katalogu. Zwraca czytelny błąd
    /// zamiast panikować, gdy plików brak — pozwala to wywołującemu po
    /// prostu nie uruchamiać klasyfikacji zamiast crashować całą aplikację.
    pub fn load(dir: &Path) -> Result<Self> {
        let model_path = dir.join("yamnet.onnx");
        let mapping_path = dir.join("category_mapping.json");

        if !model_path.exists() || !mapping_path.exists() {
            anyhow::bail!(
                "Brak plików modelu w {} — uruchom `./ml/bootstrap.sh`, żeby włączyć klasyfikację dźwięku.",
                dir.display()
            );
        }

        let session = Session::builder()
            .context("nie udało się utworzyć sesji ONNX Runtime")?
            .commit_from_file(&model_path)
            .with_context(|| format!("nie udało się wczytać modelu {}", model_path.display()))?;

        let mapping_json = std::fs::read_to_string(&mapping_path)
            .with_context(|| format!("nie udało się odczytać {}", mapping_path.display()))?;
        let mapping: CategoryMappingFile = serde_json::from_str(&mapping_json)
            .with_context(|| format!("nieprawidłowy JSON w {}", mapping_path.display()))?;

        Ok(Self {
            session,
            categories: mapping.categories,
            unique_categories: mapping.unique_categories,
        })
    }

    /// Uruchamia jeden przebieg inferencji na mono próbkach (16 kHz,
    /// znormalizowane -1.0..=1.0). Zwraca wyniki per kategoria — dla każdej
    /// kategorii bierzemy NAJSILNIEJSZĄ z należących do niej klas AudioSet
    /// (uśrednioną po klatkach czasowych modelu), posortowane malejąco.
    pub fn classify(&mut self, mono_16k: &[f32]) -> Result<Vec<ClassificationResult>> {
        let tensor = Tensor::from_array(([mono_16k.len()], mono_16k.to_vec()))
            .context("nie udało się zbudować tensora wejściowego")?;

        // Wejście podajemy pozycyjnie (nie po nazwie "waveform") — model ma
        // dokładnie jedno wejście, więc kolejność wystarcza, a nazwa węzła
        // wejściowego w wyeksportowanym grafie bywa inna niż w kodzie
        // Pythona (zależy dokładnie, jak `tf2onnx` zamrozi/nazwie graf —
        // patrz `ml/export_yamnet_onnx.py`). Pozycyjne wiązanie nie zależy
        // od tego szczegółu.
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("inferencja ONNX nie powiodła się")?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("nieoczekiwany kształt wyjścia modelu")?;

        // Kształt wyjścia YAMNet to [liczba_klatek, 521] — uśredniamy po
        // klatkach czasowych, żeby dostać jedną wartość na klasę dla całego
        // okna zamiast osobnego wyniku per klatka ~0.96 s.
        let num_classes = self.categories.len();
        if num_classes == 0 {
            return Ok(Vec::new());
        }
        let num_frames = data.len() / num_classes;
        anyhow::ensure!(
            num_frames * num_classes == data.len(),
            "wyjście modelu ({} wartości) nie dzieli się równo przez liczbę klas ({}); shape={:?}",
            data.len(),
            num_classes,
            shape
        );

        let mut per_class_avg = vec![0.0f32; num_classes];
        for frame in 0..num_frames {
            for class in 0..num_classes {
                per_class_avg[class] += data[frame * num_classes + class];
            }
        }
        if num_frames > 0 {
            for v in &mut per_class_avg {
                *v /= num_frames as f32;
            }
        }

        let mut per_category: HashMap<&str, f32> = HashMap::new();
        for (class_idx, &score) in per_class_avg.iter().enumerate() {
            if let Some(category) = self.categories.get(class_idx) {
                let entry = per_category.entry(category.as_str()).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
        }

        let mut results: Vec<ClassificationResult> = per_category
            .into_iter()
            .map(|(category, confidence)| ClassificationResult { category: category.to_string(), confidence })
            .collect();
        results.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(results)
    }
}

/// Jak często wątek klasyfikatora budzi się, żeby sprawdzić bufor i
/// ewentualnie sklasyfikować nowe okno audio. Osobna, wolniejsza pętla od
/// callbacku audio (10-20 ms) i od rysowania kropek radaru (20-40 ms) — to
/// zamierzone, klasyfikacja "sytuacji" nie musi (i nie powinna) migać co
/// klatkę.
const CLASSIFY_INTERVAL: Duration = Duration::from_millis(500);

/// Uchwyt wątku klasyfikatora — `Drop` sygnalizuje wątkowi zatrzymanie się
/// (sprawdzane raz na [`CLASSIFY_INTERVAL`], więc zatrzymanie nie jest
/// natychmiastowe, ale nie blokuje wątku UI przy zamykaniu/przełączaniu).
pub struct MlClassifierHandle {
    running: Arc<AtomicBool>,
}

impl Drop for MlClassifierHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Wczytuje model z [`SoundClassifier::default_model_dir`] i — jeśli się uda
/// — uruchamia wątek w tle, który co [`CLASSIFY_INTERVAL`] czyta najnowsze
/// audio z `analysis.raw_audio_buffer` (patrz `devices.rs::update_analysis`),
/// miksuje je do mono 16 kHz i klasyfikuje, zapisując wynik z powrotem do
/// `analysis.latest_classification`.
///
/// Zwraca `Err`, jeśli modelu nie ma — wywołujący (UI) pokazuje wtedy
/// komunikat błędu zamiast cichego "nic się nie dzieje".
pub fn start_ml_classifier_thread(analysis: SharedAudioAnalysis) -> Result<MlClassifierHandle> {
    let mut classifier = SoundClassifier::load(&SoundClassifier::default_model_dir())?;

    // Jeśli ktoś edytował kategorie w `ml/class_mapping.py` i wyeksportował
    // nowy `category_mapping.json`, mogą się pojawić kategorie, dla których
    // `RadarConfig.category_colors` (wczytany z `config/radar_config.json`
    // sprzed tej zmiany) nie ma jeszcze koloru — dogrywamy brakujące jako
    // neutralny szary, zamiast po cichu nie rysować dla nich odznaki.
    if let Ok(mut guard) = analysis.lock() {
        for category in &classifier.unique_categories {
            guard.radar_config.category_colors.entry(category.clone()).or_insert([160, 160, 160]);
        }
    }

    let running = Arc::new(AtomicBool::new(true));
    let running_thread = running.clone();

    std::thread::spawn(move || {
        while running_thread.load(Ordering::Relaxed) {
            std::thread::sleep(CLASSIFY_INTERVAL);

            let Ok(guard) = analysis.lock() else { continue };
            if !guard.radar_config.ml_enabled {
                continue;
            }
            let channels = guard.raw_audio_channels;
            let sample_rate = guard.raw_audio_sample_rate;
            // Bufor trzyma trochę więcej niż jedno okno analizy (patrz
            // `devices::RAW_AUDIO_BUFFER_SECS`) — bierzemy tylko najnowsze
            // `WINDOW_SECS`, żeby klasyfikować świeże audio, nie całą historię.
            let want_frames = (sample_rate as f32 * WINDOW_SECS) as usize * channels.max(1);
            let total = guard.raw_audio_buffer.len();
            let raw: Vec<f32> = guard
                .raw_audio_buffer
                .iter()
                .skip(total.saturating_sub(want_frames))
                .copied()
                .collect();
            drop(guard);

            if raw.is_empty() || channels == 0 || sample_rate == 0 {
                continue;
            }

            let mono = downmix_to_mono(&raw, channels);
            let mono_16k = resample_linear(&mono, sample_rate, TARGET_SAMPLE_RATE);
            if mono_16k.is_empty() {
                continue;
            }

            // Kierunek liczymy z tego samego okna, tym samym mechanizmem co
            // żywy radar pasmowy — spójność z resztą aplikacji zamiast
            // osobnej logiki kierunkowej tylko dla ML.
            let stats = direction::analyze_channels(&raw, channels);
            let (angle, intensity) = direction::compute_direction(&stats.rms);
            let angle = (intensity > 0.01).then_some(angle);

            match classifier.classify(&mono_16k) {
                Ok(results) => {
                    if let Some(top) = results.first() {
                        if let Ok(mut lock) = analysis.lock() {
                            lock.latest_classification = Some(ClassificationEvent {
                                category: top.category.clone(),
                                confidence: top.confidence,
                                angle,
                                spawned_at: Instant::now(),
                            });
                        }
                    }
                }
                Err(e) => eprintln!("[ml_classifier] inferencja nieudana: {e}"),
            }
        }
    });

    Ok(MlClassifierHandle { running })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ładuje PRAWDZIWY model z `assets/ml/` (jeśli wygenerowany przez
    /// `./ml/bootstrap.sh`) i klasyfikuje syntetyczny ton — sprawdza, że
    /// integracja z `ort` faktycznie działa na wyeksportowanym pliku, nie
    /// tylko że kod się kompiluje. `#[ignore]`, bo `assets/ml/` jest w
    /// `.gitignore` i nie ma go w świeżym klonie repo — uruchom ręcznie po
    /// `./ml/bootstrap.sh` przez `cargo test -- --ignored ml_classifier`.
    #[test]
    #[ignore]
    fn classify_runs_on_real_exported_model() {
        let dir = SoundClassifier::default_model_dir();
        let mut classifier = SoundClassifier::load(&dir).expect("uruchom ./ml/bootstrap.sh przed tym testem");

        let sample_rate = TARGET_SAMPLE_RATE as f32;
        let mono_16k: Vec<f32> = (0..sample_rate as usize)
            .map(|i| (i as f32 / sample_rate * std::f32::consts::TAU * 440.0).sin() * 0.5)
            .collect();

        let results = classifier.classify(&mono_16k).expect("inferencja nie powinna zwrócić błędu");
        assert!(!results.is_empty(), "oczekiwano co najmniej jednego wyniku klasyfikacji");
        assert!(results[0].confidence >= results.last().unwrap().confidence, "wyniki powinny być posortowane malejąco");
        println!("Top wynik dla tonu 440 Hz: {} ({:.1}%)", results[0].category, results[0].confidence * 100.0);
    }

    #[test]
    fn load_returns_readable_error_when_model_files_are_missing() {
        let dir = std::env::temp_dir().join("hearradar_ml_test_nonexistent");
        match SoundClassifier::load(&dir) {
            Ok(_) => panic!("oczekiwano błędu przy braku plików modelu"),
            Err(err) => assert!(
                err.to_string().contains("bootstrap.sh"),
                "błąd powinien wskazywać jak naprawić brak modelu, otrzymano: {err}"
            ),
        }
    }

    #[test]
    fn default_model_dir_points_inside_repo_assets() {
        let dir = SoundClassifier::default_model_dir();
        assert!(dir.ends_with("assets/ml") || dir.ends_with("assets\\ml"));
        assert!(dir.starts_with(env!("CARGO_MANIFEST_DIR")));
    }

    // ---------- downmix_to_mono ----------

    #[test]
    fn downmix_averages_channels_per_frame() {
        // 2 kanały, 2 klatki: (1.0, 3.0) -> 2.0, (0.0, 0.0) -> 0.0
        let interleaved = [1.0, 3.0, 0.0, 0.0];
        assert_eq!(downmix_to_mono(&interleaved, 2), vec![2.0, 0.0]);
    }

    #[test]
    fn downmix_is_noop_for_mono() {
        let samples = [0.1, 0.2, 0.3];
        assert_eq!(downmix_to_mono(&samples, 1), samples.to_vec());
    }

    // ---------- resample_linear ----------

    #[test]
    fn resample_same_rate_is_identity() {
        let input = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input.to_vec());
    }

    #[test]
    fn resample_halves_length_when_downsampling_by_half() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let out = resample_linear(&input, 32_000, 16_000);
        assert_eq!(out.len(), 500);
    }

    #[test]
    fn resample_empty_input_is_empty_output() {
        assert!(resample_linear(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn resample_preserves_a_constant_signal() {
        let input = vec![0.5f32; 100];
        let out = resample_linear(&input, 48_000, 16_000);
        assert!(!out.is_empty());
        for v in out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    // ---------- default_category_colors ----------

    #[test]
    fn default_category_colors_cover_every_class_mapping_category() {
        // Muszą się zgadzać z ml/class_mapping.py::all_categories() —
        // inaczej UI nie znajdzie koloru dla jakiejś kategorii.
        let colors = default_category_colors();
        for expected in [
            "Strzały/wybuchy", "Kroki", "Mowa", "Pojazdy",
            "Mechaniczne/przeładowanie", "Muzyka/otoczenie", "Cisza", "Inne",
        ] {
            assert!(colors.contains_key(expected), "brak koloru dla kategorii '{expected}'");
        }
    }
}
