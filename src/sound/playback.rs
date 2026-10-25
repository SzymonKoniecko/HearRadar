//! Odtwarzanie nagrania WAV przez ten sam potok analizy co żywe audio.
//!
//! Próbki idą przez `devices::update_analysis` w tempie czasu rzeczywistego, więc radar
//! (kropki, łuki, słupki warstw) reaguje tak samo jak podczas grania. Wybór pliku jest
//! w panelu „Źródło audio” (patrz `ui::app`).

use crate::sound::clip_analysis::read_wav;
use crate::sound::devices::{self, RadarConfig, SharedAudioAnalysis};
use crate::ui::radar_egui::{layer_text, RadarState, TimedDirectionEvent};
use crate::sound::layers::{interleave, LayerHit};
use anyhow::{bail, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Jak długi fragment próbek idzie naraz do analizy. 10 ms, jak typowy bufor cpal.
const CHUNK_S: f64 = 0.01;

/// Plik, który można odtworzyć, z etykietą do listy w UI.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayableFile {
    pub label: String,
    pub path: PathBuf,
}

/// Odtwarza jeden plik w osobnym wątku. Zatrzymuje się sam na końcu nagrania albo przy `Drop`.
pub struct FilePlayer {
    name: String,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl FilePlayer {
    /// Wczytuje plik od razu (błąd zwracany tutaj, nie w wątku) i startuje odtwarzanie.
    pub fn start(file: &PlayableFile, analysis: SharedAudioAnalysis) -> Result<Self> {
        let (rate, channels) = read_wav(&file.path)?;
        let n = channels.len();
        if n == 0 || channels[0].is_empty() {
            bail!("pusty plik: {}", file.path.display());
        }
        let samples = interleave(&channels);
        let chunk_len = ((rate as f64 * CHUNK_S).round().max(1.0) as usize) * n;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let started = Instant::now();
            for (k, chunk) in samples.chunks(chunk_len).enumerate() {
                if stop_worker.load(Ordering::Relaxed) {
                    break;
                }
                // Pacing po czasie od startu, a nie po samym `sleep`, żeby błędy nie sumowały się w dryf.
                let due = started + Duration::from_secs_f64(k as f64 * CHUNK_S);
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    thread::sleep(wait);
                }
                devices::update_analysis(chunk, n, rate as f32, &analysis);
            }
        });
        Ok(Self {
            name: file.label.clone(),
            stop,
            worker: Some(worker),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `true`, gdy odtwarzanie dobiegło końca (albo wątek już nie działa).
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().map_or(true, JoinHandle::is_finished)
    }
}

impl Drop for FilePlayer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Analiza całego nagrania bez odtwarzania w czasie rzeczywistym. Te same decyzje warstw co na
/// żywo, tylko szybciej. Używa `radar --classify-session`.
pub fn classify_recording(path: &Path) -> Result<Vec<LayerHit>> {
    let (rate, channels) = read_wav(path)?;
    let n = channels.len();
    if n == 0 || channels[0].is_empty() {
        bail!("pusty plik: {}", path.display());
    }
    let mut samples = interleave(&channels);
    // Cisza na końcu, żeby zdążyły się rozstrzygnąć ostatnie zdarzenia (okno szczytu ok. 0,35 s).
    samples.resize(samples.len() + (rate as usize / 2) * n, 0.0);

    let analysis = SharedAudioAnalysis::new();
    let chunk_len = ((rate as f64 * CHUNK_S).round().max(1.0) as usize) * n;
    for chunk in samples.chunks(chunk_len) {
        devices::update_analysis(chunk, n, rate as f32, &analysis);
    }
    let hits = analysis.lock().map(|mut g| g.drain_layer_hits()).unwrap_or_default();
    Ok(hits.into_iter().map(|queued| queued.hit).collect())
}

/// Symulacja radaru 3+3 bez okna: przepuszcza nagranie tym samym potokiem co na żywo (szybki
/// detektor, klasy, stan słupków) i zwraca tekst z tym, co radar pokazałby co `every_s` sekund.
pub fn radar_dump(path: &Path, every_s: f64, steps_gain: f32) -> Result<String> {
    let (rate, channels) = read_wav(path)?;
    let n = channels.len();
    if n == 0 || channels[0].is_empty() {
        bail!("pusty plik: {}", path.display());
    }
    let mut samples = interleave(&channels);
    samples.resize(samples.len() + (rate as usize / 2) * n, 0.0);

    let analysis = SharedAudioAnalysis::new();
    let mut cfg = RadarConfig::default();
    cfg.layer_steps_gain = steps_gain;
    let mut state = RadarState::new();
    let chunk_len = ((rate as f64 * CHUNK_S).round().max(1.0) as usize) * n;
    let mut out = String::new();
    let mut next_dump = 0.0;
    for (k, part) in samples.chunks(chunk_len).enumerate() {
        devices::update_analysis(part, n, rate as f32, &analysis);
        let now = (k + 1) as f64 * CHUNK_S;
        let (rms, energies, direction, layers) = analysis
            .lock()
            .map(|mut g| {
                (
                    g.channel_rms.clone(),
                    g.channel_rule_energies.clone(),
                    g.drain_direction_events(),
                    g.drain_layer_hits(),
                )
            })
            .unwrap_or_default();
        let timed: Vec<TimedDirectionEvent> = direction
            .iter()
            .map(|queued| TimedDirectionEvent { at: now, event: queued.event })
            .collect();
        for queued in &layers {
            state.layer_bars.record(queued.hit.onset_s, queued.hit);
        }
        state.update(
            now,
            n,
            &rms,
            &energies,
            &cfg.color_rules,
            cfg.decay_secs,
            cfg.radar_sensitivity,
            &timed,
        );
        state.layer_bars.prune(now, cfg.decay_secs);
        if now + 1e-9 >= next_dump {
            out.push_str(&format!("== t = {now:.2} s\n{}", layer_text(&state, &cfg, now)));
            next_dump += every_s;
        }
    }
    Ok(out)
}

/// Pliki do odsłuchu z katalogu sesji: `<sesja>/audio.wav` i `<sesja>/clips/*.wav`.
/// Posortowane po etykiecie. Brak katalogu daje pustą listę.
pub fn list_playable(root: &Path) -> Vec<PlayableFile> {
    let mut out = Vec::new();
    let Ok(sessions) = fs::read_dir(root) else {
        return out;
    };
    for session in sessions.flatten() {
        let dir = session.path();
        if !dir.is_dir() {
            continue;
        }
        let name = session.file_name().to_string_lossy().into_owned();

        let audio = dir.join("audio.wav");
        if audio.is_file() {
            out.push(PlayableFile {
                label: format!("{name} / cały zapis"),
                path: audio,
            });
        }

        if let Ok(clips) = fs::read_dir(dir.join("clips")) {
            for clip in clips.flatten().map(|entry| entry.path()) {
                let is_wav = clip.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
                if !is_wav {
                    continue;
                }
                let clip_name = clip.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                out.push(PlayableFile {
                    label: format!("{name} / klip {clip_name}"),
                    path: clip,
                });
            }
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hearradar_playback_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn lists_full_recordings_and_clips_of_each_session() {
        let root = temp_root("list");
        let session = root.join("sesja_a");
        fs::create_dir_all(session.join("clips")).unwrap();
        fs::write(session.join("audio.wav"), b"x").unwrap();
        fs::write(session.join("clips").join("E001.wav"), b"x").unwrap();
        fs::write(session.join("clips").join("notatki.txt"), b"x").unwrap();
        fs::create_dir_all(root.join("pusta")).unwrap();

        let files = list_playable(&root);
        let _ = fs::remove_dir_all(&root);

        let labels: Vec<&str> = files.iter().map(|f| f.label.as_str()).collect();
        assert_eq!(labels, vec!["sesja_a / cały zapis", "sesja_a / klip E001.wav"]);
    }

    #[test]
    fn missing_folder_gives_an_empty_list() {
        assert!(list_playable(Path::new("nie/ma/takiego/katalogu")).is_empty());
    }

    /// Cisza, potem głośny szum przez 0,1 s zaczynający się w 0,3 s (1 s całości w 48 kHz).
    fn click_samples() -> Vec<f32> {
        let mut state = 0x9E37_79B9u32;
        (0..2 * 48_000)
            .map(|i| {
                if !(14_400..19_200).contains(&i) {
                    return 0.0;
                }
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.5
            })
            .collect()
    }

    /// Plik WAV 16-bit mono z podanych próbek (−1..1).
    fn write_mono_wav(path: &Path, rate: u32, samples: &[f32]) {
        let data_len = (samples.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
        }
        fs::write(path, out).unwrap();
    }

    /// Odtwarzanie w czasie rzeczywistym: cisza, potem głośny szum przez 0,1 s, potem cisza.
    /// Trafienia warstw muszą się pojawić w kolejce, tak jak przy grze.
    #[test]
    fn playing_a_file_produces_layer_hits() {
        let root = temp_root("play");
        let path = root.join("klik.wav");
        write_mono_wav(&path, 48_000, &click_samples());

        let analysis = SharedAudioAnalysis::new();
        let file = PlayableFile {
            label: "klik".to_string(),
            path,
        };
        let player = FilePlayer::start(&file, analysis.clone()).expect("plik powinien się otworzyć");
        std::thread::sleep(Duration::from_millis(2_300));
        let hits = analysis.lock().unwrap().drain_layer_hits();
        drop(player);
        let _ = fs::remove_dir_all(&root);

        assert!(!hits.is_empty(), "odtwarzanie powinno dać trafienia warstw");
        assert_eq!(hits[0].hit.class, Some(crate::sound::layers::Class::A), "głośny dźwięk to klasa A");
    }

    /// Ta sama decyzja bez odtwarzania: `classify_recording` idzie tym samym potokiem, ale szybko.
    #[test]
    fn classifying_a_recording_matches_playback_without_waiting() {
        let root = temp_root("classify");
        let path = root.join("klik.wav");
        write_mono_wav(&path, 48_000, &click_samples());

        let hits = classify_recording(&path).expect("nagranie powinno się przeanalizować");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(hits.len(), 1, "jeden głośny dźwięk, jedna decyzja");
        assert_eq!(hits[0].class, Some(crate::sound::layers::Class::A));
        assert!((hits[0].onset_s - 0.3).abs() < 0.05, "onset: {}", hits[0].onset_s);
    }
}
