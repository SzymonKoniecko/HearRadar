//! Analiza offline klipów WAV: kierunek z różnicy czasu dotarcia do uszu (ITD)
//! i różnica poziomów (ILD) dla zdarzenia w znanej chwili.
//!
//! Używane przez `radar --analyze-clips <folder>`. Klipy z `hearanalytics` mają
//! zdarzenie w stałej chwili (domyślnie 0,2 s od początku klipu).

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use crate::sound::binaural::{estimate_itd, DEFAULT_EAR_DISTANCE_M};
use crate::sound::layers::{self, LayerHit, LayerTracker};

/// Okno ITD wokół zdarzenia: od 5 ms przed chwilą zdarzenia do 40 ms po niej.
const WINDOW_BEFORE_S: f64 = 0.005;
const WINDOW_LENGTH_S: f64 = 0.040;

/// Wynik dla jednego pliku. `None`, gdy nie dało się policzyć (np. mono albo cisza).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipReport {
    pub file: String,
    pub azimuth_deg: Option<f32>,
    pub confidence: Option<f32>,
    pub ild_db: Option<f32>,
}

/// Czyta plik WAV i zwraca częstotliwość oraz kanały jako f32 (−1..1).
pub fn read_wav(path: &Path) -> Result<(u32, Vec<Vec<f32>>)> {
    let bytes = fs::read(path).with_context(|| format!("odczyt {}", path.display()))?;
    parse_wav(&bytes).with_context(|| format!("{}", path.display()))
}

/// Parser WAV: chunk `fmt ` i `data`, PCM 16-bit (także WAVE_FORMAT_EXTENSIBLE).
pub fn parse_wav(bytes: &[u8]) -> Result<(u32, Vec<Vec<f32>>)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("to nie jest plik WAV");
    }
    let mut format: Option<(u16, u16, u32, u16)> = None; // (format, kanały, Hz, bity)
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    bail!("uszkodzony chunk fmt");
                }
                format = Some((
                    u16::from_le_bytes([body[0], body[1]]),
                    u16::from_le_bytes([body[2], body[3]]),
                    u32::from_le_bytes([body[4], body[5], body[6], body[7]]),
                    u16::from_le_bytes([body[14], body[15]]),
                ));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos = body_start + size + (size % 2);
    }

    let (fmt, channels, rate, bits) = format.context("brak chunku fmt")?;
    if !(fmt == 1 || fmt == 0xFFFE) || bits != 16 {
        bail!("obsługiwany jest tylko PCM 16-bit (format {fmt}, {bits} bitów)");
    }
    if channels == 0 {
        bail!("zero kanałów");
    }
    let data = data.context("brak chunku data")?;
    let n = channels as usize;
    let frames = data.len() / (2 * n);
    let mut out = vec![Vec::with_capacity(frames); n];
    for frame in 0..frames {
        for (c, channel) in out.iter_mut().enumerate() {
            let i = (frame * n + c) * 2;
            let value = i16::from_le_bytes([data[i], data[i + 1]]);
            channel.push(value as f32 / 32768.0);
        }
    }
    Ok((rate, out))
}

/// Liczy kierunek i ILD dla zdarzenia w chwili `onset_s` (sekundy od początku klipu).
pub fn analyze_clip(channels: &[Vec<f32>], rate: u32, onset_s: f64) -> (Option<f32>, Option<f32>, Option<f32>) {
    if channels.len() < 2 {
        return (None, None, None);
    }
    let start = ((onset_s - WINDOW_BEFORE_S) * rate as f64).max(0.0) as usize;
    let length = (WINDOW_LENGTH_S * rate as f64) as usize;
    let end = (start + length).min(channels[0].len()).min(channels[1].len());
    if end <= start {
        return (None, None, None);
    }
    match estimate_itd(
        &channels[0][start..end],
        &channels[1][start..end],
        rate as f32,
        DEFAULT_EAR_DISTANCE_M,
    ) {
        Some(est) => (Some(est.azimuth_deg), Some(est.confidence), Some(est.ild_db)),
        None => (None, None, None),
    }
}

/// Klasa i warstwa zdarzenia w klipie, z onsetem w `onset_s` (jak [`analyze_clip`]). Tracker
/// wykrywa zdarzenia sam, więc bierzemy decyzję z najbliższym czasem. `None`, gdy takiej nie ma.
pub fn classify_clip(channels: &[Vec<f32>], rate: u32, onset_s: f64) -> Option<LayerHit> {
    if channels.is_empty() {
        return None;
    }
    let n = channels.len();
    let samples = layers::interleave(channels);
    let part = ((rate as f64 * 0.01).round().max(1.0) as usize) * n;

    let mut tracker = LayerTracker::default();
    let mut hits = Vec::new();
    for chunk in samples.chunks(part) {
        tracker.push_frames(chunk, n, rate as f32);
        hits.extend(tracker.take_ready());
    }
    hits.into_iter()
        .min_by(|a, b| (a.onset_s - onset_s).abs().total_cmp(&(b.onset_s - onset_s).abs()))
        .filter(|hit| (hit.onset_s - onset_s).abs() < 0.05)
}

/// Klasa i warstwa dla każdego klipu w folderze (posortowane po nazwie).
pub fn classify_folder(dir: &Path, onset_s: f64) -> Result<Vec<(String, Option<LayerHit>)>> {
    let mut out = Vec::new();
    for path in wav_paths(dir)? {
        let (rate, channels) = read_wav(&path)?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        out.push((name, classify_clip(&channels, rate, onset_s)));
    }
    Ok(out)
}

/// Pliki WAV w folderze, posortowane po nazwie.
fn wav_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("odczyt folderu {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav")))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Przechodzi po wszystkich plikach WAV w folderze (posortowane po nazwie).
pub fn analyze_folder(dir: &Path, onset_s: f64) -> Result<Vec<ClipReport>> {
    let mut reports = Vec::new();
    for path in wav_paths(dir)? {
        let (rate, channels) = read_wav(&path)?;
        let (azimuth_deg, confidence, ild_db) = analyze_clip(&channels, rate, onset_s);
        reports.push(ClipReport {
            file: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            azimuth_deg,
            confidence,
            ild_db,
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// Pseudolosowy impuls 40 ms zaczynający się w `start` (w próbkach), reszta cisza.
    fn burst(n: usize, start: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut out = vec![0.0f32; n];
        for sample in out.iter_mut().skip(start).take((0.04 * RATE as f64) as usize) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *sample = ((state >> 16) % 16_001) as f32 / 16_000.0 - 0.5;
        }
        out
    }

    fn delayed(signal: &[f32], by: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; signal.len()];
        out[by..].copy_from_slice(&signal[..signal.len() - by]);
        out
    }

    /// Plik WAV 16-bit PCM z podanych kanałów (f32 w zakresie −1..1).
    fn wav_bytes(channels: &[Vec<f32>], rate: u32) -> Vec<u8> {
        let n = channels.len();
        let frames = channels[0].len();
        let data_len = frames * n * 2;
        let mut out = Vec::with_capacity(44 + data_len);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * n as u32 * 2).to_le_bytes());
        out.extend_from_slice(&((n * 2) as u16).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for frame in 0..frames {
            for channel in channels {
                let v = (channel[frame] * 32767.0).round() as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn expected_azimuth(delay_samples: usize) -> f32 {
        (delay_samples as f32 / RATE as f32 * 343.0 / DEFAULT_EAR_DISTANCE_M)
            .asin()
            .to_degrees()
    }

    #[test]
    fn wav_roundtrip_keeps_channels_and_rate() {
        let left = vec![0.5, -0.25, 0.0];
        let right = vec![-0.5, 0.25, 0.0];
        let (rate, channels) = parse_wav(&wav_bytes(&[left.clone(), right.clone()], RATE)).unwrap();
        assert_eq!(rate, RATE);
        assert_eq!(channels.len(), 2);
        for (a, b) in channels[0].iter().zip(&left) {
            assert!((a - b).abs() < 1e-3);
        }
        for (a, b) in channels[1].iter().zip(&right) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn source_on_the_right_gives_the_expected_positive_angle() {
        let base = burst(RATE as usize, 48_000 / 5, 7); // impuls w 0,2 s
        let left = delayed(&base, 10); // lewe ucho później → źródło z prawej
        let (azimuth, confidence, _) = analyze_clip(&[left, base], RATE, 0.2);
        let azimuth = azimuth.expect("azymut dla wyraźnego impulsu");
        assert!((azimuth - expected_azimuth(10)).abs() < 2.0, "azymut {azimuth}");
        assert!(confidence.unwrap() > 0.0);
    }

    #[test]
    fn source_on_the_left_gives_a_negative_angle() {
        let base = burst(RATE as usize, 48_000 / 5, 9);
        let right = delayed(&base, 10);
        let (azimuth, _, _) = analyze_clip(&[base, right], RATE, 0.2);
        assert!(azimuth.unwrap() < -15.0);
    }

    #[test]
    fn mono_clip_gives_no_angle() {
        let base = burst(RATE as usize, 48_000 / 5, 3);
        assert_eq!(analyze_clip(&[base], RATE, 0.2), (None, None, None));
    }

    #[test]
    fn folder_is_read_file_by_file_in_name_order() {
        let dir = std::env::temp_dir().join(format!("hearradar_clips_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = burst(RATE as usize, 48_000 / 5, 11);
        fs::write(dir.join("E002_left.wav"), wav_bytes(&[base.clone(), delayed(&base, 10)], RATE)).unwrap();
        fs::write(dir.join("E001_right.wav"), wav_bytes(&[delayed(&base, 10), base.clone()], RATE)).unwrap();
        fs::write(dir.join("notatki.txt"), "ignorowany").unwrap();

        let reports = analyze_folder(&dir, 0.2).unwrap();
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(reports.len(), 2, "tylko pliki WAV");
        assert_eq!(reports[0].file, "E001_right.wav");
        assert!(reports[0].azimuth_deg.unwrap() > 15.0, "prawy: dodatni kąt");
        assert!(reports[1].azimuth_deg.unwrap() < -15.0, "lewy: ujemny kąt");
    }

    #[test]
    fn loud_burst_at_the_onset_is_classified_as_a_shot() {
        let base = burst(RATE as usize, 48_000 / 5, 13); // ±0,5, ok. −11 dBFS, w 0,2 s
        let hit = classify_clip(&[base.clone(), base], RATE, 0.2).expect("decyzja dla klipu z zapasem");
        assert_eq!(hit.class, Some(crate::sound::layers::Class::A));
        assert_eq!(hit.layer, crate::sound::layers::Layer::Shots);
    }
}
