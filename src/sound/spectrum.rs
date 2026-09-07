//! Szybki, przybliżony podział energii kanału na 3 pasma częstotliwości
//! (niskie/średnie/wysokie) — bez FFT, jednobiegunowymi filtrami IIR.
//!
//! Filtr resetuje swój stan na początku każdego bufora audio (ok. 10-20 ms),
//! więc to nie jest precyzyjna analiza widmowa, tylko szybki, poglądowy
//! podział wystarczający do kolorowania kropek radaru wg "typu" dźwięku
//! (bas / środek / wysokie tony). Wystarczy to dla krótkich ramek, bo stan
//! przejściowy filtra jednobiegunowego zanika w ciągu kilku milisekund.

use std::f32::consts::TAU;

/// Progi domyślnych 3 pasm ([`default_color_rules`]) — zachowane jako stałe,
/// żeby dawne "low/mid/high" zostały udokumentowane w jednym miejscu.
const LOW_CUTOFF_HZ: f32 = 200.0;
const HIGH_CUTOFF_HZ: f32 = 2000.0;

fn one_pole_lowpass(input: &[f32], sample_rate: f32, cutoff_hz: f32) -> Vec<f32> {
    let dt = 1.0 / sample_rate;
    let rc = 1.0 / (TAU * cutoff_hz);
    let alpha = dt / (rc + dt);
    let mut y = 0.0f32;
    input
        .iter()
        .map(|&x| {
            y += alpha * (x - y);
            y
        })
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Energia sygnału w paśmie `[freq_min, freq_max)` Hz (RMS), liczona jako
/// różnica dwóch dolnoprzepustowych filtrów jednobiegunowych (bandpass ≈
/// lowpass(max) - lowpass(min)). `freq_min <= 0.0` daje czysty lowpass;
/// `freq_max` powyżej sensownego zakresu słyszalnego działa jak highpass.
fn band_energy(channel_samples: &[f32], sample_rate: f32, freq_min: f32, freq_max: f32) -> f32 {
    if channel_samples.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }
    let low_cut = one_pole_lowpass(channel_samples, sample_rate, freq_max.max(1.0));
    if freq_min <= 0.0 {
        return rms(&low_cut);
    }
    let low_cut_min = one_pole_lowpass(channel_samples, sample_rate, freq_min.max(1.0));
    let band: Vec<f32> = low_cut
        .iter()
        .zip(low_cut_min.iter())
        .map(|(&hi, &lo)| hi - lo)
        .collect();
    rms(&band)
}

// ── Konfigurowalne reguły "parametr → kolor" ──────────────────────────────────
//
// Dowolna liczba reguł zdefiniowanych przez użytkownika: każda reguła to
// zakres częstotliwości + kolor, edytowalne w panelu ustawień radaru.
// [`default_color_rules`] odtwarza dawny sztywny podział na 3 pasma
// (niskie/średnie/wysokie), ale to tylko jeden z możliwych zestawów reguł.

/// Jedna reguła kolorowania kropki: zakres częstotliwości `[freq_min, freq_max)`
/// Hz i przypisany jej kolor RGB. Użytkownik może mieć ich dowolnie wiele —
/// pierwsza dopasowana (a właściwie: energetycznie dominująca) reguła
/// decyduje o kolorze danej ramki analizy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorRule {
    pub freq_min: f32,
    pub freq_max: f32,
    pub color: [u8; 3],
}

impl ColorRule {
    pub fn new(freq_min: f32, freq_max: f32, color: [u8; 3]) -> Self {
        Self { freq_min, freq_max, color }
    }
}

/// Domyślne reguły — dokładnie te same 3 sztywne pasma (Low/Mid/High), więc
/// domyślne zachowanie/kolory radaru się nie zmieniają, dopóki użytkownik
/// nie zacznie edytować listy w ustawieniach.
pub fn default_color_rules() -> Vec<ColorRule> {
    vec![
        ColorRule::new(0.0, LOW_CUTOFF_HZ, [255, 90, 90]),
        ColorRule::new(LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ, [0, 220, 220]),
        ColorRule::new(HIGH_CUTOFF_HZ, 20_000.0, [255, 220, 90]),
    ]
}

/// Energia (RMS) jednego kanału w każdej z podanych reguł, w tej samej
/// kolejności co `rules`.
fn rule_energies(channel_samples: &[f32], sample_rate: f32, rules: &[ColorRule]) -> Vec<f32> {
    rules
        .iter()
        .map(|r| band_energy(channel_samples, sample_rate, r.freq_min, r.freq_max))
        .collect()
}

/// Demultipleksuje przeplecione próbki na kanały i liczy [`rule_energies`]
/// dla każdego z nich wg dowolnej, zdefiniowanej przez użytkownika listy reguł.
/// Zwraca `Vec` (per kanał) z `Vec` (per reguła, w kolejności `rules`).
pub fn analyze_channel_rule_energies(
    samples: &[f32],
    channels: usize,
    sample_rate: f32,
    rules: &[ColorRule],
) -> Vec<Vec<f32>> {
    let channels = channels.max(1);
    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for frame in samples.chunks_exact(channels) {
        for (i, &s) in frame.iter().enumerate() {
            per_channel[i].push(s);
        }
    }

    per_channel
        .iter()
        .map(|ch| rule_energies(ch, sample_rate, rules))
        .collect()
}

/// Indeks energetycznie dominującej reguły (albo `None`, gdy lista pusta).
pub fn dominant_rule_index(energies: &[f32]) -> Option<usize> {
    energies
        .iter()
        .enumerate()
        .fold(None, |best, (i, &e)| match best {
            Some((_, best_e)) if best_e >= e => best,
            _ => Some((i, e)),
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_tone_dominates_the_low_default_rule() {
        let sample_rate = 48_000.0;
        let freq = 60.0; // wyraźnie w domyślnym paśmie "low" (0-200 Hz)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = rule_energies(&samples, sample_rate, &default_color_rules());
        assert_eq!(dominant_rule_index(&energies), Some(0));
    }

    #[test]
    fn high_tone_dominates_the_high_default_rule() {
        let sample_rate = 48_000.0;
        let freq = 8000.0; // wyraźnie w domyślnym paśmie "high" (2-20 kHz)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = rule_energies(&samples, sample_rate, &default_color_rules());
        assert_eq!(dominant_rule_index(&energies), Some(2));
    }

    #[test]
    fn silence_has_zero_energy_in_every_rule() {
        let energies = rule_energies(&[0.0; 512], 48_000.0, &default_color_rules());
        for e in energies {
            assert_eq!(e, 0.0);
        }
    }

    #[test]
    fn empty_input_returns_zeros() {
        let energies = rule_energies(&[], 48_000.0, &default_color_rules());
        for e in energies {
            assert_eq!(e, 0.0);
        }
    }

    #[test]
    fn analyze_channel_rule_energies_splits_interleaved_samples_per_channel() {
        let sample_rate = 48_000.0;
        // Kanał L: ton niski, kanał R: ton wysoki, przeplecione.
        let n = 2400;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            samples.push((t * TAU * 60.0).sin()); // L: low
            samples.push((t * TAU * 8000.0).sin()); // R: high
        }
        let rules = default_color_rules();
        let per_channel = analyze_channel_rule_energies(&samples, 2, sample_rate, &rules);
        assert_eq!(per_channel.len(), 2);
        assert_eq!(dominant_rule_index(&per_channel[0]), Some(0));
        assert_eq!(dominant_rule_index(&per_channel[1]), Some(2));
    }

    // ---------- reguły kolorowania o dowolnej liczbie/zakresach ----------

    #[test]
    fn dominant_rule_index_picks_the_strongest_energy() {
        assert_eq!(dominant_rule_index(&[0.1, 0.9, 0.3]), Some(1));
        assert_eq!(dominant_rule_index(&[]), None);
        assert_eq!(dominant_rule_index(&[0.5]), Some(0));
    }

    #[test]
    fn arbitrary_number_of_rules_each_gets_its_own_energy() {
        let sample_rate = 48_000.0;
        // 5 reguł zamiast sztywnych 3 — użytkownik może zdefiniować dowolną liczbę.
        let rules = vec![
            ColorRule::new(0.0, 100.0, [1, 0, 0]),
            ColorRule::new(100.0, 500.0, [2, 0, 0]),
            ColorRule::new(500.0, 3000.0, [3, 0, 0]),
            ColorRule::new(3000.0, 9000.0, [4, 0, 0]),
            ColorRule::new(9000.0, 20_000.0, [5, 0, 0]),
        ];
        let freq = 5000.0; // powinno dominować w regule 4 (indeks 3: 3000-9000 Hz)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = rule_energies(&samples, sample_rate, &rules);
        assert_eq!(energies.len(), rules.len());
        assert_eq!(dominant_rule_index(&energies), Some(3));
    }

    #[test]
    fn analyze_channel_rule_energies_matches_channel_and_rule_count() {
        let sample_rate = 48_000.0;
        let rules = default_color_rules();
        let samples = vec![0.1f32; 6 * 8]; // 6 klatek, 8 kanałów
        let result = analyze_channel_rule_energies(&samples, 8, sample_rate, &rules);
        assert_eq!(result.len(), 8);
        for per_channel in &result {
            assert_eq!(per_channel.len(), rules.len());
        }
    }

    #[test]
    fn default_color_rules_has_three_rules_covering_full_audible_range() {
        let rules = default_color_rules();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].freq_min, 0.0);
        assert_eq!(rules.last().unwrap().freq_max, 20_000.0);
    }
}
