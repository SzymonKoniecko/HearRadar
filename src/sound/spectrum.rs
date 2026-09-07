//! Szybki, przybliżony podział energii kanału na 3 pasma częstotliwości
//! (niskie/średnie/wysokie) — bez FFT, jednobiegunowymi filtrami IIR.
//!
//! Filtr resetuje swój stan na początku każdego bufora audio (ok. 10-20 ms),
//! więc to nie jest precyzyjna analiza widmowa, tylko szybki, poglądowy
//! podział wystarczający do kolorowania kropek radaru wg "typu" dźwięku
//! (bas / środek / wysokie tony). Wystarczy to dla krótkich ramek, bo stan
//! przejściowy filtra jednobiegunowego zanika w ciągu kilku milisekund.

use std::f32::consts::TAU;

/// Pasmo częstotliwości przypisane kropce na podstawie dominującej energii.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyBand {
    /// < ~200 Hz — bas, wybuchy, silniki, uderzenia.
    Low,
    /// ~200 Hz .. 2 kHz — mowa, kroki, większość efektów.
    Mid,
    /// > ~2 kHz — syczące dźwięki, wysokie tony, brzęk.
    High,
}

impl FrequencyBand {
    pub const ALL: [Self; 3] = [Self::Low, Self::Mid, Self::High];
}

const LOW_CUTOFF_HZ: f32 = 200.0;
const HIGH_CUTOFF_HZ: f32 = 2000.0;

/// RMS energii w każdym z 3 pasm dla jednego kanału: `[low, mid, high]`.
pub type BandEnergies = [f32; 3];

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

fn one_pole_highpass(input: &[f32], sample_rate: f32, cutoff_hz: f32) -> Vec<f32> {
    let dt = 1.0 / sample_rate;
    let rc = 1.0 / (TAU * cutoff_hz);
    let alpha = rc / (rc + dt);
    let mut y = 0.0f32;
    let mut prev_x = 0.0f32;
    input
        .iter()
        .map(|&x| {
            y = alpha * (y + x - prev_x);
            prev_x = x;
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

/// Dzieli próbki JEDNEGO kanału (już wyodrębnione z interleaved bufora) na
/// 3 pasma i zwraca RMS każdego z nich.
pub fn split_bands(channel_samples: &[f32], sample_rate: f32) -> BandEnergies {
    if channel_samples.is_empty() || sample_rate <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let low = one_pole_lowpass(channel_samples, sample_rate, LOW_CUTOFF_HZ);
    let high = one_pole_highpass(channel_samples, sample_rate, HIGH_CUTOFF_HZ);
    // Środek pasma = sygnał minus niskie i wysokie (przybliżenie band-pass).
    let mid: Vec<f32> = channel_samples
        .iter()
        .zip(low.iter())
        .zip(high.iter())
        .map(|((&x, &l), &h)| x - l - h)
        .collect();

    [rms(&low), rms(&mid), rms(&high)]
}

/// Który z 3 pasm dominuje energetycznie w danej ramce.
pub fn dominant_band(energies: BandEnergies) -> FrequencyBand {
    let [low, mid, high] = energies;
    if low >= mid && low >= high {
        FrequencyBand::Low
    } else if high >= mid && high >= low {
        FrequencyBand::High
    } else {
        FrequencyBand::Mid
    }
}

/// Demultipleksuje przeplecione (interleaved) próbki na kanały i liczy
/// [`split_bands`] dla każdego z nich niezależnie.
pub fn analyze_channel_bands(samples: &[f32], channels: usize, sample_rate: f32) -> Vec<BandEnergies> {
    let channels = channels.max(1);
    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for frame in samples.chunks_exact(channels) {
        for (i, &s) in frame.iter().enumerate() {
            per_channel[i].push(s);
        }
    }

    per_channel
        .iter()
        .map(|ch| split_bands(ch, sample_rate))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_tone_dominates_low_band() {
        let sample_rate = 48_000.0;
        let freq = 60.0; // wyraźnie w paśmie "low"
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = split_bands(&samples, sample_rate);
        assert_eq!(dominant_band(energies), FrequencyBand::Low);
    }

    #[test]
    fn high_tone_dominates_high_band() {
        let sample_rate = 48_000.0;
        let freq = 8000.0; // wyraźnie w paśmie "high"
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = split_bands(&samples, sample_rate);
        assert_eq!(dominant_band(energies), FrequencyBand::High);
    }

    #[test]
    fn silence_has_zero_energy_in_every_band() {
        let energies = split_bands(&[0.0; 512], 48_000.0);
        assert_eq!(energies, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_input_returns_zeros() {
        assert_eq!(split_bands(&[], 48_000.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn analyze_channel_bands_splits_interleaved_samples_per_channel() {
        let sample_rate = 48_000.0;
        // Kanał L: ton niski, kanał R: ton wysoki, przeplecione.
        let n = 2400;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            samples.push((t * TAU * 60.0).sin()); // L: low
            samples.push((t * TAU * 8000.0).sin()); // R: high
        }
        let bands = analyze_channel_bands(&samples, 2, sample_rate);
        assert_eq!(bands.len(), 2);
        assert_eq!(dominant_band(bands[0]), FrequencyBand::Low);
        assert_eq!(dominant_band(bands[1]), FrequencyBand::High);
    }
}
