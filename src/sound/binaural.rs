//! Azymut z różnicy czasu dotarcia dźwięku do uszu (ITD) dla stereo binauralnego.
//!
//! Gra miksuje dźwięk do stereo z efektem HRTF (patrz `ML_RADAR_ANALYSIS.md`, Q13),
//! więc lewy i prawy kanał niosą różnicę czasu między uszami. Ten moduł szuka jej
//! metodą GCC-PHAT, w paśmie, w którym faza jest wiarygodna, i zamienia ją na kąt.
//!
//! Jak `direction.rs`, moduł nie zależy od `cpal`: operuje na próbkach `f32`,
//! więc jest w pełni testowalny bez urządzenia audio.
//!
//! Konwencja: kąt dodatni = źródło z prawej (jak `ChannelRole::radar_angle_degrees`).
//! ITD dodatnie = lewe ucho słyszy później. ILD dodatnie = prawy kanał głośniejszy.

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Prędkość dźwięku w powietrzu (m/s), ok. 20 °C.
pub const SPEED_OF_SOUND_M_S: f32 = 343.0;
/// Rozstaw uszu (m). Wartość orientacyjna, do kalibracji dla gry.
#[allow(dead_code)] // używane przy podłączaniu do zdarzeń (S5)
pub const DEFAULT_EAR_DISTANCE_M: f32 = 0.2;
/// Pasmo, w którym faza jest wiarygodna dla ITD. Powyżej ok. 1,5 kHz jest niejednoznaczna.
pub const ITD_BAND_HZ: (f32, f32) = (150.0, 1500.0);
/// Część brzegów okna Tukey objęta zanikiem. Środek zostaje płaski, więc przesunięcie
/// fragmentu w oknie nie zmienia wagi jednego kanału względem drugiego.
const TUKEY_ALPHA: f32 = 0.2;
/// Poniżej tej sumy kwadratów okno uznajemy za ciszę.
const SILENCE_ENERGY: f32 = 1e-9;
/// Najkrótsze okno, dla którego liczymy.
const MIN_SAMPLES: usize = 64;

/// Wynik estymacji dla jednego okna stereo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItdEstimate {
    /// Różnica czasu dotarcia (s). Dodatnia, gdy źródło jest z prawej.
    pub itd_s: f32,
    /// Azymut w stopniach, zakres −90..=90. Dodatni = prawo.
    /// Przód i tył dają tę samą wartość (niejednoznaczność stożka).
    pub azimuth_deg: f32,
    /// Udział piku w sumie dodatnich korelacji w przeszukiwanym zakresie, 0..=1.
    pub confidence: f32,
    /// `10·log10(E_P / E_L)`. Dodatnia, gdy prawy kanał jest głośniejszy.
    pub ild_db: f32,
}

/// Szuka ITD metodą GCC-PHAT i zamienia ją na azymut.
///
/// `left` i `right` to to samo okno z dwóch kanałów, o tej samej długości. Zwraca
/// `None` dla zbyt krótkiego okna, niezgodnych długości, ciszy albo gdy jedna
/// strona jest cicha, bo wtedy ITD nie ma sensu.
#[allow(dead_code)] // podłączenie do zdarzeń w S5
pub fn estimate_itd(
    left: &[f32],
    right: &[f32],
    sample_rate: f32,
    ear_distance_m: f32,
) -> Option<ItdEstimate> {
    let n = left.len();
    if n < MIN_SAMPLES || right.len() != n || sample_rate <= 0.0 || ear_distance_m <= 0.0 {
        return None;
    }
    let energy_l: f32 = left.iter().map(|s| s * s).sum();
    let energy_r: f32 = right.iter().map(|s| s * s).sum();
    if energy_l < SILENCE_ENERGY || energy_r < SILENCE_ENERGY {
        return None;
    }

    let n_fft = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n_fft);
    let inverse = planner.plan_fft_inverse(n_fft);

    let window = tukey(n, TUKEY_ALPHA);
    let mut buf_l = padded_input(left, &window, n_fft);
    let mut buf_r = padded_input(right, &window, n_fft);
    forward.process(&mut buf_l);
    forward.process(&mut buf_r);

    // GCC-PHAT: iloczyn widm normalizowany do samej fazy, tylko w paśmie ITD.
    let (f_lo, f_hi) = ITD_BAND_HZ;
    let bin_hz = sample_rate / n_fft as f32;
    let mut cross: Vec<Complex<f32>> = (0..n_fft)
        .map(|k| {
            let freq = bin_hz * k.min(n_fft - k) as f32;
            let product = buf_l[k] * buf_r[k].conj();
            let magnitude = product.norm();
            if freq < f_lo || freq > f_hi || magnitude < 1e-12 {
                Complex::new(0.0, 0.0)
            } else {
                product / magnitude
            }
        })
        .collect();
    inverse.process(&mut cross);
    let correlation: Vec<f32> = cross.iter().map(|c| c.re).collect();

    // Opóźnienia fizycznie możliwe: ±(d/c)·fs, z dwiema próbkami zapasu na interpolację.
    let max_lag = (ear_distance_m / SPEED_OF_SOUND_M_S * sample_rate).ceil() as isize + 2;
    let at = |lag: isize| correlation[lag.rem_euclid(n_fft as isize) as usize];

    let (mut best_lag, mut best) = (0isize, f32::MIN);
    for lag in -max_lag..=max_lag {
        if at(lag) > best {
            best = at(lag);
            best_lag = lag;
        }
    }
    if best <= 0.0 {
        return None;
    }
    let total: f32 = (-max_lag..=max_lag).map(|lag| at(lag).max(0.0)).sum();
    let confidence = (best / total).clamp(0.0, 1.0);

    // Interpolacja paraboliczna, żeby ITD miało ułamki próbki.
    let (y_prev, y_next) = (at(best_lag - 1), at(best_lag + 1));
    let curvature = y_prev - 2.0 * best + y_next;
    let offset = if curvature.abs() > 1e-12 {
        (0.5 * (y_prev - y_next) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let itd_s = (best_lag as f32 + offset) / sample_rate;

    let sin_theta = (itd_s * SPEED_OF_SOUND_M_S / ear_distance_m).clamp(-1.0, 1.0);
    Some(ItdEstimate {
        itd_s,
        azimuth_deg: sin_theta.asin().to_degrees(),
        confidence,
        ild_db: 10.0 * (energy_r / energy_l).log10(),
    })
}

/// Okno Tukey o długości `n`: brzegi zanikają, środek ma wagę 1.
fn tukey(n: usize, alpha: f32) -> Vec<f32> {
    let edge = alpha * (n - 1) as f32 / 2.0;
    (0..n)
        .map(|i| {
            let distance = (i as f32).min((n - 1 - i) as f32);
            if edge <= 0.0 || distance >= edge {
                1.0
            } else {
                0.5 * (1.0 - (PI * distance / edge).cos())
            }
        })
        .collect()
}

/// Próbki przemnożone przez okno, dopełnione zerami do `n_fft`.
fn padded_input(samples: &[f32], window: &[f32], n_fft: usize) -> Vec<Complex<f32>> {
    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    for (slot, (s, w)) in buffer.iter_mut().zip(samples.iter().zip(window)) {
        *slot = Complex::new(s * w, 0.0);
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{FftPlanner, num_complex::Complex};

    const FS: f32 = 48_000.0;
    const WINDOW: usize = 2048;
    const BURST_LEN: usize = 1024;
    const BURST_AT: usize = 384;
    const TOLERANCE_DEG: f32 = 5.0;

    /// Deterministyczny szum (xorshift), wartości z zakresu −1..1.
    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Krótki impuls szumu w środku okna, reszta zera.
    fn burst(seed: u32) -> Vec<f32> {
        let mut out = vec![0.0; WINDOW];
        out[BURST_AT..BURST_AT + BURST_LEN].copy_from_slice(&noise(BURST_LEN, seed));
        out
    }

    /// Przesuwa sygnał o `delay` próbek, także ułamkowo, w dziedzinie częstotliwości.
    /// Dodatnie `delay` = później.
    fn delayed(signal: &[f32], delay: f32) -> Vec<f32> {
        let n_fft = (4 * signal.len()).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        let mut buf: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| Complex::new(signal.get(i).copied().unwrap_or(0.0), 0.0))
            .collect();
        forward.process(&mut buf);
        for (k, bin) in buf.iter_mut().enumerate() {
            let signed = if k <= n_fft / 2 { k as f32 } else { k as f32 - n_fft as f32 };
            let phase = -2.0 * PI * signed * delay / n_fft as f32;
            *bin *= Complex::new(phase.cos(), phase.sin());
        }
        inverse.process(&mut buf);
        buf.iter().take(signal.len()).map(|c| c.re / n_fft as f32).collect()
    }

    /// Stereo z jednego źródła pod kątem `azimuth_deg`. Dalsze ucho dostaje opóźnienie ITD.
    fn stereo_from_azimuth(azimuth_deg: f32, source: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let itd_samples =
            DEFAULT_EAR_DISTANCE_M * azimuth_deg.to_radians().sin() / SPEED_OF_SOUND_M_S * FS;
        // Źródło z prawej: lewe ucho później, więc lewy kanał jest opóźniony.
        let left = delayed(source, itd_samples.max(0.0));
        let right = delayed(source, (-itd_samples).max(0.0));
        (left, right)
    }

    /// Dodaje do `dst` przesunięty o `shift` próbek i wzmocniony `gain` sygnał `src`.
    fn add_scaled_shifted(dst: &mut [f32], src: &[f32], shift: usize, gain: f32) {
        for (i, d) in dst.iter_mut().enumerate().skip(shift) {
            *d += gain * src[i - shift];
        }
    }

    /// Dodaje szum tak, żeby stosunek sygnału do szumu wynosił `snr_db`.
    fn add_noise_at_snr(signal: &mut [f32], snr_db: f32, seed: u32) {
        let power = signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32;
        let target = (power / 10f32.powf(snr_db / 10.0)).sqrt();
        let raw = noise(signal.len(), seed);
        let raw_rms = (raw.iter().map(|s| s * s).sum::<f32>() / raw.len() as f32).sqrt();
        for (s, n) in signal.iter_mut().zip(&raw) {
            *s += n * target / raw_rms;
        }
    }

    fn estimate(left: &[f32], right: &[f32]) -> ItdEstimate {
        estimate_itd(left, right, FS, DEFAULT_EAR_DISTANCE_M)
            .expect("okno z energią na obu kanałach powinno dać wynik")
    }

    #[test]
    fn identical_channels_give_centre() {
        let s = burst(1);
        let est = estimate(&s, &s);
        assert!(est.azimuth_deg.abs() < 0.5, "azymut: {}", est.azimuth_deg);
        assert!(est.ild_db.abs() < 1e-3, "ILD: {}", est.ild_db);
        assert!(est.confidence > 0.0);
    }

    #[test]
    fn recovers_known_azimuths_within_tolerance() {
        let s = burst(2);
        for azimuth in [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0] {
            let (l, r) = stereo_from_azimuth(azimuth, &s);
            let est = estimate(&l, &r);
            assert!(
                (est.azimuth_deg - azimuth).abs() <= TOLERANCE_DEG,
                "θ={azimuth}: otrzymano {}",
                est.azimuth_deg
            );
        }
    }

    #[test]
    fn source_on_the_right_gives_positive_angle_and_itd() {
        let s = burst(3);
        let (l, r) = stereo_from_azimuth(40.0, &s);
        let est = estimate(&l, &r);
        assert!(est.azimuth_deg > 0.0, "azymut: {}", est.azimuth_deg);
        assert!(est.itd_s > 0.0, "ITD: {}", est.itd_s);
    }

    #[test]
    fn ild_is_positive_when_right_channel_is_louder() {
        let s = burst(4);
        let quieter_left: Vec<f32> = s.iter().map(|x| x * 0.5).collect();
        let est = estimate(&quieter_left, &s);
        // Prawy o czynnik 2 głośniejszy: 10·log10(4) ≈ 6,02 dB.
        assert!((est.ild_db - 6.02).abs() < 0.05, "ILD: {}", est.ild_db);
    }

    #[test]
    fn survives_noise_at_10_db_snr() {
        let s = burst(5);
        let (mut l, mut r) = stereo_from_azimuth(30.0, &s);
        add_noise_at_snr(&mut l, 10.0, 11);
        add_noise_at_snr(&mut r, 10.0, 12);
        let est = estimate(&l, &r);
        assert!(
            (est.azimuth_deg - 30.0).abs() <= 10.0,
            "θ=30 przy SNR 10 dB: otrzymano {}",
            est.azimuth_deg
        );
    }

    #[test]
    fn survives_an_early_reflection_from_the_other_side() {
        let s = burst(6);
        let (mut l, mut r) = stereo_from_azimuth(30.0, &s);
        // Odbicie 8 ms później, −6 dB, z lewej: lewe ucho słyszy je pierwsze.
        add_scaled_shifted(&mut l, &s, 400, 0.5);
        add_scaled_shifted(&mut r, &s, 410, 0.5);
        let est = estimate(&l, &r);
        assert!(
            (est.azimuth_deg - 30.0).abs() <= TOLERANCE_DEG,
            "θ=30 z odbiciem: otrzymano {}",
            est.azimuth_deg
        );
    }

    #[test]
    fn front_and_back_sources_share_the_same_itd() {
        let s = burst(7);
        let (lf, rf) = stereo_from_azimuth(30.0, &s);
        let (lb, rb) = stereo_from_azimuth(150.0, &s);
        let front = estimate(&lf, &rf);
        let back = estimate(&lb, &rb);
        assert!(
            (front.azimuth_deg - back.azimuth_deg).abs() < 0.5,
            "przód {} vs tył {}",
            front.azimuth_deg,
            back.azimuth_deg
        );
        assert!((back.azimuth_deg - 30.0).abs() <= TOLERANCE_DEG);
    }

    #[test]
    fn silence_or_one_silent_side_gives_none() {
        let s = burst(8);
        let zeros = vec![0.0; WINDOW];
        assert!(estimate_itd(&zeros, &zeros, FS, DEFAULT_EAR_DISTANCE_M).is_none());
        assert!(estimate_itd(&s, &zeros, FS, DEFAULT_EAR_DISTANCE_M).is_none());
    }

    #[test]
    fn short_or_mismatched_windows_give_none() {
        assert!(estimate_itd(&[1.0; 8], &[1.0; 8], FS, DEFAULT_EAR_DISTANCE_M).is_none());
        assert!(estimate_itd(&[1.0; 128], &[1.0; 64], FS, DEFAULT_EAR_DISTANCE_M).is_none());
    }

    #[test]
    fn lag_beyond_the_head_stays_within_the_physical_range() {
        // Impulsy 60 próbek od siebie, czyli dalej niż fizycznie możliwe ±28.
        let mut l = vec![0.0; WINDOW];
        let mut r = vec![0.0; WINDOW];
        l[500] = 1.0;
        r[560] = 1.0;
        let est = estimate(&l, &r);
        assert!(est.azimuth_deg.abs() <= 90.0, "azymut: {}", est.azimuth_deg);
        assert!(est.itd_s.abs() <= 31.0 / FS, "ITD: {}", est.itd_s);
    }
}
