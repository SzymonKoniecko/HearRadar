//! Szybki, przybliżony podział energii kanału na kilka pasm częstotliwości
//! (sub-bas / bas / średnie / wysoko-średnie / wysokie) — bez FFT,
//! jednobiegunowymi filtrami IIR.
//!
//! Filtr resetuje swój stan na początku każdego bufora audio (ok. 10-20 ms),
//! więc to nie jest precyzyjna analiza widmowa, tylko szybki, poglądowy
//! podział wystarczający do kolorowania kropek radaru wg "typu" dźwięku.
//! Wystarczy to dla krótkich ramek, bo stan przejściowy filtra
//! jednobiegunowego zanika w ciągu kilku milisekund.

use std::f32::consts::TAU;

/// Progi domyślnych 5 pasm ([`default_color_rules`]) — zachowane jako stałe,
/// żeby granice zostały udokumentowane w jednym miejscu.
const SUB_BASS_CUTOFF_HZ: f32 = 100.0;
const BASS_CUTOFF_HZ: f32 = 500.0;
const MID_CUTOFF_HZ: f32 = 2000.0;
const PRESENCE_CUTOFF_HZ: f32 = 6000.0;

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
    // Pusty albo odwrócony zakres (min >= max) to błąd konfiguracji. Bez tego
    // różnica dwóch filtrów dawałaby niezerową energię "z niczego" i taka
    // reguła mogłaby wygrywać kolor kropek — traktujemy ją jak nieaktywną.
    if freq_max <= freq_min {
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
// [`default_color_rules`] odtwarza domyślny podział na 5 pasm, ale to tylko
// jeden z możliwych zestawów reguł.

fn default_weight() -> f32 {
    1.0
}

/// Jedna reguła kolorowania kropki: zakres częstotliwości `[freq_min, freq_max)`
/// Hz i przypisany jej kolor RGB. Użytkownik może mieć ich dowolnie wiele —
/// pierwsza dopasowana (a właściwie: energetycznie dominująca) reguła
/// decyduje o kolorze danej ramki analizy. `label` to opcjonalny opis "sytuacji"
/// (np. "Kroki", "Strzały") pokazywany w UI — czysto informacyjny, nie
/// wpływa na analizę.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColorRule {
    pub freq_min: f32,
    pub freq_max: f32,
    pub color: [u8; 3],
    #[serde(default)]
    pub label: Option<String>,
    /// Mnożnik energii tej reguły przed porównaniem dominacji (patrz
    /// [`dominant_rule_index`]). Domyślnie `1.0`. Większość naturalnych
    /// dźwięków (kroki, mowa, szum otoczenia) ma znacznie więcej energii w
    /// niskich/średnich pasmach niż w wysokich, więc przy porównaniu surowej
    /// energii RMS pasmo wysokie prawie nigdy nie wygrywa — nawet gdy realnie
    /// słychać w nim coś istotnego (alarm, dzwonek, tłukące się szkło).
    /// `weight` pozwala to skompensować bez zmiany granic pasma. Pole ma
    /// `#[serde(default)]`, żeby wczytanie starszego pliku konfiguracji
    /// (sprzed dodania tego pola) dało neutralne `1.0`, nie `0.0`.
    #[serde(default = "default_weight")]
    pub weight: f32,
}

impl ColorRule {
    pub fn new(freq_min: f32, freq_max: f32, color: [u8; 3]) -> Self {
        Self { freq_min, freq_max, color, label: None, weight: default_weight() }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Domyślne reguły — 5 pasm zamiast dawnych sztywnych 3, z wagą rosnącą wraz
/// z częstotliwością. W trzech niezależnych nagraniach testowych
/// (`sample_log_0001..0003`) dawne pasmo "Wysokie" (2000-20000 Hz, waga 1.0)
/// wygrywało porównanie energii w raptem 0,5-1,5% ramek — naturalne dźwięki
/// mają zwykle opadające widmo (więcej energii nisko niż wysoko), więc bez
/// korekty wysokie tony (alarmy, dzwonki, tłukące się szkło) prawie nigdy nie
/// przebijały się na kolor radaru mimo realnej obecności. Przy podziale na 5
/// węższych pasm ta sama korekta jest rozłożona stopniowo zamiast jednego
/// skoku wagi na granicy 2000 Hz — patrz doc-komentarz [`ColorRule::weight`].
pub fn default_color_rules() -> Vec<ColorRule> {
    vec![
        ColorRule::new(0.0, SUB_BASS_CUTOFF_HZ, [200, 40, 40]).with_label("Sub-bas"),
        ColorRule::new(SUB_BASS_CUTOFF_HZ, BASS_CUTOFF_HZ, [255, 90, 90]).with_label("Bas"),
        ColorRule::new(BASS_CUTOFF_HZ, MID_CUTOFF_HZ, [0, 220, 220])
            .with_label("Średnie (mowa)")
            .with_weight(1.5),
        ColorRule::new(MID_CUTOFF_HZ, PRESENCE_CUTOFF_HZ, [255, 150, 60])
            .with_label("Wysoko-średnie")
            .with_weight(3.0),
        ColorRule::new(PRESENCE_CUTOFF_HZ, 20_000.0, [255, 220, 90])
            .with_label("Wysokie")
            .with_weight(5.0),
    ]
}

/// Energia (RMS) jednego kanału w każdej z podanych reguł, w tej samej
/// kolejności co `rules`, przemnożona przez `rule.weight` każdej reguły.
fn rule_energies(channel_samples: &[f32], sample_rate: f32, rules: &[ColorRule]) -> Vec<f32> {
    rules
        .iter()
        .map(|r| band_energy(channel_samples, sample_rate, r.freq_min, r.freq_max) * r.weight)
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

/// Kolor używany, gdy nie da się ustalić dominującej reguły (np. lista reguł
/// jest pusta) — neutralny cyjan, taki sam jak dawny "domyślny" kolor kropek.
pub const FALLBACK_DOT_RGB: (u8, u8, u8) = (0, 220, 220);

/// Wybiera kolor energetycznie dominującej reguły dla danego kanału. `energies`
/// to lista energii w tej samej kolejności co `rules` (patrz
/// [`analyze_channel_rule_energies`]); gdy jest krótsza/pusta albo `rules`
/// jest puste, zwraca [`FALLBACK_DOT_RGB`]. Publiczna (nie tylko dla UI),
/// żeby warstwa audio (`sound::devices`, przy budowaniu zdarzeń kierunku)
/// mogła kolorować zdarzenia bez zależności od `ui::radar_egui`.
pub fn resolve_color(rules: &[ColorRule], energies: &[f32]) -> (u8, u8, u8) {
    dominant_rule_index(energies)
        .and_then(|i| rules.get(i))
        .map(|r| (r.color[0], r.color[1], r.color[2]))
        .unwrap_or(FALLBACK_DOT_RGB)
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
        let freq = 60.0; // wyraźnie w domyślnym paśmie "Sub-bas" (0-100 Hz)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = rule_energies(&samples, sample_rate, &default_color_rules());
        assert_eq!(dominant_rule_index(&energies), Some(0));
    }

    #[test]
    fn high_tone_dominates_the_high_default_rule() {
        let sample_rate = 48_000.0;
        let freq = 8000.0; // wyraźnie w domyślnym paśmie "Wysokie" (6-20 kHz)
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * freq).sin())
            .collect();
        let energies = rule_energies(&samples, sample_rate, &default_color_rules());
        assert_eq!(dominant_rule_index(&energies), Some(4));
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
        assert_eq!(dominant_rule_index(&per_channel[1]), Some(4));
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
    fn weight_lets_a_quieter_high_band_win_over_a_louder_low_band() {
        let sample_rate = 48_000.0;
        // Ton niski o dużej amplitudzie + ton wysoki o mniejszej amplitudzie,
        // zmiksowane w jednym sygnale — jak realny dźwięk z opadającym widmem.
        let samples: Vec<f32> = (0..4800)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.8 * (t * TAU * 60.0).sin() + 0.2 * (t * TAU * 8000.0).sin()
            })
            .collect();

        let unweighted = vec![
            ColorRule::new(0.0, 200.0, [0, 0, 0]),
            ColorRule::new(2000.0, 20_000.0, [0, 0, 0]),
        ];
        let energies = rule_energies(&samples, sample_rate, &unweighted);
        assert_eq!(
            dominant_rule_index(&energies),
            Some(0),
            "bez wagi cichszy wysoki ton nie powinien wygrywać"
        );

        let weighted = vec![
            ColorRule::new(0.0, 200.0, [0, 0, 0]),
            ColorRule::new(2000.0, 20_000.0, [0, 0, 0]).with_weight(10.0),
        ];
        let energies = rule_energies(&samples, sample_rate, &weighted);
        assert_eq!(
            dominant_rule_index(&energies),
            Some(1),
            "z wystarczającą wagą wysoki ton powinien wygrać mimo mniejszej surowej energii"
        );
    }

    #[test]
    fn color_rule_new_defaults_weight_to_one() {
        assert_eq!(ColorRule::new(0.0, 100.0, [0, 0, 0]).weight, 1.0);
    }

    #[test]
    fn default_color_rules_boosts_weight_progressively_with_frequency() {
        let rules = default_color_rules();
        let weights: Vec<f32> = rules.iter().map(|r| r.weight).collect();
        assert_eq!(weights, vec![1.0, 1.0, 1.5, 3.0, 5.0]);
        for pair in weights.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "waga powinna rosnąć (albo zostać taka sama) wraz z częstotliwością pasma"
            );
        }
    }

    #[test]
    fn missing_weight_field_in_saved_json_defaults_to_one() {
        // Symuluje plik zapisany przed dodaniem pola `weight` do `ColorRule`.
        let json = r#"{"freq_min":0.0,"freq_max":200.0,"color":[255,90,90]}"#;
        let rule: ColorRule = serde_json::from_str(json).unwrap();
        assert_eq!(rule.weight, 1.0);
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
    fn default_color_rules_has_at_least_five_rules_covering_full_audible_range_without_gaps() {
        let rules = default_color_rules();
        assert!(rules.len() >= 5, "oczekiwano co najmniej 5 pasm domyślnych");
        assert_eq!(rules[0].freq_min, 0.0);
        assert_eq!(rules.last().unwrap().freq_max, 20_000.0);
        for pair in rules.windows(2) {
            assert_eq!(pair[0].freq_max, pair[1].freq_min, "luka/nakładka między domyślnymi pasmami");
        }
    }

    #[test]
    fn inverted_or_empty_range_never_produces_energy() {
        // Regresja: reguła z min >= max dawała niezerową energię (różnica dwóch
        // filtrów), więc mogła wygrywać kolor kropek mimo że nic nie słychać.
        let sample_rate = 48_000.0;
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 / sample_rate * TAU * 1000.0).sin())
            .collect();
        let rules = vec![
            ColorRule::new(5000.0, 200.0, [1, 0, 0]),
            ColorRule::new(1000.0, 1000.0, [2, 0, 0]),
        ];
        let energies = rule_energies(&samples, sample_rate, &rules);
        assert_eq!(energies, vec![0.0, 0.0]);
    }
}
