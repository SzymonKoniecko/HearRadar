//! Wyznaczanie kierunku i intensywności dźwięku na podstawie poziomów
//! poszczególnych kanałów audio.
//!
//! Moduł jest celowo pozbawiony zależności od `cpal`/strumieni — operuje
//! wyłącznie na znormalizowanych próbkach (`f32`, zakres -1.0..=1.0), dzięki
//! czemu cała logika jest w 100% testowalna bez fizycznego urządzenia audio.
//!
//! Zakładany układ kanałów (kolejność interleaved) odpowiada typowej
//! konwencji WASAPI/ALSA/FFmpeg:
//! - 2.0: `[L, R]`
//! - 5.1: `[FL, FR, C, LFE, RL, RR]`
//! - 7.1: `[FL, FR, C, LFE, RL, RR, SL, SR]`

/// Rola kanału w układzie głośnikowym.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelRole {
    FrontLeft,
    FrontRight,
    Center,
    /// Kanał niskotonowy (subwoofer) — nie niesie informacji kierunkowej.
    Lfe,
    RearLeft,
    RearRight,
    SideLeft,
    SideRight,
}

impl ChannelRole {
    /// Kąt kierunkowy głośnika w stopniach.
    ///
    /// Konwencja: `0°` = przód, `90°` = prawo, `-90°` = lewo, `±180°` = tył.
    /// `None` oznacza brak informacji kierunkowej (LFE).
    pub fn angle_degrees(self) -> Option<f32> {
        use ChannelRole::*;
        match self {
            Center => Some(0.0),
            FrontLeft | SideLeft => Some(-90.0),
            FrontRight | SideRight => Some(90.0),
            RearLeft => Some(-135.0),
            RearRight => Some(135.0),
            Lfe => None,
        }
    }

    /// Kąt kropki na tarczy radaru (osobna konwencja od [`Self::angle_degrees`]).
    ///
    /// Tu każdy kanał ma WŁASNY, odrębny kierunek — w przeciwieństwie do
    /// `angle_degrees`, gdzie front i side dzielą ten sam kąt (co jest
    /// właściwe dla sumowania wektorowego w [`compute_direction`], ale
    /// nieprawidłowe dla radaru 7.1, gdzie front/side/rear muszą być
    /// wizualnie rozróżnialne). Konwencja: `0°` = przód/góra ekranu,
    /// kąty rosną zgodnie z ruchem wskazówek zegara (prawo = dodatnie).
    /// `None` oznacza brak kierunku (LFE — pokazywany osobnym wskaźnikiem).
    pub fn radar_angle_degrees(self) -> Option<f32> {
        use ChannelRole::*;
        match self {
            Center => Some(0.0),
            FrontLeft => Some(-45.0),
            FrontRight => Some(45.0),
            SideLeft => Some(-90.0),
            SideRight => Some(90.0),
            RearLeft => Some(-135.0),
            RearRight => Some(135.0),
            Lfe => None,
        }
    }
}

/// Zwraca standardowy układ kanałów dla podanej ich liczby.
///
/// - `1` -> mono (traktowany jako kanał centralny)
/// - `2` -> stereo 2.0 (L, R)
/// - `6` -> 5.1 (FL, FR, C, LFE, RL, RR)
/// - `8` -> 7.1 (FL, FR, C, LFE, RL, RR, SL, SR)
/// - inne -> pierwsze dwa kanały traktowane jak stereo, reszta pomijana
pub fn channel_layout(channels: usize) -> &'static [ChannelRole] {
    use ChannelRole::*;
    match channels {
        1 => &[Center],
        2 => &[FrontLeft, FrontRight],
        6 => &[FrontLeft, FrontRight, Center, Lfe, RearLeft, RearRight],
        8 => &[
            FrontLeft, FrontRight, Center, Lfe, RearLeft, RearRight, SideLeft, SideRight,
        ],
        _ => &[FrontLeft, FrontRight],
    }
}

/// Statystyki poziomu sygnału dla każdego kanału z bufora próbek.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelStats {
    /// RMS (root-mean-square) każdego kanału — używane do wyznaczenia kierunku.
    pub rms: Vec<f32>,
    /// Wartość szczytowa (peak) każdego kanału — używane tylko do wyświetlania.
    pub peak: Vec<f32>,
}

/// Oblicza RMS i peak dla każdego kanału na podstawie znormalizowanych,
/// przeplecionych (interleaved) próbek.
///
/// Kluczowe: w przeciwieństwie do poprzedniej implementacji, ta funkcja
/// analizuje WSZYSTKIE kanały, nie tylko pierwsze dwa — dzięki temu działa
/// poprawnie dla 5.1 i 7.1.
pub fn analyze_channels(samples: &[f32], channels: usize) -> ChannelStats {
    let channels = channels.max(1);
    let mut sum_sq = vec![0.0f32; channels];
    let mut peak = vec![0.0f32; channels];
    let mut frame_count = 0usize;

    for frame in samples.chunks_exact(channels) {
        frame_count += 1;
        for (i, &sample) in frame.iter().enumerate() {
            let abs = sample.abs();
            sum_sq[i] += abs * abs;
            if abs > peak[i] {
                peak[i] = abs;
            }
        }
    }

    let rms = if frame_count == 0 {
        vec![0.0; channels]
    } else {
        sum_sq
            .iter()
            .map(|&sq| (sq / frame_count as f32).sqrt())
            .collect()
    };

    ChannelStats { rms, peak }
}

/// Próg ciszy — poniżej sumy poziomów wszystkich kanałów uznajemy, że nie ma
/// wykrywalnego dźwięku.
const SILENCE_THRESHOLD: f32 = 0.002;

/// Mnożnik skalujący średnią energię do zakresu intensywności 0.0..=1.0.
/// Dobrany tak, by dla 2.0 zachować zgodność z poprzednim zachowaniem
/// (`(rms_l + rms_r) * 4.0`), ponieważ `avg_energy * 8.0 == (rms_l+rms_r) * 4.0`
/// gdy kanałów jest dwa.
pub(crate) const INTENSITY_SCALE: f32 = 8.0;

/// Wyznacza kierunek (kąt w stopniach) oraz intensywność (0.0..=1.0) dźwięku
/// na podstawie poziomów RMS poszczególnych kanałów.
///
/// `channel_rms.len()` decyduje o użytym układzie kanałów (patrz [`channel_layout`]).
/// Kierunek wyznaczany jest metodą sumy wektorów jednostkowych ważonych
/// poziomem energii każdego kanału (podobne do VBAP), co pozwala poprawnie
/// łączyć wkład wielu głośników (np. front-right + side-right wzmacniają się,
/// zamiast się znosić).
///
/// Zwraca `(0.0, 0.0)` jeśli sygnał jest ciszej niż [`SILENCE_THRESHOLD`].
pub fn compute_direction(channel_rms: &[f32]) -> (f32, f32) {
    let layout = channel_layout(channel_rms.len());

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut total_energy = 0.0f32;

    for (&level, role) in channel_rms.iter().zip(layout.iter()) {
        let level = level.max(0.0);
        total_energy += level;

        if let Some(angle_deg) = role.angle_degrees() {
            let rad = angle_deg.to_radians();
            sum_x += level * rad.sin();
            sum_y += level * rad.cos();
        }
    }

    if total_energy < SILENCE_THRESHOLD {
        return (0.0, 0.0);
    }

    let angle = if sum_x.abs() > f32::EPSILON || sum_y.abs() > f32::EPSILON {
        sum_x.atan2(sum_y).to_degrees()
    } else {
        0.0
    };

    let avg_energy = total_energy / channel_rms.len().max(1) as f32;
    let intensity = (avg_energy * INTENSITY_SCALE).clamp(0.0, 1.0);

    (angle, intensity)
}

/// Wygładza kąt (konwencja stopni z [`compute_direction`]: `(-180, 180]`,
/// `0°` = przód) o `factor` w stronę `target`, idąc NAJKRÓTSZĄ drogą po
/// okręgu — w przeciwieństwie do zwykłego `current * (1-factor) + target *
/// factor`, które przy przejściu np. z `-135°` (tył-lewo) do `+135°`
/// (tył-prawo) leciałoby przez `0°` (PRZÓD), bo arytmetycznie `-135 → 135`
/// w górę to “krótsza” zmiana liczby niż `-135 → -180/180 → 135`, mimo że
/// rzeczywista różnica kątowa w drugą stronę (przez TYŁ) to tylko 90°, a nie
/// 270°. To właśnie ten błąd sprawiał, że wskaźnik kierunku sumarycznego
/// (patrz `ui::app::smooth_display_direction`) przy przejściu dźwięku z
/// tyłu-lewo do tyłu-prawo okrążał tarczę przez przód zamiast iść wprost
/// przez tył.
pub fn smooth_angle_deg(current: f32, target: f32, factor: f32) -> f32 {
    let diff = wrap_angle_deg(target - current);
    wrap_angle_deg(current + diff * factor)
}

/// Zawija kąt w stopniach do zakresu `(-180, 180]`.
fn wrap_angle_deg(angle: f32) -> f32 {
    ((angle + 180.0).rem_euclid(360.0)) - 180.0
}

// ── Detekcja niezależnych zdarzeń dźwiękowych (onsetów) ────────────────────
//
// W przeciwieństwie do [`compute_direction`] (JEDEN, sumaryczny wektor dla
// całej ramki — dwa przeciwstawne źródła o podobnej energii potrafią się
// wzajemnie znieść: `sum_x≈0, sum_y≈0`), poniższe funkcje wykrywają KAŻDY
// nagły skok energii na KAŻDYM kanale osobno, więc silnik z lewej i wybuch z
// prawej dają dwa niezależne zdarzenia zamiast jednego mylącego/zanikającego
// wskaźnika "z przodu".

/// Próg mnożnikowy nagłego skoku energii nad średnią kroczącą kanału, przy
/// którym zgłaszamy nowy onset (analogicznie do `TRANSIENT_RATIO` używanego
/// przy kropkach kanałów w `ui::radar_egui`, ale to osobna, niezależna stała —
/// onsety karmią osobny mechanizm wizualny).
pub const TRANSIENT_RATIO: f32 = 2.5;

/// Poniżej tego poziomu RMS w ogóle nie rozważamy zgłoszenia onsetu — szum
/// tła nie powinien generować wskaźników kierunku.
const ONSET_MIN_ENERGY: f32 = 0.02;

/// Współczynnik wygładzania średniej kroczącej energii per kanał (mniejszy =
/// wolniejsza, stabilniejsza średnia).
const ONSET_EMA_FACTOR: f32 = 0.1;

/// Maksymalna odległość kątowa (stopnie) sąsiedniego kanału, którego energia
/// wpływa na [`onset_angle_deg`] danego kanału. Kanały leżące PO PRZECIWNEJ
/// stronie tarczy (`> 90°`) nigdy nie powinny przeciągać kąta na swoją stronę
/// — stąd próg dokładnie w połowie okręgu.
const NEIGHBOR_MAX_ANGLE_DIFF_DEG: f32 = 90.0;

/// Pojedyncze, niezależne zdarzenie dźwiękowe wykryte na jednym kanale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelOnset {
    /// Indeks kanału (w `channel_rms`/`channel_layout`), na którym wykryto onset.
    pub channel_index: usize,
    /// Poziom RMS kanału w chwili wykrycia (0.0..=1.0).
    pub energy: f32,
    /// Płynny kąt onsetu w stopniach (patrz [`onset_angle_deg`]) — w
    /// przeciwieństwie do sztywnej pozycji głośnika, uwzględnia energię
    /// kątowych sąsiadów.
    pub angle_deg: f32,
}

/// Wykrywa nagłe skoki energii (onsety) na każdym kanale osobno, porównując
/// bieżący RMS ze średnią kroczącą tego kanału (stan przekazywany z zewnątrz
/// w `ema_state`, żeby funkcja zostawała czystą/testowalną bez żywego audio —
/// wywołujący jest odpowiedzialny za trzymanie tego samego `Vec` między
/// kolejnymi wywołaniami). Kanały bez kierunku (LFE) są pomijane.
///
/// `ema_state` jest automatycznie dopasowywany do rozmiaru `channel_rms`,
/// jeśli liczba kanałów strumienia się zmieni (np. przełączenie urządzenia).
pub fn detect_onsets(channel_rms: &[f32], ema_state: &mut Vec<f32>) -> Vec<ChannelOnset> {
    if ema_state.len() != channel_rms.len() {
        *ema_state = vec![0.0; channel_rms.len()];
    }

    let layout = channel_layout(channel_rms.len());
    let mut onsets = Vec::new();

    for (i, &level) in channel_rms.iter().enumerate() {
        let level = level.max(0.0);
        let has_direction = layout
            .get(i)
            .and_then(|role| role.radar_angle_degrees())
            .is_some();

        if !has_direction {
            ema_state[i] = ema_state[i] * (1.0 - ONSET_EMA_FACTOR) + level * ONSET_EMA_FACTOR;
            continue;
        }

        let prev_ema = ema_state[i];
        let is_onset = level > ONSET_MIN_ENERGY && level > prev_ema * TRANSIENT_RATIO + 0.02;
        ema_state[i] = prev_ema * (1.0 - ONSET_EMA_FACTOR) + level * ONSET_EMA_FACTOR;

        if is_onset {
            onsets.push(ChannelOnset {
                channel_index: i,
                energy: level,
                angle_deg: onset_angle_deg(channel_rms, i),
            });
        }
    }

    onsets
}

/// Płynny kąt onsetu wykrytego na `channel_index`: średnia ważona energią
/// własnego kąta kanału i kątów jego kątowych sąsiadów (odległość kątowa
/// `<= `[`NEIGHBOR_MAX_ANGLE_DIFF_DEG`]`°`, NIE sąsiedztwo w tablicy
/// `channel_layout` — w układzie 2.0 jedynym sąsiadem indeksowym L jest R,
/// czyli kanał dokładnie naprzeciwko, więc głośny R przeciągałby kąt L na
/// swoją stronę, gdyby liczyć po indeksie). Dzięki temu głośny, sąsiedni
/// kanał płynnie przesuwa kąt w swoją stronę, ale kanał po przeciwnej stronie
/// tarczy nigdy nie wpływa na wynik.
pub fn onset_angle_deg(channel_rms: &[f32], channel_index: usize) -> f32 {
    let layout = channel_layout(channel_rms.len());
    let Some(own_role) = layout.get(channel_index).copied() else {
        return 0.0;
    };
    let Some(own_angle) = own_role.radar_angle_degrees() else {
        return 0.0;
    };

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut total_energy = 0.0f32;

    for (i, role) in layout.iter().enumerate() {
        let Some(angle_deg) = role.radar_angle_degrees() else {
            continue;
        };
        if i != channel_index && wrap_angle_deg(angle_deg - own_angle).abs() > NEIGHBOR_MAX_ANGLE_DIFF_DEG {
            continue;
        }
        let energy = channel_rms.get(i).copied().unwrap_or(0.0).max(0.0);
        let rad = angle_deg.to_radians();
        sum_x += energy * rad.sin();
        sum_y += energy * rad.cos();
        total_energy += energy;
    }

    if total_energy < f32::EPSILON {
        return own_angle;
    }

    sum_x.atan2(sum_y).to_degrees()
}

/// Zwraca poglądową nazwę układu kanałów, np. "2.0", "5.1", "7.1".
/// Dla nietypowej liczby kanałów zwraca np. "4ch".
pub fn layout_name(channels: usize) -> String {
    match channels {
        1 => "1.0".to_string(),
        2 => "2.0".to_string(),
        6 => "5.1".to_string(),
        8 => "7.1".to_string(),
        n => format!("{n}ch"),
    }
}

/// Krótkie etykiety kanałów (w kolejności zgodnej z [`channel_layout`]) do
/// wyświetlania/logowania, np. `["L", "R"]` dla 2.0 albo
/// `["FL", "FR", "C", "LFE", "RL", "RR"]` dla 5.1.
pub fn channel_labels(channels: usize) -> &'static [&'static str] {
    match channels {
        1 => &["C"],
        2 => &["L", "R"],
        6 => &["FL", "FR", "C", "LFE", "RL", "RR"],
        8 => &["FL", "FR", "C", "LFE", "RL", "RR", "SL", "SR"],
        _ => &["L", "R"],
    }
}

/// Formatuje poziomy poszczególnych kanałów do czytelnego, poglądowego
/// stringa, np. `"2.0: [L:10, R:20]"` albo
/// `"5.1: [FL:5, FR:12, C:0, LFE:3, RL:0, RR:0]"`.
///
/// Wartości to poziom RMS przeskalowany poglądowo do przedziału 0..=100
/// (nie mylić z `intensity` z [`compute_direction`], które ma inne skalowanie).
pub fn format_channel_levels(rms: &[f32]) -> String {
    let labels = channel_labels(rms.len());

    let parts: Vec<String> = rms
        .iter()
        .zip(labels.iter())
        .map(|(&level, label)| {
            let percent = (level.clamp(0.0, 1.0) * 100.0).round() as i32;
            format!("{label}:{percent}")
        })
        .collect();

    format!("{}: [{}]", layout_name(rms.len()), parts.join(", "))
}

/// Konwertuje próbkę `i16` do znormalizowanego zakresu `-1.0..=1.0`.
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

/// Konwertuje próbkę `u16` (bez znaku, środek = cisza) do zakresu `-1.0..=1.0`.
pub fn u16_to_f32(sample: u16) -> f32 {
    (sample as f32 / u16::MAX as f32 - 0.5) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_angle_deg_from_rear_left_to_rear_right_goes_through_the_back_not_the_front() {
        // Dokładnie zgłoszony błąd: dźwięk przechodzi z tyłu-lewo (-135°) do
        // tyłu-prawo (+135°) — poprawnie powinien iść przez TYŁ (±180°),
        // czyli każdy krok powinien mieć |kąt| >= 135° (nigdy nie zbliżać się
        // do 0° / przodu).
        let mut angle = -135.0f32;
        for _ in 0..40 {
            angle = smooth_angle_deg(angle, 135.0, 0.25);
            assert!(
                angle.abs() >= 135.0 - 1e-3,
                "kąt {angle} przeszedł przez przód zamiast przez tył"
            );
        }
        assert!((angle - 135.0).abs() < 1e-2);
    }

    #[test]
    fn smooth_angle_deg_short_hop_not_crossing_the_seam_moves_directly() {
        // -10° -> 10°: najkrótsza droga jest wprost przez 0°, więc TU
        // przechodzenie przez przód jest poprawne.
        let angle = smooth_angle_deg(-10.0, 10.0, 0.5);
        assert!((angle - 0.0).abs() < 1e-3);
    }

    #[test]
    fn smooth_angle_deg_reaches_target_and_stays_there() {
        let angle = smooth_angle_deg(90.0, 90.0, 0.25);
        assert!((angle - 90.0).abs() < 1e-3);
    }

    #[test]
    fn smooth_angle_deg_result_stays_in_valid_range() {
        let angle = smooth_angle_deg(179.0, -179.0, 0.5);
        assert!((-180.0..=180.0).contains(&angle));
        // -179 i 179 są 2° od siebie przez szew ±180° — powinno wylądować
        // blisko ±180, nie w okolicy 0.
        assert!(angle.abs() > 170.0, "angle={angle}");
    }

    const EPS: f32 = 1e-3;

    fn assert_close(a: f32, b: f32, label: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{label}: oczekiwano {b}, otrzymano {a}"
        );
    }

    // ---------- channel_layout ----------

    #[test]
    fn layout_mono_is_center() {
        assert_eq!(channel_layout(1), &[ChannelRole::Center]);
    }

    #[test]
    fn layout_stereo_2_0() {
        assert_eq!(
            channel_layout(2),
            &[ChannelRole::FrontLeft, ChannelRole::FrontRight]
        );
    }

    #[test]
    fn layout_5_1_has_six_roles_in_order() {
        use ChannelRole::*;
        assert_eq!(
            channel_layout(6),
            &[FrontLeft, FrontRight, Center, Lfe, RearLeft, RearRight]
        );
    }

    #[test]
    fn layout_7_1_extends_5_1_with_side_channels() {
        use ChannelRole::*;
        assert_eq!(
            channel_layout(8),
            &[
                FrontLeft, FrontRight, Center, Lfe, RearLeft, RearRight, SideLeft, SideRight
            ]
        );
    }

    #[test]
    fn layout_unknown_channel_count_falls_back_to_stereo() {
        assert_eq!(
            channel_layout(4),
            &[ChannelRole::FrontLeft, ChannelRole::FrontRight]
        );
    }

    #[test]
    fn lfe_has_no_direction() {
        assert_eq!(ChannelRole::Lfe.angle_degrees(), None);
    }

    #[test]
    fn radar_angle_distinguishes_front_side_rear() {
        use ChannelRole::*;
        assert_eq!(FrontLeft.radar_angle_degrees(), Some(-45.0));
        assert_eq!(SideLeft.radar_angle_degrees(), Some(-90.0));
        assert_eq!(RearLeft.radar_angle_degrees(), Some(-135.0));
        assert_eq!(FrontRight.radar_angle_degrees(), Some(45.0));
        assert_eq!(SideRight.radar_angle_degrees(), Some(90.0));
        assert_eq!(RearRight.radar_angle_degrees(), Some(135.0));
        assert_eq!(Center.radar_angle_degrees(), Some(0.0));
        assert_eq!(Lfe.radar_angle_degrees(), None);
    }

    // ---------- analyze_channels (RMS/peak per-channel) ----------

    #[test]
    fn analyze_channels_stereo_exact_rms() {
        // L: stała amplituda 1.0, R: stała amplituda 0.5
        let samples = [1.0, 0.5, -1.0, -0.5, 1.0, 0.5];
        let stats = analyze_channels(&samples, 2);
        assert_close(stats.rms[0], 1.0, "rms L");
        assert_close(stats.rms[1], 0.5, "rms R");
        assert_close(stats.peak[0], 1.0, "peak L");
        assert_close(stats.peak[1], 0.5, "peak R");
    }

    #[test]
    fn analyze_channels_reads_all_six_channels_for_5_1() {
        // Dźwięk tylko na kanale RL (indeks 4) — sprawdza, że kanały 2..5
        // NIE są ignorowane (to był błąd w poprzedniej implementacji).
        let frame = [0.0, 0.0, 0.0, 0.0, 0.8, 0.0];
        let samples: Vec<f32> = frame.iter().cycle().take(frame.len() * 3).copied().collect();
        let stats = analyze_channels(&samples, 6);
        assert_close(stats.rms[4], 0.8, "rms RL (5.1)");
        for (i, &v) in stats.rms.iter().enumerate() {
            if i != 4 {
                assert_close(v, 0.0, &format!("rms kanału {i} powinno być ciche"));
            }
        }
    }

    #[test]
    fn analyze_channels_empty_input_returns_zeros() {
        let stats = analyze_channels(&[], 2);
        assert_eq!(stats.rms, vec![0.0, 0.0]);
        assert_eq!(stats.peak, vec![0.0, 0.0]);
    }

    // ---------- compute_direction: 2.0 ----------

    #[test]
    fn stereo_2_0_pure_right() {
        let (angle, intensity) = compute_direction(&[0.0, 0.1]);
        assert_close(angle, 90.0, "kąt (pure right)");
        assert_close(intensity, 0.4, "intensywność (pure right)");
    }

    #[test]
    fn stereo_2_0_pure_left() {
        let (angle, _) = compute_direction(&[0.1, 0.0]);
        assert_close(angle, -90.0, "kąt (pure left)");
    }

    #[test]
    fn stereo_2_0_centered_sound_is_front() {
        let (angle, intensity) = compute_direction(&[0.1, 0.1]);
        assert_close(angle, 0.0, "kąt (L=R, dźwięk na wprost)");
        assert_close(intensity, 0.8, "intensywność (L=R)");
    }

    #[test]
    fn stereo_2_0_slight_pan_right_is_between_front_and_right() {
        // L:0.05, R:0.1 — większa energia po prawej, kąt powinien być > 0° i ≤ 90°
        let (angle, _) = compute_direction(&[0.05, 0.1]);
        assert!(
            angle > 0.0 && angle <= 90.0 + f32::EPSILON * 100.0,
            "lekkie przesunięcie w prawo powinno dać kąt w (0, 90], otrzymano {angle}"
        );
    }

    #[test]
    fn stereo_2_0_silence_gives_zero() {
        assert_eq!(compute_direction(&[0.0, 0.0]), (0.0, 0.0));
    }

    #[test]
    fn stereo_2_0_matches_legacy_intensity_scaling() {
        // Poprzedni wzór: intensity = ((rms_l + rms_r) * 4.0).min(1.0)
        let rms_l = 0.05f32;
        let rms_r = 0.03f32;
        let legacy = ((rms_l + rms_r) * 4.0).min(1.0);
        let (_, intensity) = compute_direction(&[rms_l, rms_r]);
        assert_close(intensity, legacy, "zgodność skalowania intensywności z 2.0");
    }

    // ---------- compute_direction: 5.1 ----------

    #[test]
    fn surround_5_1_equal_rear_channels_is_directly_behind() {
        // [FL, FR, C, LFE, RL, RR]
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.2, 0.2]);
        assert_close(angle.abs(), 180.0, "równe RL/RR = dźwięk z tyłu (±180°)");
    }

    #[test]
    fn surround_5_1_rear_left_only() {
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.2, 0.0]);
        assert_close(angle, -135.0, "sam RL -> tył-lewo");
    }

    #[test]
    fn surround_5_1_rear_right_only() {
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]);
        assert_close(angle, 135.0, "sam RR -> tył-prawo");
    }

    #[test]
    fn surround_5_1_center_channel_is_front() {
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.15, 0.0, 0.0, 0.0]);
        assert_close(angle, 0.0, "sam kanał centralny -> przód");
    }

    #[test]
    fn surround_5_1_lfe_only_has_no_direction_but_has_intensity() {
        let (angle, intensity) = compute_direction(&[0.0, 0.0, 0.0, 0.3, 0.0, 0.0]);
        assert_close(angle, 0.0, "sam LFE nie niesie kierunku");
        assert!(intensity > 0.0, "LFE nadal powinien podnosić intensywność");
    }

    #[test]
    fn surround_5_1_front_beats_rear_when_louder() {
        // FR=0.3 (90°) vs RL=0.05 (-135°) — FR dominuje, kąt po prawej stronie (> 0°).
        // FR leży dokładnie na 90°, a składowa RL lekko przesuwa sumaryczny wektor
        // powyżej 90° — dlatego sprawdzamy tylko że kąt jest dodatni (prawa strona).
        let (angle, _) = compute_direction(&[0.0, 0.3, 0.0, 0.0, 0.05, 0.0]);
        assert!(
            angle > 0.0,
            "dominujący FR powinien dać kąt po prawej stronie (> 0°), otrzymano {angle}"
        );
    }

    #[test]
    fn surround_5_1_silence_across_all_channels() {
        assert_eq!(
            compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (0.0, 0.0)
        );
    }

    // ---------- compute_direction: 7.1 ----------

    #[test]
    fn surround_7_1_side_right_only() {
        // [FL, FR, C, LFE, RL, RR, SL, SR]
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15]);
        assert_close(angle, 90.0, "sam SR -> prawo (tak jak FR)");
    }

    #[test]
    fn surround_7_1_side_left_only() {
        let (angle, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0]);
        assert_close(angle, -90.0, "sam SL -> lewo (tak jak FL)");
    }

    #[test]
    fn surround_7_1_front_and_side_right_reinforce_each_other() {
        let (angle, intensity_combined) =
            compute_direction(&[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]);
        assert_close(angle, 90.0, "FR + SR wzmacniają się w tym samym kierunku");

        let (_, intensity_single) = compute_direction(&[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            intensity_combined > intensity_single,
            "dwa zgodne kierunkowo kanały powinny dać wyższą intensywność niż jeden"
        );
    }

    #[test]
    fn surround_7_1_distinguishes_side_from_rear() {
        let (angle_side, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]);
        let (angle_rear, _) = compute_direction(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]);
        assert!(
            (angle_side - angle_rear).abs() > 30.0,
            "boczny prawy (90°) i tylny prawy (135°) muszą być wyraźnie różne"
        );
    }

    #[test]
    fn surround_7_1_full_mix_all_channels_active() {
        // Realistyczny miks: coś gra wszędzie, ale dominuje przód-prawo.
        let rms = [0.05, 0.4, 0.1, 0.2, 0.05, 0.05, 0.05, 0.1];
        let (angle, intensity) = compute_direction(&rms);
        assert!(
            angle > 0.0 && angle < 90.0,
            "dominacja FR powinna dać kąt w (0, 90), otrzymano {angle}"
        );
        assert!(intensity > 0.0 && intensity <= 1.0);
    }

    // ---------- konwersje formatów próbek ----------

    #[test]
    fn i16_conversion_bounds() {
        assert_close(i16_to_f32(i16::MAX), 1.0, "i16::MAX -> 1.0");
        assert_close(i16_to_f32(-i16::MAX), -1.0, "-i16::MAX -> -1.0");
        assert_close(i16_to_f32(0), 0.0, "0 -> 0.0");
    }

    #[test]
    fn u16_conversion_bounds() {
        assert_close(u16_to_f32(0), -1.0, "0 -> -1.0 (cisza/min)");
        assert_close(u16_to_f32(u16::MAX), 1.0, "u16::MAX -> 1.0");
        assert_close(u16_to_f32(32768), 0.0, "środek zakresu -> ~0.0");
    }

    // ---------- format_channel_levels / layout_name / channel_labels ----------

    #[test]
    fn layout_name_matches_known_configurations() {
        assert_eq!(layout_name(1), "1.0");
        assert_eq!(layout_name(2), "2.0");
        assert_eq!(layout_name(6), "5.1");
        assert_eq!(layout_name(8), "7.1");
        assert_eq!(layout_name(4), "4ch");
    }

    #[test]
    fn channel_labels_match_layout() {
        assert_eq!(channel_labels(2), &["L", "R"]);
        assert_eq!(
            channel_labels(6),
            &["FL", "FR", "C", "LFE", "RL", "RR"]
        );
        assert_eq!(
            channel_labels(8),
            &["FL", "FR", "C", "LFE", "RL", "RR", "SL", "SR"]
        );
    }

    #[test]
    fn format_channel_levels_stereo() {
        let formatted = format_channel_levels(&[0.10, 0.20]);
        assert_eq!(formatted, "2.0: [L:10, R:20]");
    }

    #[test]
    fn format_channel_levels_5_1() {
        let formatted = format_channel_levels(&[0.05, 0.12, 0.0, 0.03, 0.0, 0.0]);
        assert_eq!(formatted, "5.1: [FL:5, FR:12, C:0, LFE:3, RL:0, RR:0]");
    }

    #[test]
    fn format_channel_levels_7_1() {
        let formatted =
            format_channel_levels(&[0.5, 0.5, 0.1, 0.02, 0.0, 0.0, 0.3, 0.3]);
        assert_eq!(
            formatted,
            "7.1: [FL:50, FR:50, C:10, LFE:2, RL:0, RR:0, SL:30, SR:30]"
        );
    }

    #[test]
    fn format_channel_levels_clamps_out_of_range_values() {
        // RMS teoretycznie nie powinno przekraczać 1.0, ale funkcja ma być odporna.
        let formatted = format_channel_levels(&[1.5, -0.2]);
        assert_eq!(formatted, "2.0: [L:100, R:0]");
    }

    // ---------- detect_onsets / onset_angle_deg ----------

    #[test]
    fn opposite_channels_with_similar_energy_yield_two_separate_onsets_not_one_cancelled() {
        // Dokładnie scenariusz "silnik z lewej + wybuch z prawej": SL (-90°) i
        // SR (90°) leżą dokładnie naprzeciw siebie i mają podobną energię, więc
        // compute_direction dałby prawie zerowy wektor sumaryczny. detect_onsets
        // powinien zamiast tego zgłosić DWA niezależne zdarzenia, każde na
        // swoim kanale. [FL, FR, C, LFE, RL, RR, SL, SR]
        let mut ema = vec![0.0; 8];
        let onsets = detect_onsets(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3], &mut ema);
        let left = onsets.iter().find(|o| o.channel_index == 6).unwrap();
        let right = onsets.iter().find(|o| o.channel_index == 7).unwrap();
        assert_close(left.angle_deg, -90.0, "onset lewego kanału (SL)");
        assert_close(right.angle_deg, 90.0, "onset prawego kanału (SR)");
    }

    #[test]
    fn detect_onsets_ignores_steady_energy_without_a_sudden_jump() {
        let mut ema = vec![0.0, 0.0];
        // Kilka ramek o stałym poziomie — po "rozgrzaniu" średniej kroczącej
        // dalsze ramki o tym samym poziomie nie powinny być onsetami.
        for _ in 0..50 {
            detect_onsets(&[0.2, 0.2], &mut ema);
        }
        let onsets = detect_onsets(&[0.2, 0.2], &mut ema);
        assert!(onsets.is_empty(), "stały poziom nie powinien generować onsetów");
    }

    #[test]
    fn detect_onsets_silence_yields_nothing() {
        let mut ema = vec![0.0, 0.0];
        assert!(detect_onsets(&[0.0, 0.0], &mut ema).is_empty());
    }

    #[test]
    fn detect_onsets_skips_lfe_channel() {
        // [FL, FR, C, LFE, RL, RR]
        let mut ema = vec![0.0; 6];
        let onsets = detect_onsets(&[0.0, 0.0, 0.0, 0.5, 0.0, 0.0], &mut ema);
        assert!(
            onsets.iter().all(|o| o.channel_index != 3),
            "LFE nie niesie kierunku, nie powinien generować onsetu"
        );
    }

    #[test]
    fn detect_onsets_resizes_ema_state_for_new_channel_count() {
        let mut ema = vec![0.0, 0.0];
        let _ = detect_onsets(&[0.1, 0.1], &mut ema);
        assert_eq!(ema.len(), 2);
        // Zmiana urządzenia -> inna liczba kanałów; nie powinno panikować.
        let _ = detect_onsets(&[0.0; 6], &mut ema);
        assert_eq!(ema.len(), 6);
    }

    #[test]
    fn onset_angle_deg_pure_channel_matches_its_own_fixed_angle() {
        // [FL, FR, C, LFE, RL, RR, SL, SR] -> SL=index6 (-90°)
        let angle = onset_angle_deg(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0], 6);
        assert_close(angle, -90.0, "onset tylko na SL -> kąt SL");
    }

    #[test]
    fn onset_angle_deg_interpolates_smoothly_towards_a_louder_neighbor() {
        // [FL, FR, C, LFE, RL, RR, SL, SR] -> SR=index7 (90°), RR=index5 (135°)
        let angle_quiet_neighbor = onset_angle_deg(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.3], 7);
        let angle_loud_neighbor = onset_angle_deg(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.3], 7);

        assert!(
            angle_quiet_neighbor > 90.0 && angle_quiet_neighbor < angle_loud_neighbor,
            "cichy sąsiad powinien dać kąt bliżej 90°: {angle_quiet_neighbor}"
        );
        assert!(
            angle_loud_neighbor < 135.0,
            "kąt nie powinien przekroczyć kąta sąsiada (135°): {angle_loud_neighbor}"
        );
    }

    #[test]
    fn onset_angle_deg_never_pulled_by_the_opposite_side_of_the_dial() {
        // [FL, FR, C, LFE, RL, RR, SL, SR] -> SL=index6 (-90°), SR=index7 (90°),
        // dokładnie naprzeciw siebie (180° różnicy). Nawet bardzo głośny SR
        // nie powinien przesunąć kąta onsetu na SL.
        let angle = onset_angle_deg(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.9], 6);
        assert_close(angle, -90.0, "SL nie powinien być ciągnięty przez przeciwległy SR");
    }
}
