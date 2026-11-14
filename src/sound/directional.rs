//! Rozdzielanie kierunkowe dla trybu jednousznego: dzieli dźwięk na część, która przychodzi
//! z przeciwnej strony niż grane ucho, i resztę (swoja strona, środek). Style kierunkowe
//! (`MonoCueStyle::is_directional`) nakładają wskazówkę tylko na tę pierwszą część. Dzięki temu
//! środek (np. własna broń) i swoja strona zostają czyste, a dźwięk z drugiej strony dostaje
//! wskazówkę w całości, bez rozmycia przez swoją kopię w drugim kanale.
//!
//! Zwykłe style działają na całym KANALE przeciwnym. Gra prawie nigdy nie daje dźwięku tylko
//! do jednego kanału: w nagraniach z gry różnica poziomów L/P głośnych dźwięków to zwykle
//! 2–8 dB, a drugi kanał niesie kopię tego samego dźwięku. Dlatego tam efekt dostawało
//! wszystko, środek trochę, strona przeciwna trochę bardziej.
//!
//! Jak działa rozdzielanie (nowa decyzja co ok. 2,7 ms):
//!   1. L i P dzielimy na paski co ok. 94 Hz (STFT: okno analizy 512 próbek, nowa ramka co 128;
//!      okna są asymetryczne, patrz [`analysis_window`], więc wyjście czeka tylko 256 próbek),
//!   2. w każdym pasku porównujemy moc L i P, czyli różnicę poziomów (ILD) w dB,
//!   3. z różnicy liczymy wagę `w`: 0 dla swojej strony i środka (do [`ILD_DEAD_ZONE_DB`]),
//!      rośnie do 1 przy [`ILD_FULL_DB`] w stronę przeciwną,
//!   4. ułamek `w` paska (L+P razem) to część przeciwna, a `1 − w` to reszta. Część przeciwna
//!      dostaje jeszcze wzmocnienie `g`: ustępowanie stronie własnej w paśmie, w którym ta gra
//!      (patrz [`SplitParams::duck_db`]), i opcjonalny cień głowy ([`SplitParams::shadow`]),
//!   5. składamy z powrotem trzy sygnały: część bez zmian, część z cieńszą barwą Oddechu
//!      i resztę.
//!
//! Bez `g` część przeciwna i reszta sumują się dokładnie do L+P, opóźnionego o [`LATENCY`]
//! próbek (ok. 5,3 ms przy 48 kHz). Dźwięki z różnych stron grające naraz zwykle zajmują
//! w danej chwili różne paski, więc każdy trafia do swojej części.
//!
//! Rozdzielanie wykrywa też początek dźwięku z przeciwnej strony ([`SplitSample::fresh`]),
//! żeby wskazówki krótkie (grzebień i podobne) mogły mocno oznaczyć początek, a potem się
//! wycofać.

use std::collections::VecDeque;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::sound::mono_router::{TREMOLO_HPF_HZ, TREMOLO_HZ};

/// Długość okna analizy FFT w próbkach: przy 48 kHz paski co ok. 94 Hz.
pub(crate) const FFT_LEN: usize = 512;
/// Skok między ramkami: ćwierć okna, więc decyzja odświeża się co ok. 2,7 ms.
const HOP: usize = FFT_LEN / 4;
/// Okno syntezy jest niezerowe tylko na ostatnich `SYNTH_LEN` próbkach ramki (patrz
/// [`analysis_window`]), więc wyjście czeka tylko na nie, a nie na całe okno analizy.
const SYNTH_LEN: usize = 2 * HOP;
/// Opóźnienie rozdzielania w próbkach: 256, czyli ok. 5,3 ms przy 48 kHz (dawniej całe okno,
/// 512 próbek, ok. 10,7 ms).
pub(crate) const LATENCY: usize = SYNTH_LEN;
/// Liczba pasków od 0 Hz do połowy częstotliwości próbkowania.
const BINS: usize = FFT_LEN / 2 + 1;
/// Bufor sumowania ramek wyjścia: dwa okna, więc odczyt nigdy nie trafia w zapisywany zakres.
const OLA_LEN: usize = 2 * FFT_LEN;

/// Martwa strefa wokół środka (dB różnicy L/P w stronę przeciwną): poniżej tego progu dźwięk
/// uznajemy za środek albo swoją stronę i nie oznaczamy go wcale.
pub(crate) const ILD_DEAD_ZONE_DB: f32 = 1.5;
/// Różnica L/P (dB), od której dźwięk w całości trafia do części przeciwnej. W nagraniach
/// z gry głośne dźwięki z boku mają w paśmie 1–4 kHz zwykle 3–9 dB, więc 6 dB łapie większość
/// wyraźnie bocznych.
pub(crate) const ILD_FULL_DB: f32 = 6.0;

/// Wygładzanie mocy w paskach (ms), żeby decyzja nie migotała od ramki do ramki.
const POWER_SMOOTH_MS: f32 = 10.0;
/// Narastanie wagi `w` (ms): nowy dźwięk z boku trafia do części przeciwnej prawie od razu.
const WEIGHT_ATTACK_MS: f32 = 5.0;
/// Opadanie wagi `w` (ms): wolniej, żeby wskazówka nie urywała się w krótkich przerwach dźwięku.
const WEIGHT_RELEASE_MS: f32 = 80.0;

/// Dolnoprzepustowy barwy kierunkowej (Hz). Wyżej niż w kanałowym Oddechu (4 kHz), żeby nie
/// zabierać szczegółów krótkich dźwięków z głuchej strony: 8 kHz traci ok. 3 dB zamiast 7 dB.
/// Górnoprzepustowy jak w Oddechu ([`TREMOLO_HPF_HZ`]).
pub(crate) const TIMBRE_LPF_HZ: f32 = 8_000.0;

/// Domyślna głębokość oddechu (0–1) w stylach z oddechem. W trybie „w dół” 0,44 to dołek
/// −5 dB i szczyt na poziomie oryginału.
pub(crate) const DEFAULT_DEPTH: f32 = 0.44;
/// Domyślna siła cieńszej barwy (0–1) w stylach z oddechem.
pub(crate) const DEFAULT_TIMBRE: f32 = 1.0;
/// Najgłębszy dopuszczalny oddech: przy 1,0 dno cyklu byłoby pełną ciszą.
pub(crate) const MAX_DEPTH: f32 = 0.95;
/// Domyślna siła lekkiego brzęczenia w stylu „Oddech + brzęczenie” (0–1). 0,5 to wahanie 6 dB
/// przy 45 Hz: wyraźna chropowatość na krótkich dźwiękach, ale bez schodzenia do ciszy.
pub(crate) const DEFAULT_BUZZ_DEPTH: f32 = 0.5;

/// Ustępowanie stronie własnej: pasma ok. 1/3 oktawy od [`DUCK_LOW_HZ`] do [`DUCK_HIGH_HZ`].
const DUCK_LOW_HZ: f32 = 150.0;
const DUCK_HIGH_HZ: f32 = 12_000.0;
const DUCK_BANDS_PER_OCTAVE: f32 = 3.0;
/// Pasek poza pasmami ustępowania.
const NO_BAND: usize = usize::MAX;
/// Energia pasm narasta w 5 ms, a opada w 150 ms (mniej więcej tyle trwa maskowanie po dźwięku).
const DUCK_ATTACK_MS: f32 = 5.0;
const DUCK_RELEASE_MS: f32 = 150.0;
/// Kiedy ustępować, wg `x` = poziom strony własnej minus przeciwnej w paśmie (dB). Pełne
/// ustępowanie dla `x` od −15 do 0 dB: strona własna ma co ratować i grozi jej maskowanie.
/// Zero poniżej −20 dB (to tylko tło, przeciwnej nie trzeba ruszać) i powyżej +6 dB (strona
/// własna i tak wygrywa).
const DUCK_OFF_LOW_DB: f32 = -20.0;
const DUCK_FULL_LOW_DB: f32 = -15.0;
const DUCK_FULL_HIGH_DB: f32 = 0.0;
const DUCK_OFF_HIGH_DB: f32 = 6.0;
/// Domyślna i największa głębokość ustępowania (dB).
pub(crate) const DEFAULT_DUCK_DB: f32 = 6.0;
pub(crate) const MAX_DUCK_DB: f32 = 12.0;
/// Największy udział cienia głowy (0 = brak, 1 = cała zmierzona różnica L/P).
pub(crate) const MAX_SHADOW: f32 = 0.7;

/// Domyślna głębokość dołków grzebienia „w dół” (dB) i dozwolony zakres.
pub(crate) const DEFAULT_COMB_NOTCH_DB: f32 = -14.0;
pub(crate) const MIN_COMB_NOTCH_DB: f32 = -24.0;
pub(crate) const MAX_COMB_NOTCH_DB: f32 = -6.0;
/// Domyślny poziom strony przeciwnej (dB) i dozwolony zakres. −3 dB przywraca część
/// naturalnej hierarchii, którą mono L+P zabiera (cień głowy).
pub(crate) const DEFAULT_OPPOSITE_LEVEL_DB: f32 = -3.0;
pub(crate) const MIN_OPPOSITE_LEVEL_DB: f32 = -12.0;
pub(crate) const MAX_OPPOSITE_LEVEL_DB: f32 = 6.0;
/// Oznacz i wycofaj: siła wskazówki po początku dźwięku (0–1, 1 = bez wycofywania) i dodatkowe
/// ściszenie ogona (dB, od [`MIN_TAIL_LEVEL_DB`] do 0).
pub(crate) const DEFAULT_CUE_TAIL: f32 = 0.4;
pub(crate) const DEFAULT_TAIL_LEVEL_DB: f32 = -2.0;
pub(crate) const MIN_TAIL_LEVEL_DB: f32 = -6.0;
/// Od tej siły wskazówki po początku dźwięku wycofywanie (razem ze ściszeniem ogona) jest wyłączone.
pub(crate) const NO_FADE_CUE_TAIL: f32 = 0.995;

/// Początek dźwięku z przeciwnej strony: energia części przeciwnej w ramce rośnie o ponad
/// 6 dB względem średniej z ok. 400 ms. Średnia jest dłuższa niż sama seria strzałów między
/// strzałami, więc długa seria nie zgłasza początku przy każdym strzale.
const ONSET_RISE_DB: f32 = 6.0;
const ONSET_AVERAGE_MS: f32 = 400.0;
/// Poniżej tej energii (cisza, ok. −110 dB) nie szukamy początku.
const ONSET_FLOOR: f32 = 1e-6;
/// Po początku: pełna świeżość przez 200 ms, potem w 300 ms spada do zera.
const FRESH_HOLD_MS: f32 = 200.0;
const FRESH_DECAY_MS: f32 = 300.0;
/// O ile ramek opóźniamy świeżość względem ramki, w której wykryto początek. Okno analizy
/// najmocniej waży najnowsze próbki, więc początek widać już ok. ramkę po wejściu dźwięku, a na
/// wyjściu dźwięk pojawia się po [`LATENCY`] (dwóch ramkach). Bez dodatkowego opóźnienia
/// oznaczenie zaczyna się najpóźniej razem z dźwiękiem.
const FRESH_DELAY_HOPS: usize = 0;

/// Która strona jest przeciwna do granego ucha, czyli którą stronę oznaczamy wskazówką.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opposite {
    Left,
    Right,
}

/// Ustawienia rozdzielania na jedną próbkę (zmieniają się najwyżej co callback).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SplitParams {
    /// Siła cieńszej barwy Oddechu w `SplitSample::part_timbre` (0–1).
    pub(crate) timbre: f32,
    /// Barwa bez podnoszenia środka pasma, czyli nigdzie głośniej niż oryginał.
    pub(crate) subtractive: bool,
    /// Ile dB strona przeciwna ustępuje w paśmie, w którym gra strona własna (0 = wyłączone).
    pub(crate) duck_db: f32,
    /// Udział cienia głowy: część przeciwna ściszona o `shadow · (ILD − martwa strefa)` dB.
    pub(crate) shadow: f32,
}

/// Jedna próbka wyjścia rozdzielania, opóźniona o [`LATENCY`] względem wejścia.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SplitSample {
    /// Dźwięk z przeciwnej strony (z ustępowaniem i cieniem głowy).
    pub(crate) part: f32,
    /// Ten sam dźwięk z cieńszą barwą Oddechu.
    pub(crate) part_timbre: f32,
    /// Reszta: swoja strona i środek.
    pub(crate) rest: f32,
    /// Świeżość dźwięku z przeciwnej strony: 1 tuż po jego początku, potem spada do 0.
    pub(crate) fresh: f32,
}

/// Stan rozdzielania kierunkowego dla jednego strumienia wyjściowego. Trzymany między
/// callbackami, bo okno FFT zbiera próbki z kilku kolejnych buforów.
#[derive(Clone)]
pub(crate) struct DirectionalSplit {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// Asymetryczne okna analizy i syntezy (patrz [`analysis_window`] i [`synthesis_window`]).
    analysis: Vec<f32>,
    synthesis: Vec<f32>,
    /// Skala składania ramek: odwrotność sumy iloczynów okien na skok, razem z 1/N odwrotnego FFT.
    ola_scale: f32,
    /// Ostatnie [`FFT_LEN`] próbek L i P (bufor kołowy) i pozycja następnego zapisu.
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    in_pos: usize,
    /// Sumy nakładanych ramek trzech sygnałów wyjścia, indeksowane numerem próbki modulo
    /// [`OLA_LEN`].
    ola_part: Vec<f32>,
    ola_part_timbre: Vec<f32>,
    ola_rest: Vec<f32>,
    /// Numer następnej próbki wejścia od startu albo ostatniego [`Self::reset`].
    t: u64,
    /// Próbki zebrane od ostatniej ramki.
    since_hop: usize,
    /// Bufor FFT (część bez zmian + część z barwą) i bufor reszty.
    buf: Vec<Complex<f32>>,
    buf_rest: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    /// Suma L+P każdego paska z bieżącej ramki.
    sum: Vec<Complex<f32>>,
    /// Surowa i wygładzona moc L i P każdego paska.
    raw_l: Vec<f32>,
    raw_r: Vec<f32>,
    pow_l: Vec<f32>,
    pow_r: Vec<f32>,
    /// Różnica L/P w stronę przeciwną (dB) z bieżącej ramki.
    ild_opp: Vec<f32>,
    /// Wygładzona waga strony przeciwnej `w` i strony własnej (lustrzane progi) każdego paska.
    weight: Vec<f32>,
    weight_own: Vec<f32>,
    /// Barwa przy pełnej sile: z wyrównaniem głośności i w wersji „w dół” (szczyt 1).
    timbre_loud: Vec<f32>,
    timbre_peak: Vec<f32>,
    /// Pasmo ustępowania każdego paska ([`NO_BAND`] poza zakresem).
    band_of_bin: Vec<usize>,
    /// Energia stron w pasmach: z bieżącej ramki i wygładzona, oraz wzmocnienie ustępowania.
    band_now_own: Vec<f32>,
    band_now_opp: Vec<f32>,
    band_own: Vec<f32>,
    band_opp: Vec<f32>,
    band_gain: Vec<f32>,
    /// Średnia energii części przeciwnej i stan świeżości (patrz [`SplitSample::fresh`]).
    opp_energy_avg: f32,
    fresh_now: f32,
    hold_left: u32,
    hold_hops: u32,
    decay_step: f32,
    fresh_fifo: VecDeque<f32>,
    fresh_out: f32,
    /// Współczynniki wygładzania na jedną ramkę.
    power_keep: f32,
    weight_attack_keep: f32,
    weight_release_keep: f32,
    duck_attack_keep: f32,
    duck_release_keep: f32,
    onset_keep: f32,
    sample_rate: f32,
}

impl std::fmt::Debug for DirectionalSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectionalSplit")
            .field("sample_rate", &self.sample_rate)
            .field("t", &self.t)
            .finish_non_exhaustive()
    }
}

/// Współczynnik "ile zostaje" wygładzania wykładniczego na jedną ramkę dla stałej czasu `ms`.
fn keep_per_hop(ms: f32, sample_rate: f32) -> f32 {
    (-(HOP as f32) / (ms * 0.001 * sample_rate)).exp()
}

/// Ile ramek trwa `ms` milisekund.
fn hops_for(ms: f32, sample_rate: f32) -> f32 {
    (ms * 0.001 * sample_rate / HOP as f32).max(1.0)
}

/// Okresowe okno Hanna długości `len` w punkcie `n` (0 ≤ n < len).
fn hann(len: usize, n: usize) -> f32 {
    0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / len as f64).cos() as f32
}

/// Okno analizy o niskim opóźnieniu (Mauler i Martin, 2007): długie zbocze narastające
/// (połowa okna Hanna długości `2·(N − HOP)`) i krótkie opadające (połowa okna Hanna
/// długości [`SYNTH_LEN`]), oba pod pierwiastkiem. Rozdzielczość częstotliwości zostaje prawie
/// taka jak przy zwykłym oknie długości N, a najnowsze próbki ważą najwięcej.
fn analysis_window() -> Vec<f32> {
    let long = 2 * (FFT_LEN - HOP);
    let short_start = FFT_LEN - SYNTH_LEN;
    (0..FFT_LEN)
        .map(|n| if n < FFT_LEN - HOP { hann(long, n).sqrt() } else { hann(SYNTH_LEN, n - short_start).sqrt() })
        .collect()
}

/// Okno syntezy do [`analysis_window`]: zero poza ostatnimi [`SYNTH_LEN`] próbkami, a tam
/// takie, żeby iloczyn obu okien był oknem Hanna długości [`SYNTH_LEN`]. Takie okna nakładane co
/// [`HOP`] sumują się dokładnie do 1, więc bez zmian w paskach wyjście to wejście opóźnione
/// o [`LATENCY`].
fn synthesis_window(analysis: &[f32]) -> Vec<f32> {
    let short_start = FFT_LEN - SYNTH_LEN;
    (0..FFT_LEN)
        .map(|n| {
            if n < short_start {
                0.0
            } else {
                let product = hann(SYNTH_LEN, n - short_start);
                if analysis[n] > 0.0 { product / analysis[n] } else { 0.0 }
            }
        })
        .collect()
}

/// Płynne przejście 0 → 1 (smoothstep) dla `x` z zakresu 0..1.
#[inline]
fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Waga dla różnicy poziomów `ild_db` liczonej w stronę przeciwną (dodatnia = dźwięk głośniejszy
/// po stronie przeciwnej): 0 do [`ILD_DEAD_ZONE_DB`], 1 od [`ILD_FULL_DB`].
#[inline]
pub(crate) fn weight_for_ild(ild_db: f32) -> f32 {
    smoothstep((ild_db - ILD_DEAD_ZONE_DB) / (ILD_FULL_DB - ILD_DEAD_ZONE_DB))
}

/// Ile ustępowania (0–1) przy różnicy `x` = poziom strony własnej minus przeciwnej w paśmie (dB).
#[inline]
fn duck_activation(x: f32) -> f32 {
    if x <= DUCK_OFF_LOW_DB || x >= DUCK_OFF_HIGH_DB {
        0.0
    } else if x < DUCK_FULL_LOW_DB {
        smoothstep((x - DUCK_OFF_LOW_DB) / (DUCK_FULL_LOW_DB - DUCK_OFF_LOW_DB))
    } else if x <= DUCK_FULL_HIGH_DB {
        1.0
    } else {
        smoothstep((DUCK_OFF_HIGH_DB - x) / (DUCK_OFF_HIGH_DB - DUCK_FULL_HIGH_DB))
    }
}

/// Mnożnik oddechu dla bieżącej próbki. Przesuwa `phase` o jedną próbkę oddechu ([`TREMOLO_HZ`]).
/// - `subtractive`: od 1 w dół do `1 − depth`, czyli nigdy głośniej niż oryginał.
/// - inaczej: wahanie wokół 1,0 bez zmiany średniej, od `(1 − depth) / (1 − depth/2)`
///   do `1 / (1 − depth/2)`.
///
/// Szczyt wypada w tej samej fazie co w `mono_router::am_modulate`.
#[inline]
pub(crate) fn breath_gain(depth: f32, sample_rate: f32, phase: &mut f32, subtractive: bool) -> f32 {
    *phase = (*phase + TREMOLO_HZ / sample_rate.max(1.0)).fract();
    let lfo = 0.5 * (1.0 - (*phase * std::f32::consts::TAU).sin());
    let depth = depth.clamp(0.0, MAX_DEPTH);
    let gain = 1.0 - depth * lfo;
    if subtractive { gain } else { gain / (1.0 - depth * 0.5) }
}

/// Krzywa ważenia A jako mnożnik amplitudy (1,0 przy 1 kHz): jak ucho odbiera głośność
/// w zależności od częstotliwości. Bas i najwyższe tony liczą się do głośności dużo mniej.
pub(crate) fn a_weight(f: f64) -> f64 {
    let ra = |f: f64| {
        let f2 = f * f;
        let num = 12_194.0f64.powi(2) * f2 * f2;
        let den = (f2 + 20.6f64.powi(2))
            * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt()
            * (f2 + 12_194.0f64.powi(2));
        num / den
    };
    ra(f) / ra(1_000.0)
}

/// Kształt cieńszej barwy dla każdego paska: moduł filtrów górnoprzepustowego [`TREMOLO_HPF_HZ`]
/// i dolnoprzepustowego [`TIMBRE_LPF_HZ`], oba pierwszego rzędu.
fn timbre_shape(sample_rate: f32) -> Vec<f32> {
    (0..BINS)
        .map(|k| {
            let f = k as f32 * sample_rate / FFT_LEN as f32;
            let x = f / TREMOLO_HPF_HZ;
            let y = f / TIMBRE_LPF_HZ;
            (x / (1.0 + x * x).sqrt()) / (1.0 + y * y).sqrt()
        })
        .collect()
}

/// Barwa z wyrównaniem głośności: kształt podniesiony tak, żeby szum różowy (widmo podobne do
/// dźwięków gry) był po zmianie tak samo głośny dla ucha (ważenie A). Tracimy dół i górę,
/// a środek pasma idzie trochę w górę. Wyrównanie po surowym RMS przesadzałoby, bo liczyłoby
/// bas, którego ucho prawie nie odbiera jako głośności.
fn timbre_curve(sample_rate: f32) -> Vec<f32> {
    let shape = timbre_shape(sample_rate);
    // Moc szumu różowego w pasku ~ 1/f, razem z ważeniem A.
    let (mut before, mut after) = (0.0f64, 0.0f64);
    for (k, &h) in shape.iter().enumerate() {
        let f = k as f64 * f64::from(sample_rate) / FFT_LEN as f64;
        if (20.0..=20_000.0).contains(&f) {
            let weight = a_weight(f).powi(2) / f;
            before += weight;
            after += weight * f64::from(h * h);
        }
    }
    let makeup = if after > 0.0 { (before / after).sqrt() as f32 } else { 1.0 };
    shape.into_iter().map(|h| h * makeup).collect()
}

/// Barwa „w dół”: kształt podzielony przez swoje maksimum, więc żaden pasek nie jest głośniej
/// niż oryginał. Dół i góra są tylko ścinane.
fn timbre_curve_peak(sample_rate: f32) -> Vec<f32> {
    let shape = timbre_shape(sample_rate);
    let peak = shape.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    shape.into_iter().map(|h| h / peak).collect()
}

/// Wyrównania cieńszej barwy dla wersji liczonej filtrami w czasie (przód i tył w stylu „Wg
/// kierunku”, patrz `mono_router::zone_timbre`): `(z wyrównaniem głośności, „w dół”)`. Te same
/// mnożniki, którymi [`timbre_curve`] i [`timbre_curve_peak`] skalują kształt [`timbre_shape`].
pub(crate) fn timbre_makeups(sample_rate: f32) -> (f32, f32) {
    let shape = timbre_shape(sample_rate.max(1.0));
    let loud = timbre_curve(sample_rate.max(1.0));
    let (at, peak) = shape.iter().cloned().enumerate().fold((0, 0.0f32), |best, (k, h)| if h > best.1 { (k, h) } else { best });
    let peak = peak.max(1e-6);
    (loud[at] / peak, 1.0 / peak)
}

impl DirectionalSplit {
    pub(crate) fn new(sample_rate: f32) -> Self {
        let sample_rate = sample_rate.max(1.0);
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_LEN);
        let inverse = planner.plan_fft_inverse(FFT_LEN);
        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        let analysis = analysis_window();
        let synthesis = synthesis_window(&analysis);
        // Średnia suma nakładanych iloczynów okien w jednej próbce (dla tej pary dokładnie 1).
        let overlap_gain: f32 = analysis.iter().zip(&synthesis).map(|(a, s)| a * s).sum::<f32>() / HOP as f32;
        let band_of_bin: Vec<usize> = (0..BINS)
            .map(|k| {
                let f = k as f32 * sample_rate / FFT_LEN as f32;
                if (DUCK_LOW_HZ..DUCK_HIGH_HZ).contains(&f) {
                    (DUCK_BANDS_PER_OCTAVE * (f / DUCK_LOW_HZ).log2()).floor() as usize
                } else {
                    NO_BAND
                }
            })
            .collect();
        let bands = (DUCK_BANDS_PER_OCTAVE * (DUCK_HIGH_HZ / DUCK_LOW_HZ).log2()).ceil() as usize;
        Self {
            forward,
            inverse,
            ola_scale: 1.0 / (overlap_gain * FFT_LEN as f32),
            analysis,
            synthesis,
            in_l: vec![0.0; FFT_LEN],
            in_r: vec![0.0; FFT_LEN],
            in_pos: 0,
            ola_part: vec![0.0; OLA_LEN],
            ola_part_timbre: vec![0.0; OLA_LEN],
            ola_rest: vec![0.0; OLA_LEN],
            t: 0,
            since_hop: 0,
            buf: vec![Complex::new(0.0, 0.0); FFT_LEN],
            buf_rest: vec![Complex::new(0.0, 0.0); FFT_LEN],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            sum: vec![Complex::new(0.0, 0.0); BINS],
            raw_l: vec![0.0; BINS],
            raw_r: vec![0.0; BINS],
            pow_l: vec![0.0; BINS],
            pow_r: vec![0.0; BINS],
            ild_opp: vec![0.0; BINS],
            weight: vec![0.0; BINS],
            weight_own: vec![0.0; BINS],
            timbre_loud: timbre_curve(sample_rate),
            timbre_peak: timbre_curve_peak(sample_rate),
            band_of_bin,
            band_now_own: vec![0.0; bands],
            band_now_opp: vec![0.0; bands],
            band_own: vec![0.0; bands],
            band_opp: vec![0.0; bands],
            band_gain: vec![1.0; bands],
            opp_energy_avg: 0.0,
            fresh_now: 0.0,
            hold_left: 0,
            hold_hops: hops_for(FRESH_HOLD_MS, sample_rate).round() as u32,
            decay_step: 1.0 / hops_for(FRESH_DECAY_MS, sample_rate),
            fresh_fifo: VecDeque::with_capacity(FRESH_DELAY_HOPS + 1),
            fresh_out: 0.0,
            power_keep: keep_per_hop(POWER_SMOOTH_MS, sample_rate),
            weight_attack_keep: keep_per_hop(WEIGHT_ATTACK_MS, sample_rate),
            weight_release_keep: keep_per_hop(WEIGHT_RELEASE_MS, sample_rate),
            duck_attack_keep: keep_per_hop(DUCK_ATTACK_MS, sample_rate),
            duck_release_keep: keep_per_hop(DUCK_RELEASE_MS, sample_rate),
            onset_keep: keep_per_hop(ONSET_AVERAGE_MS, sample_rate),
            sample_rate,
        }
    }

    /// Czyści okno i całą historię, jak po starcie. Wołane, gdy styl kierunkowy wraca po
    /// przerwie, żeby nie wypuścić resztek dźwięku sprzed przełączenia.
    pub(crate) fn reset(&mut self) {
        self.in_l.fill(0.0);
        self.in_r.fill(0.0);
        self.in_pos = 0;
        self.ola_part.fill(0.0);
        self.ola_part_timbre.fill(0.0);
        self.ola_rest.fill(0.0);
        self.t = 0;
        self.since_hop = 0;
        self.pow_l.fill(0.0);
        self.pow_r.fill(0.0);
        self.weight.fill(0.0);
        self.weight_own.fill(0.0);
        self.band_own.fill(0.0);
        self.band_opp.fill(0.0);
        self.band_gain.fill(1.0);
        self.opp_energy_avg = 0.0;
        self.fresh_now = 0.0;
        self.hold_left = 0;
        self.fresh_fifo.clear();
        self.fresh_out = 0.0;
    }

    /// Jedna próbka wejścia (L, P) na próbkę wyjścia rozdzielania (patrz [`SplitSample`]),
    /// opóźnioną o [`LATENCY`] próbek (pierwsze [`LATENCY`] to cisza).
    pub(crate) fn process(&mut self, left: f32, right: f32, opposite: Opposite, params: SplitParams) -> SplitSample {
        self.in_l[self.in_pos] = left;
        self.in_r[self.in_pos] = right;
        self.in_pos = (self.in_pos + 1) % FFT_LEN;

        // Próbka sprzed LATENCY jest już kompletna: synteza każdej ramki pokrywa tylko jej
        // ostatnie LATENCY próbek, a ostatnia ramka, która ją obejmuje, skończyła się najpóźniej
        // próbkę temu. Slot próbki s to s mod OLA_LEN.
        let read = ((self.t + (OLA_LEN - LATENCY) as u64) % OLA_LEN as u64) as usize;
        let out = if self.t >= LATENCY as u64 {
            SplitSample {
                part: self.ola_part[read],
                part_timbre: self.ola_part_timbre[read],
                rest: self.ola_rest[read],
                fresh: self.fresh_out,
            }
        } else {
            SplitSample::default()
        };
        self.ola_part[read] = 0.0;
        self.ola_part_timbre[read] = 0.0;
        self.ola_rest[read] = 0.0;

        self.since_hop += 1;
        if self.since_hop == HOP {
            self.since_hop = 0;
            self.process_frame(opposite, params);
        }
        self.t += 1;
        out
    }

    /// Ramka STFT kończąca się na próbce `self.t`: analiza, wagi per pasek, ustępowanie,
    /// synteza trzech sygnałów i świeżość.
    fn process_frame(&mut self, opposite: Opposite, params: SplitParams) {
        let timbre = params.timbre.clamp(0.0, 1.0);
        let duck_db = params.duck_db.clamp(0.0, MAX_DUCK_DB);
        let shadow = params.shadow.clamp(0.0, MAX_SHADOW);

        // L jako część rzeczywista, P jako urojona: jedno FFT liczy oba kanały naraz.
        for i in 0..FFT_LEN {
            let idx = (self.in_pos + i) % FFT_LEN; // in_pos wskazuje teraz najstarszą próbkę
            let w = self.analysis[i];
            self.buf[i] = Complex::new(self.in_l[idx] * w, self.in_r[idx] * w);
        }
        self.forward.process_with_scratch(&mut self.buf, &mut self.scratch);

        let half_i = Complex::new(0.0, -0.5);
        for k in 0..BINS {
            let z = self.buf[k];
            let zc = self.buf[(FFT_LEN - k) % FFT_LEN].conj();
            let xl = (z + zc) * 0.5;
            let xr = (z - zc) * half_i;
            self.raw_l[k] = xl.norm_sqr();
            self.raw_r[k] = xr.norm_sqr();
            self.sum[k] = xl + xr;
        }

        // Wagi stron per pasek i energia stron w pasmach ustępowania.
        let keep = self.power_keep;
        let mut opp_energy = 0.0f32;
        self.band_now_own.fill(0.0);
        self.band_now_opp.fill(0.0);
        for k in 0..BINS {
            // Moc wygładzona po sąsiednich paskach (±1) i w czasie, osobno dla L i P.
            let lo = k.saturating_sub(1);
            let hi = (k + 1).min(BINS - 1);
            let n = (hi - lo + 1) as f32;
            let pl: f32 = self.raw_l[lo..=hi].iter().sum::<f32>() / n;
            let pr: f32 = self.raw_r[lo..=hi].iter().sum::<f32>() / n;
            self.pow_l[k] = keep * self.pow_l[k] + (1.0 - keep) * pl;
            self.pow_r[k] = keep * self.pow_r[k] + (1.0 - keep) * pr;

            const EPS: f32 = 1e-12;
            let ild_right = 10.0 * ((self.pow_r[k] + EPS) / (self.pow_l[k] + EPS)).log10();
            let ild_opposite = match opposite {
                Opposite::Right => ild_right,
                Opposite::Left => -ild_right,
            };
            self.ild_opp[k] = ild_opposite;
            let target = weight_for_ild(ild_opposite);
            let w_keep = if target > self.weight[k] { self.weight_attack_keep } else { self.weight_release_keep };
            self.weight[k] = w_keep * self.weight[k] + (1.0 - w_keep) * target;
            // Strona własna z lustrzanymi progami: środek (np. własna broń) ma wagę 0.
            let target_own = weight_for_ild(-ild_opposite);
            let own_keep =
                if target_own > self.weight_own[k] { self.weight_attack_keep } else { self.weight_release_keep };
            self.weight_own[k] = own_keep * self.weight_own[k] + (1.0 - own_keep) * target_own;

            let power = self.sum[k].norm_sqr();
            let (w, w_own) = (self.weight[k], self.weight_own[k]);
            opp_energy += w * w * power;
            let band = self.band_of_bin[k];
            if band != NO_BAND {
                self.band_now_own[band] += w_own * w_own * power;
                self.band_now_opp[band] += w * w * power;
            }
        }

        // Ustępowanie: w paśmie, w którym strona własna gra i grozi jej maskowanie, część
        // przeciwna dostaje do `duck_db` mniej.
        for band in 0..self.band_gain.len() {
            let smooth = |old: f32, now: f32, attack: f32, release: f32| {
                let k = if now > old { attack } else { release };
                k * old + (1.0 - k) * now
            };
            self.band_own[band] =
                smooth(self.band_own[band], self.band_now_own[band], self.duck_attack_keep, self.duck_release_keep);
            self.band_opp[band] =
                smooth(self.band_opp[band], self.band_now_opp[band], self.duck_attack_keep, self.duck_release_keep);
            self.band_gain[band] = if duck_db > 0.0 {
                const EPS: f32 = 1e-20;
                let x = 10.0 * ((self.band_own[band] + EPS) / (self.band_opp[band] + EPS)).log10();
                10f32.powf(-duck_db * duck_activation(x) / 20.0)
            } else {
                1.0
            };
        }

        // Synteza: część przeciwna (z ustępowaniem i cieniem głowy), jej wersja z barwą, reszta.
        let curve = if params.subtractive { &self.timbre_peak } else { &self.timbre_loud };
        let j = Complex::new(0.0, 1.0);
        for (k, &curve_k) in curve.iter().enumerate() {
            let w = self.weight[k];
            let band = self.band_of_bin[k];
            let duck = if band != NO_BAND { self.band_gain[band] } else { 1.0 };
            let shade = if shadow > 0.0 {
                10f32.powf(-shadow * (self.ild_opp[k] - ILD_DEAD_ZONE_DB).max(0.0) / 20.0)
            } else {
                1.0
            };
            let color = 1.0 + timbre * (curve_k - 1.0);
            let mut part = self.sum[k] * (w * duck * shade);
            let mut part_timbre = part * color;
            let mut rest = self.sum[k] * (1.0 - w);
            if k == 0 || k == FFT_LEN / 2 {
                // Pasek 0 Hz i Nyquista sygnału rzeczywistego jest rzeczywisty.
                part.im = 0.0;
                part_timbre.im = 0.0;
                rest.im = 0.0;
            }
            // Część bez zmian i z barwą to sygnały rzeczywiste, więc jedno odwrotne FFT liczy
            // je naraz (pierwsza jako rzeczywista, druga jako urojona). Reszta idzie osobno.
            self.buf[k] = part + j * part_timbre;
            self.buf_rest[k] = rest;
            if k > 0 && k < FFT_LEN / 2 {
                self.buf[FFT_LEN - k] = part.conj() + j * part_timbre.conj();
                self.buf_rest[FFT_LEN - k] = rest.conj();
            }
        }
        self.inverse.process_with_scratch(&mut self.buf, &mut self.scratch);
        self.inverse.process_with_scratch(&mut self.buf_rest, &mut self.scratch);

        // Ramka obejmuje próbki t-N+1 ..= t; ich sloty to (t + N + 1 + i) mod OLA_LEN. Okno
        // syntezy jest niezerowe tylko na ostatnich SYNTH_LEN próbkach, więc tylko je dopisujemy.
        let base = ((self.t + FFT_LEN as u64 + 1) % OLA_LEN as u64) as usize;
        for i in FFT_LEN - SYNTH_LEN..FFT_LEN {
            let slot = (base + i) % OLA_LEN;
            let scale = self.synthesis[i] * self.ola_scale;
            self.ola_part[slot] += self.buf[i].re * scale;
            self.ola_part_timbre[slot] += self.buf[i].im * scale;
            self.ola_rest[slot] += self.buf_rest[i].re * scale;
        }

        self.update_freshness(opp_energy);
    }

    /// Wykrywa początek dźwięku z przeciwnej strony i prowadzi świeżość: 1 przez
    /// [`FRESH_HOLD_MS`] od początku, potem liniowo do 0 w [`FRESH_DECAY_MS`].
    fn update_freshness(&mut self, opp_energy: f32) {
        let rise = 10f32.powf(ONSET_RISE_DB / 10.0);
        let onset = opp_energy > ONSET_FLOOR && opp_energy > self.opp_energy_avg * rise;
        self.opp_energy_avg = self.onset_keep * self.opp_energy_avg + (1.0 - self.onset_keep) * opp_energy;
        if onset {
            self.fresh_now = 1.0;
            self.hold_left = self.hold_hops;
        } else if self.hold_left > 0 {
            self.hold_left -= 1;
        } else {
            self.fresh_now = (self.fresh_now - self.decay_step).max(0.0);
        }
        self.fresh_fifo.push_back(self.fresh_now);
        while self.fresh_fifo.len() > FRESH_DELAY_HOPS {
            self.fresh_out = self.fresh_fifo.pop_front().unwrap_or(0.0);
        }
    }
}

/// Sygnały testowe i pomiar głośności dla ucha, wspólne dla testów tego modułu i `mono_router`.
#[cfg(test)]
pub(crate) mod test_signals {
    use super::a_weight;
    use rustfft::{FftPlanner, num_complex::Complex};

    pub(crate) const SR: f32 = 48_000.0;

    /// Biały szum z prostego generatora LCG, deterministyczny, amplituda ok. ±0,1.
    pub(crate) fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    /// Szum różowy (filtr Paula Kelleta na białym szumie), widmo podobne do dźwięków gry.
    pub(crate) fn pink(n: usize, seed: u32) -> Vec<f32> {
        let white = noise(n, seed);
        let (mut b0, mut b1, mut b2) = (0.0f32, 0.0f32, 0.0f32);
        white
            .iter()
            .map(|&w| {
                b0 = 0.99765 * b0 + w * 0.099_046;
                b1 = 0.963 * b1 + w * 0.296_516_4;
                b2 = 0.57 * b2 + w * 1.052_691_3;
                (b0 + b1 + b2 + w * 0.1848) * 0.3
            })
            .collect()
    }

    /// Rozkłada dźwięk na L i P z różnicą poziomów `ild_db` (dodatnia = głośniej po prawej).
    pub(crate) fn pan(src: &[f32], ild_db: f32) -> (Vec<f32>, Vec<f32>) {
        let gl = 10f32.powf(-ild_db / 40.0);
        let gr = 10f32.powf(ild_db / 40.0);
        (src.iter().map(|x| x * gl).collect(), src.iter().map(|x| x * gr).collect())
    }

    /// Głośność dla ucha (moc ważona krzywą A) z pierwszych 65 536 próbek, jednym FFT, w dB.
    pub(crate) fn a_weighted_db(x: &[f32]) -> f32 {
        let n = 65_536;
        let mut buf: Vec<Complex<f32>> = x[..n].iter().map(|&v| Complex::new(v, 0.0)).collect();
        FftPlanner::<f32>::new().plan_fft_forward(n).process(&mut buf);
        let power: f64 = (1..n / 2)
            .map(|k| {
                let f = k as f64 * f64::from(SR) / n as f64;
                f64::from(buf[k].norm_sqr()) * a_weight(f).powi(2)
            })
            .sum();
        (10.0 * (power + 1e-30).log10()) as f32
    }

    /// Amplituda składowej `hz` w sygnale (korelacja z sinusem i cosinusem), dla porównań w dB.
    pub(crate) fn tone_level(x: &[f32], hz: f32) -> f32 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, v) in x.iter().enumerate() {
            let ang = std::f32::consts::TAU * hz * i as f32 / SR;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        2.0 * (re * re + im * im).sqrt() / x.len().max(1) as f32
    }

    /// Indeks modulacji `m` obwiedni na częstotliwości `hz`: obwiednia z bloków RMS po 1 ms
    /// (dla sinusa 1 kHz to jeden okres na blok), potem składowa `hz` względem średniej.
    /// Dla obwiedni `c·(1 + m·sin)` zwraca `m`. Średnia jest odejmowana, a obwiednia mnożona
    /// przez okno Hanna, żeby przy niecałkowitej liczbie cykli w oknie składowa stała ani inne
    /// częstotliwości nie przeciekały do mierzonej.
    pub(crate) fn modulation_index(x: &[f32], hz: f32) -> f32 {
        let block = (0.001 * SR) as usize;
        let env: Vec<f32> = x
            .chunks_exact(block)
            .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / block as f32).sqrt())
            .collect();
        let n = env.len().max(2);
        let mean = env.iter().sum::<f32>() / env.len().max(1) as f32;
        let (mut re, mut im, mut window_sum) = (0.0f32, 0.0f32, 0.0f32);
        for (k, e) in env.iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * k as f32 / (n - 1) as f32).cos();
            let ang = std::f32::consts::TAU * hz * k as f32 * 0.001;
            re += w * (e - mean) * ang.cos();
            im += w * (e - mean) * ang.sin();
            window_sum += w;
        }
        2.0 * (re * re + im * im).sqrt() / window_sum.max(1e-12) / mean.max(1e-12)
    }
}

#[cfg(test)]
mod tests {
    use super::test_signals::*;
    use super::*;

    /// Ustawienia bez barwy, ustępowania i cienia: samo rozdzielanie.
    fn plain(timbre: f32) -> SplitParams {
        SplitParams { timbre, ..SplitParams::default() }
    }

    fn run(left: &[f32], right: &[f32], opposite: Opposite, params: SplitParams) -> Vec<SplitSample> {
        let mut s = DirectionalSplit::new(SR);
        left.iter().zip(right).map(|(&l, &r)| s.process(l, r, opposite, params)).collect()
    }

    /// `(część przeciwna, reszta)` dla każdej próbki; z `timbre > 0` część z barwą.
    fn split(left: &[f32], right: &[f32], opposite: Opposite, timbre: f32) -> (Vec<f32>, Vec<f32>) {
        run(left, right, opposite, plain(timbre))
            .into_iter()
            .map(|s| (if timbre > 0.0 { s.part_timbre } else { s.part }, s.rest))
            .unzip()
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
    }

    fn db(ratio: f32) -> f32 {
        20.0 * ratio.max(1e-12).log10()
    }

    #[test]
    fn parts_add_up_to_the_input_after_one_window() {
        // Bez ustępowania i cienia część przeciwna + reszta = L+P co do próbki, z każdego kierunku.
        let a = noise(12_000, 1);
        let b = noise(12_000, 2);
        let left: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + 0.3 * y).collect();
        let right: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.4 * x + y).collect();
        let out = run(&left, &right, Opposite::Right, plain(0.0));
        for t in 2 * FFT_LEN..left.len() {
            let expected = left[t - LATENCY] + right[t - LATENCY];
            assert!((out[t].part + out[t].rest - expected).abs() < 1e-3, "próbka {t}");
        }
    }

    #[test]
    fn opposite_part_lines_up_with_the_delayed_input() {
        let (l, r) = pan(&noise(12_000, 1), 9.0);
        let out = run(&l, &r, Opposite::Right, plain(0.0));
        for t in 4_000..l.len() {
            let expected = l[t - LATENCY] + r[t - LATENCY];
            assert!((out[t].part - expected).abs() < 1e-3, "próbka {t}: {} zamiast {expected}", out[t].part);
            assert!(out[t].rest.abs() < 1e-3, "próbka {t}: reszta {}", out[t].rest);
        }
    }

    #[test]
    fn part_without_timbre_is_the_same_in_both_versions() {
        let (l, r) = pan(&noise(8_000, 2), 9.0);
        for s in run(&l, &r, Opposite::Right, plain(0.0)) {
            assert!((s.part - s.part_timbre).abs() < 1e-5, "{s:?}");
        }
    }

    #[test]
    fn center_and_own_side_go_entirely_to_the_rest() {
        for ild in [0.0, -6.0] {
            let (l, r) = pan(&noise(24_000, 3), ild);
            let (opp, rest) = split(&l, &r, Opposite::Right, 1.0);
            let ratio = energy(&opp[4_000..]) / energy(&rest[4_000..]);
            assert!(ratio < 1e-6, "ILD {ild}: część przeciwna ma być pusta, stosunek {ratio}");
        }
    }

    #[test]
    fn opposite_side_goes_entirely_to_the_opposite_part() {
        let (l, r) = pan(&noise(24_000, 4), 6.0);
        let (opp, rest) = split(&l, &r, Opposite::Right, 0.0);
        let ratio = energy(&rest[4_000..]) / energy(&opp[4_000..]);
        assert!(ratio < 1e-4, "reszta ma być pusta, stosunek {ratio}");
    }

    #[test]
    fn slightly_off_center_sound_is_shared_between_both_parts() {
        let (l, r) = pan(&noise(24_000, 5), 3.0);
        let (opp, rest) = split(&l, &r, Opposite::Right, 0.0);
        let share = energy(&opp[4_000..]) / (energy(&opp[4_000..]) + energy(&rest[4_000..]));
        assert!(share > 0.01 && share < 0.5, "lekko z boku: część trafia do przeciwnej, jest {share}");
    }

    #[test]
    fn right_ear_flips_the_sides() {
        let (l, r) = pan(&noise(24_000, 6), -6.0);
        let (opp, rest) = split(&l, &r, Opposite::Left, 0.0);
        assert!(energy(&rest[4_000..]) / energy(&opp[4_000..]) < 1e-4, "przy prawym uchu przeciwna jest lewa");
    }

    #[test]
    fn simultaneous_sounds_from_both_sides_are_split_apart() {
        // Głośny niski dźwięk z lewej (250 Hz, ILD −8 dB) i cichszy wysoki z prawej (3 kHz,
        // +8 dB), grane naraz, ucho lewe: wysoki ma trafić do części przeciwnej, niski do reszty.
        let n = 48_000;
        let g = 10f32.powf(8.0 / 40.0);
        let tone = |hz: f32, i: usize| (std::f32::consts::TAU * hz * i as f32 / SR).sin();
        let left: Vec<f32> = (0..n).map(|i| 0.4 * tone(250.0, i) * g + 0.1 * tone(3_000.0, i) / g).collect();
        let right: Vec<f32> = (0..n).map(|i| 0.4 * tone(250.0, i) / g + 0.1 * tone(3_000.0, i) * g).collect();
        let (opp, rest) = split(&left, &right, Opposite::Right, 0.0);
        let (o, r) = (&opp[12_000..], &rest[12_000..]);
        let high_in_opp_db = db(tone_level(o, 3_000.0) / tone_level(r, 3_000.0));
        let low_in_rest_db = db(tone_level(r, 250.0) / tone_level(o, 250.0));
        assert!(high_in_opp_db > 30.0, "dźwięk z prawej ma być w części przeciwnej: {high_in_opp_db:.1} dB");
        assert!(low_in_rest_db > 30.0, "dźwięk z lewej ma zostać w reszcie: {low_in_rest_db:.1} dB");
    }

    #[test]
    fn thinner_timbre_keeps_loudness_of_pink_noise() {
        // Barwa z wyrównaniem (tryb dawny, nie „w dół”).
        let (l, r) = pan(&pink(96_000, 8), 9.0);
        let plain_sum: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        let (opp, _) = split(&l, &r, Opposite::Right, 1.0);
        let diff = a_weighted_db(&opp[24_000..]) - a_weighted_db(&plain_sum[24_000 - LATENCY..]);
        assert!(diff.abs() < 1.0, "cieńsza barwa nie powinna ściszać dla ucha: {diff:+.2} dB");
    }

    #[test]
    fn thinner_timbre_cuts_bass_relative_to_the_middle() {
        // Kształt barwy przy pełnej sile: 100 Hz wyraźnie niżej niż 1 kHz, jak w zwykłym Oddechu.
        let curve = timbre_curve(SR);
        let at = |hz: f32| curve[(hz * FFT_LEN as f32 / SR).round() as usize];
        let ratio_db = db(at(1_000.0) / at(100.0));
        assert!(ratio_db > 7.0, "dół ma być ścięty względem środka: {ratio_db:.1} dB");
    }

    #[test]
    fn subtractive_timbre_never_boosts_and_keeps_most_of_the_top() {
        let curve = timbre_curve_peak(SR);
        let peak = curve.iter().cloned().fold(0.0f32, f32::max);
        assert!(peak <= 1.0 + 1e-6, "barwa „w dół” nie może podbijać: {peak}");
        let at = |hz: f32| curve[(hz * FFT_LEN as f32 / SR).round() as usize];
        let top_db = db(at(8_000.0));
        assert!(top_db > -3.5, "8 kHz ma tracić najwyżej ok. 3 dB: {top_db:.1} dB");
        assert!(db(at(100.0)) < -8.0, "dół dalej ścięty: {:.1} dB", db(at(100.0)));
    }

    #[test]
    fn breath_gain_swings_around_one_without_changing_the_average() {
        let mut phase = 0.0;
        let cycle = (SR / TREMOLO_HZ) as usize;
        let gains: Vec<f32> = (0..cycle).map(|_| breath_gain(0.6, SR, &mut phase, false)).collect();
        let mean = gains.iter().sum::<f32>() / cycle as f32;
        let max = gains.iter().cloned().fold(f32::MIN, f32::max);
        let min = gains.iter().cloned().fold(f32::MAX, f32::min);
        assert!((mean - 1.0).abs() < 1e-3, "średnia ma zostać 1,0: {mean}");
        let swing_db = db(max / min);
        let expected = -20.0 * (1.0f32 - 0.6).log10();
        assert!((swing_db - expected).abs() < 0.1, "wahanie {swing_db:.2} dB zamiast {expected:.2} dB");
    }

    #[test]
    fn subtractive_breath_only_goes_down_from_the_original() {
        let mut phase = 0.0;
        let cycle = (SR / TREMOLO_HZ) as usize;
        let gains: Vec<f32> = (0..cycle).map(|_| breath_gain(DEFAULT_DEPTH, SR, &mut phase, true)).collect();
        let max = gains.iter().cloned().fold(f32::MIN, f32::max);
        let min = gains.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max <= 1.0 + 1e-6 && max > 0.999, "szczyt na poziomie oryginału: {max}");
        assert!((db(min) + 5.0).abs() < 0.1, "domyślny dołek ok. −5 dB: {:.2} dB", db(min));
    }

    /// Dwa sinusy w tym samym paśmie 1/3 oktawy: `own_hz` po stronie własnej (L, ILD −8 dB albo
    /// środek) i 2,3 kHz po przeciwnej (P, +8 dB), plus 6 kHz po przeciwnej w innym paśmie.
    /// Zwraca poziomy części przeciwnej przy 2,3 i 6 kHz (dB) z ustępowaniem minus bez niego.
    fn duck_effect(own_amp: f32, own_centered: bool) -> (f32, f32) {
        let n = 48_000;
        let g = 10f32.powf(8.0 / 40.0);
        let tone = |hz: f32, i: usize| (std::f32::consts::TAU * hz * i as f32 / SR).sin();
        let (own_l, own_r) = if own_centered { (1.0, 1.0) } else { (g, 1.0 / g) };
        let left: Vec<f32> = (0..n)
            .map(|i| own_amp * tone(2_000.0, i) * own_l + 0.1 * (tone(2_300.0, i) + tone(6_000.0, i)) / g)
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| own_amp * tone(2_000.0, i) * own_r + 0.1 * (tone(2_300.0, i) + tone(6_000.0, i)) * g)
            .collect();
        let parts = |duck_db: f32| -> Vec<f32> {
            let params = SplitParams { duck_db, ..SplitParams::default() };
            run(&left, &right, Opposite::Right, params).iter().skip(24_000).map(|s| s.part).collect()
        };
        let (with, without) = (parts(6.0), parts(0.0));
        (
            db(tone_level(&with, 2_300.0) / tone_level(&without, 2_300.0)),
            db(tone_level(&with, 6_000.0) / tone_level(&without, 6_000.0)),
        )
    }

    #[test]
    fn opposite_side_yields_only_in_the_band_where_the_own_side_plays() {
        let (same_band, other_band) = duck_effect(0.05, false);
        assert!((same_band + 6.0).abs() < 1.0, "w paśmie strony własnej ok. −6 dB: {same_band:.2} dB");
        assert!(other_band.abs() < 0.5, "inne pasmo bez zmian: {other_band:.2} dB");
    }

    #[test]
    fn center_does_not_make_the_opposite_side_yield() {
        // Własna broń gra ze środka: nie może ściszać wrogów.
        let (same_band, _) = duck_effect(0.05, true);
        assert!(same_band.abs() < 0.5, "środek nie steruje ustępowaniem: {same_band:.2} dB");
    }

    #[test]
    fn quiet_own_background_does_not_make_the_opposite_side_yield() {
        let (same_band, _) = duck_effect(0.1 * 10f32.powf(-25.0 / 20.0), false);
        assert!(same_band.abs() < 0.5, "tło 25 dB pod przeciwną nie steruje: {same_band:.2} dB");
    }

    #[test]
    fn head_shadow_makes_sounds_further_to_the_side_quieter() {
        let level = |ild: f32, shadow: f32| {
            let (l, r) = pan(&noise(36_000, 11), ild);
            let params = SplitParams { shadow, ..SplitParams::default() };
            let part: Vec<f32> = run(&l, &r, Opposite::Right, params).iter().skip(12_000).map(|s| s.part).collect();
            10.0 * (energy(&part) as f32).log10()
        };
        let at6 = level(6.0, 0.5) - level(6.0, 0.0);
        let at10 = level(10.0, 0.5) - level(10.0, 0.0);
        assert!((at6 + 2.25).abs() < 0.3, "ILD 6 dB: połowa z 4,5 dB ponad martwą strefę: {at6:.2} dB");
        assert!((at10 + 4.25).abs() < 0.3, "ILD 10 dB: połowa z 8,5 dB: {at10:.2} dB");
    }

    #[test]
    fn freshness_marks_the_start_of_an_opposite_sound_and_then_fades() {
        // Cisza, potem od próbki 10 000 ciągły szum z prawej.
        let start = 10_000;
        let mut src = vec![0.0; start];
        src.extend(noise(60_000, 12));
        let (l, r) = pan(&src, 9.0);
        let out = run(&l, &r, Opposite::Right, plain(0.0));
        let at = |ms: f32| out[start + LATENCY + (ms * 0.001 * SR) as usize].fresh;
        assert!(at(0.0) > 0.99, "świeżość gotowa razem z początkiem dźwięku: {}", at(0.0));
        assert!(at(150.0) > 0.99, "przez ok. 200 ms pełna: {}", at(150.0));
        assert!(at(800.0) < 0.05, "przy ciągłym dźwięku potem wygasa: {}", at(800.0));
        assert!(out[start / 2].fresh == 0.0, "w ciszy brak świeżości");
    }

    #[test]
    fn silence_stays_silent_and_loud_input_stays_finite() {
        let zeros = vec![0.0; 4_000];
        let (opp, rest) = split(&zeros, &zeros, Opposite::Right, 1.0);
        assert!(opp.iter().chain(&rest).all(|&y| y == 0.0));
        let loud: Vec<f32> = noise(20_000, 9).iter().map(|x| x * 9.0).collect();
        let quiet: Vec<f32> = loud.iter().map(|x| x * 0.1).collect();
        let params = SplitParams { timbre: 1.0, subtractive: true, duck_db: MAX_DUCK_DB, shadow: MAX_SHADOW };
        for s in run(&quiet, &loud, Opposite::Right, params) {
            for y in [s.part, s.part_timbre, s.rest, s.fresh] {
                assert!(y.is_finite() && y.abs() < 10.0, "{s:?}");
            }
        }
    }

    #[test]
    fn reset_drops_everything_from_before() {
        let mut s = DirectionalSplit::new(SR);
        for x in noise(5_000, 10) {
            s.process(x, x * 2.0, Opposite::Right, plain(1.0));
        }
        s.reset();
        for _ in 0..3 * FFT_LEN {
            assert_eq!(s.process(0.0, 0.0, Opposite::Right, plain(1.0)), SplitSample::default());
        }
    }
}
