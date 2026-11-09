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
//!   1. L i P dzielimy na paski co ok. 94 Hz (STFT: okno 512 próbek, nowa ramka co 128),
//!   2. w każdym pasku porównujemy moc L i P, czyli różnicę poziomów (ILD) w dB,
//!   3. z różnicy liczymy wagę `w`: 0 dla swojej strony i środka (do [`ILD_DEAD_ZONE_DB`]),
//!      rośnie do 1 przy [`ILD_FULL_DB`] w stronę przeciwną,
//!   4. ułamek `w` paska (L+P razem) to część przeciwna; składamy ją z powrotem w dwóch
//!      wersjach naraz (jedno odwrotne FFT): bez zmian i z cieńszą barwą Oddechu,
//!   5. reszta to wejście L+P opóźnione o okno minus część przeciwna, czyli dokładnie
//!      ułamek `1 − w` każdego paska.
//!
//! Część przeciwna i reszta sumują się dokładnie do L+P, opóźnionego o [`FFT_LEN`] próbek
//! (ok. 11 ms przy 48 kHz). Dźwięki z różnych stron grające naraz zwykle zajmują w danej
//! chwili różne paski, więc każdy trafia do swojej części.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::sound::mono_router::{TREMOLO_HPF_HZ, TREMOLO_HZ, TREMOLO_LPF_HZ};

/// Długość okna FFT w próbkach. Przy 48 kHz to ok. 11 ms opóźnienia i paski co ok. 94 Hz.
pub(crate) const FFT_LEN: usize = 512;
/// Skok między ramkami: ćwierć okna (nakładanie 75 %), więc decyzja odświeża się co ok. 2,7 ms.
const HOP: usize = FFT_LEN / 4;
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

/// Domyślna głębokość oddechu (0–1) w stylach z oddechem. 0,6 to wahanie ok. 8 dB.
pub(crate) const DEFAULT_DEPTH: f32 = 0.6;
/// Domyślna siła cieńszej barwy (0–1) w stylach z oddechem. 1,0 daje kształt pasma zwykłego Oddechu.
pub(crate) const DEFAULT_TIMBRE: f32 = 1.0;
/// Najgłębszy dopuszczalny oddech: przy 1,0 dno cyklu byłoby pełną ciszą.
pub(crate) const MAX_DEPTH: f32 = 0.95;
/// Domyślna siła lekkiego brzęczenia w stylu „Oddech + brzęczenie” (0–1). 0,5 to wahanie 6 dB
/// przy 45 Hz: wyraźna chropowatość na krótkich dźwiękach, ale bez schodzenia do ciszy.
pub(crate) const DEFAULT_BUZZ_DEPTH: f32 = 0.5;

/// Która strona jest przeciwna do granego ucha, czyli którą stronę oznaczamy wskazówką.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opposite {
    Left,
    Right,
}

/// Jedna próbka wyjścia rozdzielania, opóźniona o [`FFT_LEN`] względem wejścia.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SplitSample {
    /// Dźwięk z przeciwnej strony, bez zmian.
    pub(crate) part: f32,
    /// Ten sam dźwięk z cieńszą barwą Oddechu (siła z argumentu `timbre`).
    pub(crate) part_timbre: f32,
    /// Reszta: swoja strona i środek.
    pub(crate) rest: f32,
}

/// Stan rozdzielania kierunkowego dla jednego strumienia wyjściowego. Trzymany między
/// callbackami, bo okno FFT zbiera próbki z kilku kolejnych buforów.
#[derive(Clone)]
pub(crate) struct DirectionalSplit {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// Okno sqrt-Hann, to samo przy analizie i syntezie.
    window: Vec<f32>,
    /// Skala składania ramek: skok / suma kwadratów okna, razem z 1/N odwrotnego FFT.
    ola_scale: f32,
    /// Ostatnie [`FFT_LEN`] próbek L i P (bufor kołowy) i pozycja następnego zapisu.
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    in_pos: usize,
    /// Sumy nakładanych ramek części przeciwnej (bez zmian i z barwą), indeksowane numerem
    /// próbki modulo [`OLA_LEN`].
    ola_part: Vec<f32>,
    ola_part_timbre: Vec<f32>,
    /// Numer następnej próbki wejścia od startu albo ostatniego [`Self::reset`].
    t: u64,
    /// Próbki zebrane od ostatniej ramki.
    since_hop: usize,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    /// Suma L+P każdego paska z bieżącej ramki.
    sum: Vec<Complex<f32>>,
    /// Surowa i wygładzona moc L i P każdego paska.
    raw_l: Vec<f32>,
    raw_r: Vec<f32>,
    pow_l: Vec<f32>,
    pow_r: Vec<f32>,
    /// Wygładzona waga `w` każdego paska (0 = cały do reszty, 1 = cały do części przeciwnej).
    weight: Vec<f32>,
    /// Wzmocnienie cieńszej barwy dla każdego paska przy pełnej sile: kształt filtrów zwykłego
    /// Oddechu razy wyrównanie głośności (patrz [`timbre_curve`]).
    timbre: Vec<f32>,
    /// Współczynniki wygładzania na jedną ramkę.
    power_keep: f32,
    weight_attack_keep: f32,
    weight_release_keep: f32,
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

/// Oddech bez zmiany średniej głośności: mnożnik dla bieżącej próbki. Przesuwa `phase` o jedną
/// próbkę oddechu ([`TREMOLO_HZ`]) i zwraca wartość wahającą się wokół 1,0, od
/// `(1 − depth) / (1 − depth/2)` do `1 / (1 − depth/2)`. Szczyt wypada w tej samej fazie co
/// w `mono_router::am_modulate`.
#[inline]
pub(crate) fn breath_gain(depth: f32, sample_rate: f32, phase: &mut f32) -> f32 {
    *phase = (*phase + TREMOLO_HZ / sample_rate.max(1.0)).fract();
    let lfo = 0.5 * (1.0 - (*phase * std::f32::consts::TAU).sin());
    let depth = depth.clamp(0.0, MAX_DEPTH);
    (1.0 - depth * lfo) / (1.0 - depth * 0.5)
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

/// Wzmocnienie barwy dla każdego paska przy pełnej sile. Kształt to moduł filtrów zwykłego
/// Oddechu (górnoprzepustowy [`TREMOLO_HPF_HZ`] i dolnoprzepustowy [`TREMOLO_LPF_HZ`], oba
/// pierwszego rzędu), czyli cieńsze brzmienie. Całość jest podniesiona tak, żeby szum różowy
/// (widmo podobne do dźwięków gry) był po zmianie tak samo głośny dla ucha (ważenie A): tracimy
/// dół i górę, a środek pasma idzie trochę w górę. Wyrównanie po surowym RMS przesadzałoby, bo
/// liczyłoby bas, którego ucho prawie nie odbiera jako głośności.
fn timbre_curve(sample_rate: f32) -> Vec<f32> {
    let shape: Vec<f32> = (0..BINS)
        .map(|k| {
            let f = k as f32 * sample_rate / FFT_LEN as f32;
            let x = f / TREMOLO_HPF_HZ;
            let y = f / TREMOLO_LPF_HZ;
            (x / (1.0 + x * x).sqrt()) / (1.0 + y * y).sqrt()
        })
        .collect();
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

impl DirectionalSplit {
    pub(crate) fn new(sample_rate: f32) -> Self {
        let sample_rate = sample_rate.max(1.0);
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_LEN);
        let inverse = planner.plan_fft_inverse(FFT_LEN);
        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        let window: Vec<f32> = (0..FFT_LEN)
            .map(|i| (0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT_LEN as f32).cos()).sqrt())
            .collect();
        let window_energy: f32 = window.iter().map(|w| w * w).sum();
        Self {
            forward,
            inverse,
            ola_scale: HOP as f32 / window_energy / FFT_LEN as f32,
            window,
            in_l: vec![0.0; FFT_LEN],
            in_r: vec![0.0; FFT_LEN],
            in_pos: 0,
            ola_part: vec![0.0; OLA_LEN],
            ola_part_timbre: vec![0.0; OLA_LEN],
            t: 0,
            since_hop: 0,
            buf: vec![Complex::new(0.0, 0.0); FFT_LEN],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            sum: vec![Complex::new(0.0, 0.0); BINS],
            raw_l: vec![0.0; BINS],
            raw_r: vec![0.0; BINS],
            pow_l: vec![0.0; BINS],
            pow_r: vec![0.0; BINS],
            weight: vec![0.0; BINS],
            timbre: timbre_curve(sample_rate),
            power_keep: keep_per_hop(POWER_SMOOTH_MS, sample_rate),
            weight_attack_keep: keep_per_hop(WEIGHT_ATTACK_MS, sample_rate),
            weight_release_keep: keep_per_hop(WEIGHT_RELEASE_MS, sample_rate),
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
        self.t = 0;
        self.since_hop = 0;
        self.pow_l.fill(0.0);
        self.pow_r.fill(0.0);
        self.weight.fill(0.0);
    }

    /// Jedna próbka wejścia (L, P) na próbkę wyjścia rozdzielania (patrz [`SplitSample`]),
    /// opóźnioną o [`FFT_LEN`] próbek (pierwsze [`FFT_LEN`] to cisza). `timbre` (0–1) to siła
    /// cieńszej barwy Oddechu w `SplitSample::part_timbre`.
    pub(crate) fn process(&mut self, left: f32, right: f32, opposite: Opposite, timbre: f32) -> SplitSample {
        // Przed nadpisaniem w buforze kołowym stoi wejście sprzed dokładnie FFT_LEN próbek,
        // czyli z tej samej chwili co odczytywana niżej część przeciwna.
        let delayed = self.in_l[self.in_pos] + self.in_r[self.in_pos];
        self.in_l[self.in_pos] = left;
        self.in_r[self.in_pos] = right;
        self.in_pos = (self.in_pos + 1) % FFT_LEN;

        // Próbka sprzed FFT_LEN jest już kompletna: ostatnia ramka, która ją obejmuje, skończyła
        // się najpóźniej próbkę temu. Zapis ramki niżej trafia w FFT_LEN slotów ZA tym.
        let read = ((self.t + FFT_LEN as u64) % OLA_LEN as u64) as usize;
        let (part, part_timbre) = if self.t >= FFT_LEN as u64 {
            (self.ola_part[read], self.ola_part_timbre[read])
        } else {
            (0.0, 0.0)
        };
        self.ola_part[read] = 0.0;
        self.ola_part_timbre[read] = 0.0;

        self.since_hop += 1;
        if self.since_hop == HOP {
            self.since_hop = 0;
            self.process_frame(opposite, timbre.clamp(0.0, 1.0));
        }
        self.t += 1;
        SplitSample { part, part_timbre, rest: delayed - part }
    }

    /// Ramka STFT kończąca się na próbce `self.t`: analiza, waga per pasek, synteza obu części.
    fn process_frame(&mut self, opposite: Opposite, timbre: f32) {
        // L jako część rzeczywista, P jako urojona: jedno FFT liczy oba kanały naraz.
        for i in 0..FFT_LEN {
            let idx = (self.in_pos + i) % FFT_LEN; // in_pos wskazuje teraz najstarszą próbkę
            let w = self.window[i];
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

        let keep = self.power_keep;
        let j = Complex::new(0.0, 1.0);
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
            let target = weight_for_ild(ild_opposite);
            let w_keep = if target > self.weight[k] { self.weight_attack_keep } else { self.weight_release_keep };
            self.weight[k] = w_keep * self.weight[k] + (1.0 - w_keep) * target;
            let w = self.weight[k];

            let color = 1.0 + timbre * (self.timbre[k] - 1.0);
            let mut part = self.sum[k] * w;
            let mut part_timbre = part * color;
            if k == 0 || k == FFT_LEN / 2 {
                // Pasek 0 Hz i Nyquista sygnału rzeczywistego jest rzeczywisty.
                part.im = 0.0;
                part_timbre.im = 0.0;
            }
            // Obie wersje to sygnały rzeczywiste, więc jedno odwrotne FFT liczy je naraz:
            // bez zmian wychodzi jako część rzeczywista, z barwą jako urojona.
            self.buf[k] = part + j * part_timbre;
            if k > 0 && k < FFT_LEN / 2 {
                self.buf[FFT_LEN - k] = part.conj() + j * part_timbre.conj();
            }
        }
        self.inverse.process_with_scratch(&mut self.buf, &mut self.scratch);

        // Ramka obejmuje próbki t-N+1 ..= t; ich sloty to (t + N + 1 + i) mod OLA_LEN.
        let base = ((self.t + FFT_LEN as u64 + 1) % OLA_LEN as u64) as usize;
        for i in 0..FFT_LEN {
            let slot = (base + i) % OLA_LEN;
            let scale = self.window[i] * self.ola_scale;
            self.ola_part[slot] += self.buf[i].re * scale;
            self.ola_part_timbre[slot] += self.buf[i].im * scale;
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

    fn run(left: &[f32], right: &[f32], opposite: Opposite, timbre: f32) -> Vec<SplitSample> {
        let mut s = DirectionalSplit::new(SR);
        left.iter().zip(right).map(|(&l, &r)| s.process(l, r, opposite, timbre)).collect()
    }

    /// `(część przeciwna, reszta)` dla każdej próbki; z `timbre > 0` część z barwą.
    fn split(left: &[f32], right: &[f32], opposite: Opposite, timbre: f32) -> (Vec<f32>, Vec<f32>) {
        run(left, right, opposite, timbre)
            .into_iter()
            .map(|s| (if timbre > 0.0 { s.part_timbre } else { s.part }, s.rest))
            .unzip()
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
    }

    #[test]
    fn opposite_part_lines_up_with_the_delayed_input() {
        // Dźwięk z przeciwnej strony trafia w całości do części przeciwnej, więc ta ma być
        // wejściem L+P co do próbki, opóźnionym o okno. Sprawdza też, że reszta (opóźnione
        // wejście minus część) liczy się z tej samej chwili.
        let (l, r) = pan(&noise(12_000, 1), 9.0);
        let out = run(&l, &r, Opposite::Right, 0.0);
        for t in 4_000..l.len() {
            let expected = l[t - FFT_LEN] + r[t - FFT_LEN];
            assert!((out[t].part - expected).abs() < 1e-3, "próbka {t}: {} zamiast {expected}", out[t].part);
            assert!(out[t].rest.abs() < 1e-3, "próbka {t}: reszta {}", out[t].rest);
        }
    }

    #[test]
    fn part_without_timbre_is_the_same_in_both_versions() {
        let (l, r) = pan(&noise(8_000, 2), 9.0);
        for s in run(&l, &r, Opposite::Right, 0.0) {
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
        let level = |x: &[f32], hz: f32| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, v) in x.iter().enumerate() {
                let ang = std::f32::consts::TAU * hz * i as f32 / SR;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            (re * re + im * im).sqrt()
        };
        let (o, r) = (&opp[12_000..], &rest[12_000..]);
        let high_in_opp_db = 20.0 * (level(o, 3_000.0) / level(r, 3_000.0)).log10();
        let low_in_rest_db = 20.0 * (level(r, 250.0) / level(o, 250.0)).log10();
        assert!(high_in_opp_db > 30.0, "dźwięk z prawej ma być w części przeciwnej: {high_in_opp_db:.1} dB");
        assert!(low_in_rest_db > 30.0, "dźwięk z lewej ma zostać w reszcie: {low_in_rest_db:.1} dB");
    }

    #[test]
    fn thinner_timbre_keeps_loudness_of_pink_noise() {
        let (l, r) = pan(&pink(96_000, 8), 9.0);
        let plain: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        let (opp, _) = split(&l, &r, Opposite::Right, 1.0);
        let diff = a_weighted_db(&opp[24_000..]) - a_weighted_db(&plain[24_000 - FFT_LEN..]);
        assert!(diff.abs() < 1.0, "cieńsza barwa nie powinna ściszać dla ucha: {diff:+.2} dB");
    }

    #[test]
    fn thinner_timbre_cuts_bass_relative_to_the_middle() {
        // Kształt barwy przy pełnej sile: 100 Hz wyraźnie niżej niż 1 kHz, jak w zwykłym Oddechu.
        let curve = timbre_curve(SR);
        let at = |hz: f32| curve[(hz * FFT_LEN as f32 / SR).round() as usize];
        let ratio_db = 20.0 * (at(1_000.0) / at(100.0)).log10();
        assert!(ratio_db > 7.0, "dół ma być ścięty względem środka: {ratio_db:.1} dB");
    }

    #[test]
    fn breath_gain_swings_around_one_without_changing_the_average() {
        let mut phase = 0.0;
        let cycle = (SR / TREMOLO_HZ) as usize;
        let gains: Vec<f32> = (0..cycle).map(|_| breath_gain(DEFAULT_DEPTH, SR, &mut phase)).collect();
        let mean = gains.iter().sum::<f32>() / cycle as f32;
        let max = gains.iter().cloned().fold(f32::MIN, f32::max);
        let min = gains.iter().cloned().fold(f32::MAX, f32::min);
        assert!((mean - 1.0).abs() < 1e-3, "średnia ma zostać 1,0: {mean}");
        let swing_db = 20.0 * (max / min).log10();
        let expected = -20.0 * (1.0 - DEFAULT_DEPTH).log10();
        assert!((swing_db - expected).abs() < 0.1, "wahanie {swing_db:.2} dB zamiast {expected:.2} dB");
    }

    #[test]
    fn silence_stays_silent_and_loud_input_stays_finite() {
        let zeros = vec![0.0; 4_000];
        let (opp, rest) = split(&zeros, &zeros, Opposite::Right, 1.0);
        assert!(opp.iter().chain(&rest).all(|&y| y == 0.0));
        let loud: Vec<f32> = noise(20_000, 9).iter().map(|x| x * 9.0).collect();
        let quiet: Vec<f32> = loud.iter().map(|x| x * 0.1).collect();
        let (opp, rest) = split(&quiet, &loud, Opposite::Right, 1.0);
        assert!(opp.iter().chain(&rest).all(|y| y.is_finite() && y.abs() < 10.0));
    }

    #[test]
    fn reset_drops_everything_from_before() {
        let mut s = DirectionalSplit::new(SR);
        for x in noise(5_000, 10) {
            s.process(x, x * 2.0, Opposite::Right, 1.0);
        }
        s.reset();
        for _ in 0..3 * FFT_LEN {
            assert_eq!(s.process(0.0, 0.0, Opposite::Right, 1.0), SplitSample::default());
        }
    }
}
