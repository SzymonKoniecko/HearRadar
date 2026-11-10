//! Pilnuje opóźnienia bufora mono między wejściem (np. VB-Cable) a wyjściem (słuchawki).
//!
//! Wejście i wyjście to dwa urządzenia z osobnymi zegarami, które chodzą minimalnie różnym
//! tempem (typowo kilkadziesiąt ppm). Dawniej bufor przez to powoli rósł do progu ok. 150 ms
//! i był skokowo przycinany do 50 ms, więc zwykle trzymał ok. 100 ms opóźnienia.
//!
//! Teraz:
//! - cel opóźnienia jest mały i liczony z tego, czego naprawdę wymagają urządzenia: największy
//!   kawałek wejścia plus połowa największego odczytu wyjścia plus zapas (przy typowych 10 ms
//!   po obu stronach średnio ok. 18 ms),
//! - dryf zegarów jest korygowany na bieżąco: odczyt bierze z bufora odrobinę więcej albo mniej
//!   próbek, niż oddaje (najwyżej ±0,5 %, w praktyce setne części procenta), z interpolacją,
//!   więc bufor stoi na celu i nie trzeba go przycinać,
//! - gdy bufor mimo to wyschnie (przerwa w dźwięku), zapas rośnie o [`UNDERRUN_STEP_MS`],
//!   a po dłuższym czasie bez przerw wolno wraca.
//!
//! Wszystkie wielkości są w próbkach wejścia.

/// Zapas ponad to, czego wymagają kawałki wejścia i wyjścia, na starcie (ms).
const MARGIN_START_MS: f32 = 3.0;
/// O tyle rośnie zapas po każdej przerwie (bufor wysechł), do [`MARGIN_MAX_MS`].
pub(crate) const UNDERRUN_STEP_MS: f32 = 2.0;
const MARGIN_MAX_MS: f32 = 40.0;
/// Po tylu sekundach bez przerwy zapas maleje o [`MARGIN_DECAY_MS`], do [`MARGIN_START_MS`].
const MARGIN_DECAY_AFTER_S: f32 = 30.0;
const MARGIN_DECAY_MS: f32 = 0.5;
/// Granice krótkiego celu opóźnienia (ms).
const SHORT_TARGET_MIN_MS: f32 = 2.0;
pub(crate) const SHORT_TARGET_MAX_MS: f32 = 50.0;
/// Jak długo pamiętamy największy kawałek wejścia i odczyt wyjścia (stała czasu, s).
const PEAK_MEMORY_S: f32 = 10.0;
/// Uśrednianie zapełnienia bufora (stała czasu, s).
const FILL_AVERAGE_S: f32 = 0.5;
/// W jakim czasie korekta tempa usuwa różnicę od celu (s). Razem z [`FILL_AVERAGE_S`] daje
/// tłumienie krytyczne: bufor dochodzi do celu bez przestrzelenia.
const CORRECTION_S: f32 = 2.0;
/// Największa korekta tempa odczytu (±0,5 %, ok. 9 centów, niesłyszalne).
const MAX_CORRECTION: f64 = 0.005;
/// Pierwsze pół sekundy odczytu tylko mierzymy, bez korekty i bez karania przerw (start).
const WARMUP_S: f32 = 0.5;

/// Stan opóźnienia bufora do pokazania w UI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonoLatency {
    /// Średnie zapełnienie bufora, czyli średni czas czekania próbki (ms). `None` przed pomiarem.
    pub buffer_ms: Option<f32>,
    /// Cel (ms).
    pub target_ms: f32,
    /// Przerwy (bufor wysechł) od startu.
    pub underruns: u64,
    /// Bieżąca korekta tempa odczytu (ppm, dodatnia = szybciej).
    pub correction_ppm: f32,
}

/// Sterowanie opóźnieniem bufora mono (patrz opis modułu).
#[derive(Debug, Clone)]
pub(crate) struct RingClock {
    sample_rate: f32,
    in_peak: f32,
    out_peak: f32,
    margin: f32,
    /// Próbki odczytane od ostatniej przerwy albo zmniejszenia zapasu.
    clean: f32,
    fill_avg: Option<f32>,
    /// Próbki odczytane od startu (do [`WARMUP_S`]).
    warm: f32,
    correction: f64,
    underruns: u64,
}

impl Default for RingClock {
    fn default() -> Self {
        let sample_rate = 48_000.0;
        Self {
            sample_rate,
            in_peak: 0.0,
            out_peak: 0.0,
            margin: ms_to_samples(MARGIN_START_MS, sample_rate),
            clean: 0.0,
            fill_avg: None,
            warm: 0.0,
            correction: 0.0,
            underruns: 0,
        }
    }
}

fn ms_to_samples(ms: f32, sample_rate: f32) -> f32 {
    ms * 0.001 * sample_rate
}

impl RingClock {
    /// Częstotliwość próbkowania wejścia (Hz).
    pub(crate) fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Najstarsza wartość z pamięci szczytów maleje z czasem `samples` próbek.
    fn decay(&self, samples: f32) -> f32 {
        (-samples / (PEAK_MEMORY_S * self.sample_rate)).exp()
    }

    /// Wejście dopisało `chunk` próbek przy częstotliwości `sample_rate`.
    pub(crate) fn note_input(&mut self, chunk: usize, sample_rate: f32) {
        if sample_rate.is_finite() && sample_rate > 0.0 && sample_rate != self.sample_rate {
            let scale = sample_rate / self.sample_rate;
            self.margin *= scale;
            self.sample_rate = sample_rate;
        }
        let chunk = chunk as f32;
        self.in_peak = chunk.max(self.in_peak * self.decay(chunk));
    }

    /// Krótki cel średniego zapełnienia (próbki): cały największy kawałek wejścia, połowa
    /// największego odczytu wyjścia i zapas.
    ///
    /// Kawałek wejścia liczy się w całości, bo zegary powoli przesuwają się względem siebie:
    /// gdy wejście zaczyna przychodzić tuż PO odczycie zamiast tuż przed nim, zapełnienie przy
    /// odczycie spada naraz o cały kawałek. Z tym celem zostaje wtedy jeszcze cały odczyt plus
    /// zapas, więc bufor nie wysycha.
    pub(crate) fn short_target(&self) -> f32 {
        let wanted = self.in_peak + 0.5 * self.out_peak + self.margin;
        wanted.clamp(
            ms_to_samples(SHORT_TARGET_MIN_MS, self.sample_rate),
            ms_to_samples(SHORT_TARGET_MAX_MS, self.sample_rate),
        )
    }

    /// Odczyt `need` próbek wejścia przy zapełnieniu `fill` (przed odczytem) i celu `target`.
    /// Zwraca korektę tempa odczytu: dodatnia = brać szybciej, bo bufor jest ponad celem.
    pub(crate) fn on_read(&mut self, fill: f32, need: f32, target: f32) -> f64 {
        self.out_peak = need.max(self.out_peak * self.decay(need));
        // Średnio w trakcie tego odczytu w buforze jest o połowę odczytu mniej niż przed nim.
        let mid = fill - 0.5 * need;
        let alpha = (need / (FILL_AVERAGE_S * self.sample_rate)).min(1.0);
        let avg = match self.fill_avg {
            Some(avg) => avg + alpha * (mid - avg),
            None => mid,
        };
        self.fill_avg = Some(avg);

        self.clean += need;
        if self.clean >= MARGIN_DECAY_AFTER_S * self.sample_rate {
            self.clean = 0.0;
            let start = ms_to_samples(MARGIN_START_MS, self.sample_rate);
            self.margin = (self.margin - ms_to_samples(MARGIN_DECAY_MS, self.sample_rate)).max(start);
        }

        if self.warm < WARMUP_S * self.sample_rate {
            self.warm += need;
            self.correction = 0.0;
        } else {
            let error = f64::from(avg - target) / f64::from(CORRECTION_S * self.sample_rate);
            self.correction = error.clamp(-MAX_CORRECTION, MAX_CORRECTION);
        }
        self.correction
    }

    /// Bufor wysechł. Po rozruchu zapas rośnie, żeby to się nie powtarzało.
    pub(crate) fn on_underrun(&mut self) {
        self.underruns += 1;
        self.clean = 0.0;
        self.fill_avg = None;
        if self.warm >= WARMUP_S * self.sample_rate {
            let max = ms_to_samples(MARGIN_MAX_MS, self.sample_rate);
            self.margin = (self.margin + ms_to_samples(UNDERRUN_STEP_MS, self.sample_rate)).min(max);
        }
    }

    /// Bufor skokowo zmienił zapełnienie (dopełnienie po przerwie, przycięcie): średnią liczymy
    /// od nowa, zamiast powoli doganiać skok.
    pub(crate) fn restart_average(&mut self) {
        self.fill_avg = None;
    }

    /// Stan dla UI przy celu `target` (próbki).
    pub(crate) fn status(&self, target: f32) -> MonoLatency {
        let to_ms = |samples: f32| samples / self.sample_rate * 1000.0;
        MonoLatency {
            buffer_ms: self.fill_avg.map(|avg| to_ms(avg.max(0.0))),
            target_ms: to_ms(target),
            underruns: self.underruns,
            correction_ppm: (self.correction * 1e6) as f32,
        }
    }
}

/// Interpolacja Hermite'a 3. rzędu (Catmull-Rom) między `x0` i `x1` dla `t` w 0..1, z sąsiadami
/// `xm1` i `x2`. Dla `t = 0` zwraca dokładnie `x0`.
#[inline]
pub(crate) fn hermite(xm1: f32, x0: f32, x1: f32, x2: f32, t: f32) -> f32 {
    let c1 = 0.5 * (x1 - xm1);
    let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
    let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
    ((c3 * t + c2) * t + c1) * t + x0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// Zegar po rozruchu, z kawałkami po `chunk` próbek po obu stronach.
    fn warmed(chunk: usize) -> RingClock {
        let mut clock = RingClock::default();
        for _ in 0..200 {
            clock.note_input(chunk, SR);
            clock.on_read(1_000.0, chunk as f32, 1_000.0);
        }
        clock
    }

    #[test]
    fn short_target_follows_the_device_chunks() {
        // 10 ms po obu stronach: 10 + 5 ms plus 3 ms zapasu.
        let clock = warmed(480);
        assert!((clock.short_target() - (480.0 + 240.0 + 144.0)).abs() < 1.0, "cel {}", clock.short_target());
        // Krótsze kawałki: mniejszy cel, ale nie poniżej 2 ms.
        let mut small = RingClock::default();
        small.note_input(32, SR);
        small.on_read(0.0, 32.0, 0.0);
        assert!((small.short_target() - 192.0).abs() < 1.0, "cel {}", small.short_target());
        let mut tiny = RingClock { margin: 0.0, ..RingClock::default() };
        tiny.note_input(8, SR);
        assert!((tiny.short_target() - 96.0).abs() < 1e-3, "najmniej 2 ms: {}", tiny.short_target());
    }

    #[test]
    fn correction_speeds_up_above_the_target_and_slows_down_below_it() {
        let mut clock = warmed(480);
        clock.restart_average();
        let above = clock.on_read(1_000.0 + 240.0 + 96.0, 480.0, 1_000.0);
        clock.restart_average();
        let below = clock.on_read(1_000.0 + 240.0 - 96.0, 480.0, 1_000.0);
        assert!(above > 0.0 && below < 0.0, "nad celem szybciej, pod celem wolniej: {above} {below}");
        assert!((above - 96.0 / 96_000.0).abs() < 1e-6, "2 ms różnicy usuwane w 2 s: {above}");
        clock.restart_average();
        assert_eq!(clock.on_read(1e6, 480.0, 1_000.0), MAX_CORRECTION, "korekta ma granicę");
    }

    #[test]
    fn no_correction_and_no_penalty_during_warmup() {
        let mut clock = RingClock::default();
        assert_eq!(clock.on_read(10_000.0, 480.0, 500.0), 0.0);
        let margin = clock.margin;
        clock.on_underrun();
        assert_eq!(clock.margin, margin, "przerwa na starcie nie podnosi zapasu");
        assert_eq!(clock.underruns, 1);
    }

    #[test]
    fn underruns_raise_the_margin_and_it_slowly_comes_back() {
        let mut clock = warmed(480);
        let start = clock.margin;
        clock.on_underrun();
        clock.on_underrun();
        assert!((clock.margin - (start + 192.0)).abs() < 1e-3, "dwie przerwy: +4 ms");
        // 30 s bez przerw: −0,5 ms.
        for _ in 0..3_000 {
            clock.on_read(1_000.0, 480.0, 1_000.0);
        }
        assert!((clock.margin - (start + 192.0 - 24.0)).abs() < 1e-3, "zapas {}", clock.margin);
    }

    #[test]
    fn hermite_passes_through_samples_and_follows_a_smooth_curve() {
        assert_eq!(hermite(0.1, 0.3, -0.2, 0.7, 0.0), 0.3);
        assert!((hermite(0.1, 0.3, -0.2, 0.7, 1.0) - -0.2).abs() < 1e-6);
        // Sinus 1 kHz przy 48 kHz: między próbkami błąd poniżej 0,1 %.
        let x = |i: f32| (std::f32::consts::TAU * 1_000.0 * i / SR).sin();
        for k in 0..48 {
            for t in [0.25f32, 0.5, 0.75] {
                let i = k as f32;
                let y = hermite(x(i - 1.0), x(i), x(i + 1.0), x(i + 2.0), t);
                assert!((y - x(i + t)).abs() < 1e-3, "próbka {k} + {t}");
            }
        }
    }
}
