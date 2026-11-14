//! Mono router — odtwarza w jednej słuchawce (albo obu) zmiksowany sygnał
//! ze wszystkich kanałów, korzystając z TEGO SAMEGO strumienia wejściowego,
//! który już zasila radar. Nie otwiera własnego, niezależnego przechwytywania
//! wejścia — to tylko strumień WYJŚCIOWY, czytający gotowe próbki "ważonego
//! mono" z bufora kołowego w `SharedAudioAnalysis` (patrz
//! `push_stereo_pan`/`read_stereo_pan_tagged` w `sound::devices`), do którego pisze
//! `update_analysis` przy każdej ramce audio.
//!
//! Przepływ sygnału:
//!   [urządzenie wejściowe] ──► input callback (`devices::fetch_audio_from_device`)
//!         │
//!         ├──► radar (kropki per-kanał, kierunek — bez zmian)
//!         │
//!         └──► downmix do (left, right) ──► RingBuffer w `SharedAudioAnalysis`
//!                                       │
//!                                 output callback ──► [słuchawki]
//!                                       L              R
//!                                    ┌───────────────────────────┐
//!                         Left  →    │ L+ringmod(R)      0       │
//!                         Right →    │      0       R+ringmod(L) │
//!                         Both  →    │    L+R            L+R     │
//!                                    └───────────────────────────┘
//! W trybach jednousznych (`Left`/`Right`) strona PRZECIWNA do granego ucha
//! nie jest po prostu domiksowana na płasko — dostaje ring modulation (patrz
//! `fill_output_f32`), żeby w jednej słuchawce dało się usłyszeć, że dźwięk
//! oryginalnie był z drugiej strony, zamiast bezpowrotnie tracić informację
//! L/R już na etapie miksu do mono.
//!
//! Style kierunkowe (`MonoCueStyle::is_directional`) nie modulują kanału, tylko
//! dźwięk, który przychodzi z przeciwnej strony: `sound::directional` dzieli go
//! osobno w każdym paśmie częstotliwości na część przeciwną i resztę, a efekt
//! dostaje tylko część przeciwna (patrz [`mix_for_ear`]).
//!
//! Styl „Wg kierunku” (`MonoCueStyle::PerDirection`) dokłada drugą oś. Bufor niesie też przód
//! i tył każdej ramki (`devices::ZoneFrame`, z kanałów 5.1/7.1). Przód i tył dostają swoje style
//! jeszcze przed rozdzielaniem ([`apply_zones`]), tak samo w lewym i prawym kanale, więc
//! rozdzielanie dalej widzi tę samą różnicę stron, a strona przeciwna dostaje swój styl na
//! wierzch. Dźwięk z tyłu po przeciwnej stronie ma więc oba.
//!
//! Wcześniej `mono_router` otwierał WŁASNY, drugi strumień wejściowy na
//! nazwie urządzenia z tej samej listy co radar. Na Windows ta lista
//! (`devices::fetch_output_devices`) zawiera nazwy urządzeń WYJŚCIOWYCH (do
//! przechwytywania pętlą zwrotną WASAPI) — a lookup szukał ich tylko wśród
//! `host.input_devices()`, więc zawsze zawodził i routing w ogóle nie
//! startował (błąd widoczny tylko w konsoli). Ujednolicenie do jednego
//! strumienia wejściowego usuwa ten błąd u źródła, zamiast łatać sam lookup.

use anyhow::{Result, bail};
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::sound::devices::{
    CueDirection, MonoCueStyle, MonoDirectionStyles, MonoEar, MonoLayerStyles, SharedAudioAnalysis, SideLayers,
    ZoneFrame,
};
use crate::sound::directional::{self, DirectionalSplit, Opposite, SplitParams, SplitSample};
use crate::sound::layers::Layer;

/// Częstotliwość (Hz) ring modulation dla [`MonoCueStyle::Buzz`] — na tyle
/// nisko, żeby brzmieć jak wyraźna, chropowata wibracja odróżnialna od
/// czystego dźwięku, a nie po prostu "wyższy ton".
const RING_MOD_BUZZ_HZ: f32 = 45.0;

/// Jak [`RING_MOD_BUZZ_HZ`], ale dla [`MonoCueStyle::Squeaky`] — ten sam
/// mechanizm (ring modulation), po prostu szybciej, więc brzmi wyżej.
///
/// Wcześniej 220 Hz przy pełnej głębokości modulacji ([`RING_MOD_SQUEAKY_DEPTH`]
/// = 1.0) brzmiało nieprzyjemnie piskliwie/metalicznie — obniżone do 140 Hz,
/// żeby zostać wyraźnie ponad `Buzz` (wciąż jednoznacznie odróżnialne, patrz
/// test `squeaky_oscillates_faster_than_buzz`), ale poza najbardziej
/// drażniącym dla ucha zakresem.
const RING_MOD_SQUEAKY_HZ: f32 = 140.0;

/// Głębokość modulacji amplitudy (0.0 = brak efektu, 1.0 = pełne zejście do
/// ciszy na dnie każdego cyklu, jak wcześniej) dla [`MonoCueStyle::Squeaky`].
/// Pełna głębokość przy szybkim oscylatorze brzmi jak ostre bramkowanie —
/// obniżenie do części zakresu (obwiednia nigdy nie schodzi poniżej ~45%
/// oryginalnej amplitudy) zaokrągla brzmienie do wyraźnego, ale
/// znośniejszego "trzepotania", zamiast przeszywającego pisku.
const RING_MOD_SQUEAKY_DEPTH: f32 = 0.55;

/// Odcięcie (Hz) filtra dolnoprzepustowego, którego DOPEŁNIENIE
/// (`sample - lowpass`) daje składową wysokotonową dla [`MonoCueStyle::Bright`]
/// — dodaną z powrotem do oryginału.
const BRIGHT_HIGHPASS_CUTOFF_HZ: f32 = 800.0;

/// O ile wzmocnić składową wysokotonową w [`MonoCueStyle::Bright`] — na tyle
/// mocno, żeby zmiana barwy była jednoznacznie słyszalna, ale bez ostrego
/// przesterowania przy typowych, niegłośnych sygnałach z radaru.
const BRIGHT_BOOST_GAIN: f32 = 1.5;

/// Grzebień ([`MonoCueStyle::Comb`]): opóźnienie w sekundach, odcięcie basu w Hz, wzmocnienie
/// liniowo (0,794 ≈ −2 dB).
const MAX_DELAY_SAMPLES: usize = 2_048;
const COMB_DELAY_S: f32 = 0.0012;
const COMB_HPF_HZ: f32 = 150.0;
const COMB_GAIN: f32 = 0.794;
/// Oddech: częstotliwość i filtry są wspólne ze stylami kierunkowymi z oddechem
/// (`sound::directional`), żeby brzmiały tak samo.
pub(crate) const TREMOLO_HZ: f32 = 4.0;
const TREMOLO_DEPTH: f32 = 0.4;
pub(crate) const TREMOLO_HPF_HZ: f32 = 300.0;
pub(crate) const TREMOLO_LPF_HZ: f32 = 4_000.0;

/// Wyrównanie głośności Jasnego w wersji kierunkowej: dźwięk z przeciwnej strony dostaje
/// cały efekt, który dla szumu różowego jest ok. 6,9 dB głośniejszy dla ucha (ważenie A).
/// Ściszamy o tyle, żeby zostało samo rozjaśnienie. Sprawdza to test
/// `directional_styles_keep_the_loudness_of_the_opposite_side`.
const BRIGHT_DIRECTIONAL_MAKEUP: f32 = 0.452;
/// Jak [`BRIGHT_DIRECTIONAL_MAKEUP`], dla Grzebienia: cały efekt jest ok. 5,3 dB cichszy.
const COMB_DIRECTIONAL_MAKEUP: f32 = 1.84;

/// Jasny „w dół”: największe wzmocnienie efektu to `1 + BRIGHT_BOOST_GAIN` (2,5 przy górze pasma),
/// więc dzielimy przez nie. Góra zostaje na poziomie oryginału, reszta jest ściszona.
const BRIGHT_PEAK_MAKEUP: f32 = 1.0 / (1.0 + BRIGHT_BOOST_GAIN);

/// Limiter z pierwszeństwem dla strony własnej: wyprzedzenie (ms), narastanie i powrót ściszenia
/// strony przeciwnej (ms), zanik obwiedni szczytowych (ms) i najmocniejsze ściszenie (dB).
/// Narastanie jest dwa razy krótsze niż wyprzedzenie, więc ściszenie dochodzi w ok. 86 % przed
/// szczytem; resztę łapie [`soft_limit`].
const PRIORITY_LOOKAHEAD_MS: f32 = 1.0;
const PRIORITY_ATTACK_MS: f32 = 0.5;
const PRIORITY_RELEASE_MS: f32 = 80.0;
const PRIORITY_ENV_RELEASE_MS: f32 = 20.0;
const PRIORITY_MIN_GAIN_DB: f32 = -18.0;

/// Wygładzenie świeżości (ms) przed użyciem w miksie, żeby skok przy początku dźwięku nie trzaskał.
const FRESH_SMOOTH_MS: f32 = 3.0;

/// Bufor opóźnienia: zapisuje próbkę i zwraca tę sprzed `delay_samples` (interpolacja liniowa).
#[inline]
fn delay_line_step(buf: &mut [f32], pos: &mut usize, sample: f32, delay_samples: f32) -> f32 {
    let len = buf.len();
    buf[*pos] = sample;
    let here = *pos as f32;
    *pos = (*pos + 1) % len;
    let read = (here - delay_samples).rem_euclid(len as f32);
    let i0 = read.floor() as usize % len;
    let i1 = (i0 + 1) % len;
    let frac = read - read.floor();
    buf[i0] * (1.0 - frac) + buf[i1] * frac
}

#[cfg(test)]
mod cue_style_tests {
    use super::*;

    #[test]
    fn channel_styles_stay_finite_and_bounded() {
        for style in [
            MonoCueStyle::Buzz,
            MonoCueStyle::Squeaky,
            MonoCueStyle::Bright,
            MonoCueStyle::Comb,
            MonoCueStyle::Tremolo,
        ] {
            let mut state = EffectState::default();
            for i in 0..10_000 {
                let x = (i as f32 * 0.01).sin() * 0.8;
                let y = apply_cue(style, x, 48_000.0, &mut state);
                assert!(y.is_finite() && y.abs() < 2.0, "{style:?}: {y}");
            }
        }
    }
}

/// Stan jednego efektu wskazówki (oscylatory, filtry, bufor opóźnienia), trzymany MIĘDZY
/// kolejnymi wywołaniami callbacku wyjściowego — inaczej faza oscylatora (`phase`) i stan
/// filtra (`lp`) resetowałyby się co bufor, dając trzaski na granicach zamiast płynnego efektu.
#[derive(Debug, Clone)]
struct EffectState {
    /// Faza oscylatora (0.0..=1.0) dla `Buzz`/`Squeaky` i LFO dla `Tremolo`, a w stylach
    /// kierunkowych dla brzęczenia.
    phase: f32,
    /// Faza oddechu w stylach kierunkowych z oddechem (osobna od `phase`, bo „Oddech +
    /// brzęczenie” ma dwa oscylatory naraz).
    breath_phase: f32,
    /// Stan filtrów jednobiegunowych (`Bright`, `Comb`, `Tremolo`).
    lp: f32,
    /// Drugi filtr jednobiegunowy (górnoprzepustowy jako x − LP(x) w `Tremolo`).
    lp2: f32,
    /// Pozycja zapisu w buforze opóźnienia dla `Comb`.
    pos: usize,
    delay: Vec<f32>,
}

impl Default for EffectState {
    fn default() -> Self {
        Self { phase: 0.0, breath_phase: 0.0, lp: 0.0, lp2: 0.0, pos: 0, delay: vec![0.0; MAX_DELAY_SAMPLES] }
    }
}

impl EffectState {
    /// Zeruje stan bez nowej alokacji (wołane w callbacku audio).
    fn clear(&mut self) {
        self.phase = 0.0;
        self.breath_phase = 0.0;
        self.lp = 0.0;
        self.lp2 = 0.0;
        self.pos = 0;
        self.delay.fill(0.0);
    }
}

/// Jak szybko styl „Wg warstw” przechodzi między stylami warstw (ms). Wszystkie trzy wagi mają
/// tę samą stałą czasu, więc ich suma zostaje 1 i przejście nie zmienia głośności.
const LAYER_FADE_MS: f32 = 8.0;

/// Stan wskazówki dla strony przeciwnej (patrz [`apply_cue`]/[`fill_output_f32`]), trzymany
/// przez wywołującego między callbackami.
#[derive(Debug, Clone)]
struct CueState {
    /// Efekt stylu kanałowego albo pojedynczego stylu kierunkowego.
    effect: EffectState,
    /// Rozdzielanie kierunkowe (okno FFT obu kanałów) dla stylów kierunkowych, patrz [`mix_for_ear`].
    directional: DirectionalSplit,
    /// Czy poprzedni callback używał stylu kierunkowego. Po przerwie zerujemy okno FFT.
    directional_active: bool,
    /// Styl „Wg warstw”: osobny efekt dla każdej warstwy (kolejność `layers::Layer::ALL`).
    /// Liczone są wszystkie trzy naraz, żeby przejście między nimi było płynne.
    layer_effects: [EffectState; 3],
    /// Bieżące wagi efektów warstw (suma 1), wygładzane przez [`LAYER_FADE_MS`].
    layer_weights: [f32; 3],
    /// Styl „Wg kierunku”: efekty przodu i tyłu, osobno dla lewego i prawego kanału strefy.
    /// Oba kanały idą krok w krok, więc oscylatory mają tę samą fazę.
    zone_effects: [[EffectState; 2]; 2],
    /// Wyrównania barwy stref: `(z wyrównaniem głośności, „w dół”)`, patrz [`zone_timbre`].
    zone_timbre_makeups: (f32, f32),
    /// Znaczniki warstw opóźnione o okno FFT, żeby zgadzały się z wyjściem rozdzielania.
    tag_delay: Vec<SideLayers>,
    tag_pos: usize,
    /// Współczynnik wygładzania wag warstw na jedną próbkę.
    layer_fade: f32,
    /// Stan korekcji tonu: strony własnej (w stylach kanałowych całej sumy) i przeciwnej.
    /// Filtr jest liniowy, więc dwa osobne dają tę samą sumę co jeden.
    tone: ToneState,
    tone_opp: ToneState,
    /// Świeżość dźwięku z przeciwnej strony po wygładzeniu (patrz `SplitSample::fresh`).
    fresh: f32,
    fresh_smooth: f32,
    /// Limiter z pierwszeństwem dla strony własnej (style kierunkowe).
    limiter: PriorityLimiter,
}

/// Wagi warstw na starcie: cały dźwięk z przeciwnej strony ma styl reszty.
const REST_ONLY: [f32; 3] = [0.0, 0.0, 1.0];

/// Współczynnik wygładzania jednobiegunowego na jedną próbkę dla stałej czasu `ms`.
fn smoothing_per_sample(ms: f32, sample_rate: f32) -> f32 {
    1.0 - (-1.0 / (ms * 0.001 * sample_rate.max(1.0))).exp()
}

impl CueState {
    fn new(sample_rate: f32) -> Self {
        Self {
            effect: EffectState::default(),
            directional: DirectionalSplit::new(sample_rate),
            directional_active: false,
            layer_effects: [EffectState::default(), EffectState::default(), EffectState::default()],
            layer_weights: REST_ONLY,
            zone_effects: Default::default(),
            zone_timbre_makeups: directional::timbre_makeups(sample_rate),
            tag_delay: vec![SideLayers::default(); directional::LATENCY],
            tag_pos: 0,
            layer_fade: smoothing_per_sample(LAYER_FADE_MS, sample_rate),
            tone: ToneState::default(),
            tone_opp: ToneState::default(),
            fresh: 0.0,
            fresh_smooth: smoothing_per_sample(FRESH_SMOOTH_MS, sample_rate),
            limiter: PriorityLimiter::new(sample_rate),
        }
    }

    /// Zeruje stan stylów kierunkowych po przerwie (inny styl, inne ucho).
    fn reset_directional(&mut self) {
        self.directional.reset();
        self.tag_delay.fill(SideLayers::default());
        self.tag_pos = 0;
        self.layer_weights = REST_ONLY;
        for effect in self.zone_effects.iter_mut().flatten() {
            effect.clear();
        }
        self.fresh = 0.0;
        self.tone_opp = ToneState::default();
        self.limiter.reset();
    }

    /// Zapisuje znaczniki bieżącej ramki i zwraca te sprzed okna FFT, czyli z tej samej chwili
    /// co próbka, która właśnie wychodzi z rozdzielania.
    fn delayed_tags(&mut self, tags: SideLayers) -> SideLayers {
        let out = self.tag_delay[self.tag_pos];
        self.tag_delay[self.tag_pos] = tags;
        self.tag_pos = (self.tag_pos + 1) % self.tag_delay.len();
        out
    }
}

impl Default for CueState {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

/// Jeden krok oscylatora modulacji amplitudy: przesuwa `state.phase` o
/// `freq_hz` i zwraca `sample` przemnożoną przez obwiednię sinusoidalną w
/// zakresie `[1 - depth, 1]` — `depth = 1.0` schodzi na dnie cyklu do
/// całkowitej ciszy (jak dawne `Buzz`/`Squeaky`), niższe wartości zostawiają
/// resztkę sygnału słyszalną cały czas, co brzmi łagodniej.
#[inline]
fn am_modulate(sample: f32, freq_hz: f32, depth: f32, sample_rate: f32, phase: &mut f32) -> f32 {
    *phase = (*phase + freq_hz / sample_rate).fract();
    let envelope = 1.0 - depth * 0.5 * (1.0 - (*phase * std::f32::consts::TAU).sin());
    sample * envelope
}

/// Jeden krok jednobiegunowego filtra dolnoprzepustowego, ze stanem `lp`
/// trzymanym MIĘDZY callbackami (w przeciwieństwie do
/// `sound::spectrum::one_pole_lowpass`, który resetuje stan co krótką ramkę
/// analizy — tu filtrujemy ciągły strumień wyjściowy). Zwraca przefiltrowaną
/// próbkę.
#[inline]
fn one_pole_lowpass_step(sample: f32, sample_rate: f32, cutoff_hz: f32, lp: &mut f32) -> f32 {
    let dt = 1.0 / sample_rate;
    let rc = 1.0 / (std::f32::consts::TAU * cutoff_hz);
    let alpha = dt / (rc + dt);
    *lp += alpha * (sample - *lp);
    *lp
}

/// Nakłada efekt `style` na próbkę ze strony przeciwnej do granego ucha —
/// patrz [`MonoCueStyle`] po opis każdego wariantu. `Off` zwraca próbkę bez
/// zmian (stare zachowanie sprzed tej funkcji). Styl kierunkowy daje tu efekt
/// swojego stylu kanałowego ([`MonoCueStyle::base_cue`]); w [`mix_for_ear`]
/// trafia do niego tylko część dźwięku z przeciwnej strony.
#[inline]
fn apply_cue(style: MonoCueStyle, sample: f32, sample_rate: f32, state: &mut EffectState) -> f32 {
    let sample_rate = sample_rate.max(1.0);
    match style.base_cue() {
        MonoCueStyle::Buzz => am_modulate(sample, RING_MOD_BUZZ_HZ, 1.0, sample_rate, &mut state.phase),
        MonoCueStyle::Squeaky => {
            am_modulate(sample, RING_MOD_SQUEAKY_HZ, RING_MOD_SQUEAKY_DEPTH, sample_rate, &mut state.phase)
        }
        MonoCueStyle::Comb => {
            let delayed = delay_line_step(&mut state.delay, &mut state.pos, sample, COMB_DELAY_S * sample_rate);
            let mixed = 0.5 * sample + 0.5 * delayed;
            let bass = one_pole_lowpass_step(mixed, sample_rate, COMB_HPF_HZ, &mut state.lp);
            (mixed - bass) * COMB_GAIN
        }
        MonoCueStyle::Tremolo => {
            let modulated = am_modulate(sample, TREMOLO_HZ, TREMOLO_DEPTH, sample_rate, &mut state.phase);
            let bass = one_pole_lowpass_step(modulated, sample_rate, TREMOLO_HPF_HZ, &mut state.lp2);
            one_pole_lowpass_step(modulated - bass, sample_rate, TREMOLO_LPF_HZ, &mut state.lp)
        }
        MonoCueStyle::Bright => {
            // Dopełnienie filtra dolnoprzepustowego (`sample - lowpass`) to
            // składowa wysokotonowa — dodajemy ją z powrotem do oryginału ze
            // wzmocnieniem, więc wysokie tony stają się głośniejsze względem
            // reszty widma.
            let low = one_pole_lowpass_step(sample, sample_rate, BRIGHT_HIGHPASS_CUTOFF_HZ, &mut state.lp);
            let high = sample - low;
            sample + high * BRIGHT_BOOST_GAIN
        }
        // Wyłączone, a style z oddechem idą osobną drogą w `mix_for_ear`.
        _ => sample,
    }
}

/// Wyrównanie głośności modulacji `am_modulate` o głębokości `depth`: mnożnik, po którym
/// średnia moc wraca do tej sprzed modulacji. Obwiednia `1 − d/2 + (d/2)·sin` ma średnią
/// moc `(1 − d/2)² + d²/8`.
fn am_makeup(depth: f32) -> f32 {
    let mean = 1.0 - depth * 0.5;
    let swing = depth * 0.5;
    1.0 / (mean * mean + swing * swing * 0.5).sqrt().max(1e-6)
}

/// Wyrównanie stylu kierunkowego. Dźwięk z przeciwnej strony dostaje cały efekt, a potem:
/// - `subtractive` (wskazówki tylko „w dół”): największe wzmocnienie efektu to 1, więc dźwięk
///   nigdzie i nigdy nie jest głośniej niż bez wskazówki, a średnio jest ciszej,
/// - inaczej: ma brzmieć średnio tak samo głośno jak bez efektu (zmienia się tylko charakter).
fn directional_makeup(style: MonoCueStyle, subtractive: bool) -> f32 {
    match (style, subtractive) {
        (MonoCueStyle::DirectionalBright, true) => BRIGHT_PEAK_MAKEUP,
        (_, true) => 1.0,
        (MonoCueStyle::DirectionalBuzz, false) => am_makeup(1.0),
        (MonoCueStyle::DirectionalSqueaky, false) => am_makeup(RING_MOD_SQUEAKY_DEPTH),
        (MonoCueStyle::DirectionalBright, false) => BRIGHT_DIRECTIONAL_MAKEUP,
        (MonoCueStyle::DirectionalComb, false) => COMB_DIRECTIONAL_MAKEUP,
        _ => 1.0,
    }
}

/// Grzebień „w dół”: `HP₁₅₀[(1 − γ/2)·x + (γ/2)·x(t − 1,2 ms)]`. Szczyty na poziomie oryginału,
/// dołki (ok. 417, 1250, 2083… Hz) o głębokości `20·log10(1 − γ)` dB. Szerokie dołki dają
/// rozpoznawalną „pustą” barwę, a energii nic nie dokłada.
#[inline]
fn comb_dips(sample: f32, gamma: f32, sample_rate: f32, state: &mut EffectState) -> f32 {
    let sample_rate = sample_rate.max(1.0);
    let delayed = delay_line_step(&mut state.delay, &mut state.pos, sample, COMB_DELAY_S * sample_rate);
    let mixed = (1.0 - gamma * 0.5) * sample + gamma * 0.5 * delayed;
    let bass = one_pole_lowpass_step(mixed, sample_rate, COMB_HPF_HZ, &mut state.lp);
    mixed - bass
}

/// Siła wskazówki i dodatkowy poziom dla stylu przy świeżości `fresh` (1 tuż po początku dźwięku
/// z przeciwnej strony, potem 0): `(intensywność, poziom)`. Style z
/// [`MonoCueStyle::fades_after_onset`] po początku dźwięku spadają do `cue_tail`, a dźwięk do
/// `tail_gain`, żeby po wycofaniu dołków nie zrobił się głośniejszy. Pozostałe style i siła
/// od [`directional::NO_FADE_CUE_TAIL`] (wycofywanie wyłączone): `(1, 1)`.
#[inline]
fn fade_after_onset(style: MonoCueStyle, fresh: f32, settings: &OutputSettings) -> (f32, f32) {
    if !style.fades_after_onset() || settings.cue_tail >= directional::NO_FADE_CUE_TAIL {
        return (1.0, 1.0);
    }
    let fresh = fresh.clamp(0.0, 1.0);
    let intensity = settings.cue_tail + (1.0 - settings.cue_tail) * fresh;
    let level = settings.tail_gain + (1.0 - settings.tail_gain) * fresh;
    (intensity, level)
}

/// Efekt pojedynczego stylu kierunkowego nałożony na dźwięk z przeciwnej strony (bez reszty).
/// Style z oddechem biorą wersję z cieńszą barwą, pozostałe wersję bez zmian. `Off` i style
/// spoza [`MonoCueStyle::LAYER_CHOICES`] oddają dźwięk bez wskazówki.
#[inline]
fn directional_effect(
    style: MonoCueStyle,
    split: SplitSample,
    settings: &OutputSettings,
    sample_rate: f32,
    state: &mut EffectState,
) -> f32 {
    let subtractive = settings.subtractive;
    let breath = |state: &mut EffectState| {
        directional::breath_gain(settings.breath_depth, sample_rate, &mut state.breath_phase, subtractive)
    };
    match style {
        MonoCueStyle::DirectionalBreath => split.part_timbre * breath(state),
        MonoCueStyle::BreathBuzz => {
            // Oddech oznacza długie dźwięki, lekkie brzęczenie (45 Hz, kilka cykli na krok) krótkie.
            let breathed = split.part_timbre * breath(state);
            let makeup = if subtractive { 1.0 } else { settings.buzz_makeup };
            am_modulate(breathed, RING_MOD_BUZZ_HZ, settings.buzz_depth, sample_rate, &mut state.phase) * makeup
        }
        MonoCueStyle::DirectionalComb if subtractive => comb_dips(split.part, settings.comb_gamma, sample_rate, state),
        MonoCueStyle::DirectionalBuzz
        | MonoCueStyle::DirectionalSqueaky
        | MonoCueStyle::DirectionalBright
        | MonoCueStyle::DirectionalComb => {
            apply_cue(style, split.part, sample_rate, state) * directional_makeup(style, subtractive)
        }
        _ => split.part,
    }
}

/// Cieńsza barwa Oddechu liczona filtrami w czasie, dla przodu i tyłu w stylu „Wg kierunku”
/// (dla strony przeciwnej liczy ją rozdzielanie w paskach FFT). Te same filtry pierwszego rzędu:
/// górnoprzepustowy [`TREMOLO_HPF_HZ`] i dolnoprzepustowy [`directional::TIMBRE_LPF_HZ`],
/// z wyrównaniem `makeups` (patrz `directional::timbre_makeups`) i siłą z suwaka barwy.
#[inline]
fn zone_timbre(
    sample: f32,
    settings: &OutputSettings,
    makeups: (f32, f32),
    sample_rate: f32,
    state: &mut EffectState,
) -> f32 {
    let sample_rate = sample_rate.max(1.0);
    let bass = one_pole_lowpass_step(sample, sample_rate, TREMOLO_HPF_HZ, &mut state.lp2);
    let band = one_pole_lowpass_step(sample - bass, sample_rate, directional::TIMBRE_LPF_HZ, &mut state.lp);
    let makeup = if settings.subtractive { makeups.1 } else { makeups.0 };
    sample + settings.breath_timbre * (band * makeup - sample)
}

/// Styl „Wg kierunku”: nakłada style przodu i tyłu na ich części pary `(left, right)` i zwraca
/// parę z oznaczonymi strefami. Reszta pary (kanały boczne, LFE) przechodzi bez zmian.
///
/// Lewy i prawy kanał strefy dostają ten sam efekt w tej samej fazie, więc różnica poziomów
/// stron w każdym paśmie zostaje i rozdzielanie kierunkowe działa na wyniku tak jak na oryginale.
/// Strefy nie mają wycofywania wskazówki po początku dźwięku (to dotyczy tylko strony przeciwnej).
#[inline]
fn apply_zones(
    left: f32,
    right: f32,
    zone: ZoneFrame,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) -> (f32, f32) {
    let makeups = cue_state.zone_timbre_makeups;
    let (mut left, mut right) = (left, right);
    for (slot, direction) in [CueDirection::Front, CueDirection::Back].into_iter().enumerate() {
        let style = settings.direction_styles[direction.index()];
        let gain = settings.direction_gains[direction.index()];
        let intensity = settings.direction_intensity[direction.index()];
        for (side, out) in [&mut left, &mut right].into_iter().enumerate() {
            let dry = zone[2 * slot + side];
            let state = &mut cue_state.zone_effects[slot][side];
            let part_timbre = if style.uses_breath() { zone_timbre(dry, settings, makeups, sample_rate, state) } else { dry };
            let split = SplitSample { part: dry, part_timbre, rest: 0.0, fresh: 0.0 };
            let effect = directional_effect(style, split, settings, sample_rate, state);
            *out += gain * (dry + intensity * (effect - dry)) - dry;
        }
    }
    (left, right)
}

/// Częstotliwość podziału pasm dla korekcji tonu ([`apply_tone`]) — poniżej
/// jest składowa "bas", powyżej "wysokie". 500 Hz leży w środku pasma, w którym
/// słuchacz najłatwiej słyszy barwę, a nie tylko głośność.
pub(crate) const TONE_CROSSOVER_HZ: f32 = 500.0;

/// Stan korekcji tonu (filtr dolnoprzepustowy trzymany między callbackami, tak
/// jak [`CueState`]), żeby podział na bas/wysokie nie trzaskał na granicach
/// buforów.
#[derive(Debug, Default, Clone, Copy)]
struct ToneState {
    lp: f32,
}

/// Korekcja barwy: rozdziela próbkę na składową niską (lowpass na
/// [`TONE_CROSSOVER_HZ`]) i wysoką (reszta, `sample - low`), każdą mnoży
/// osobnym wzmocnieniem i sumuje z powrotem. Przy obu wzmocnieniach `1.0`
/// jest to tożsamość — dlatego korekcja w pozycji 0 dB nic nie zmienia.
#[inline]
fn apply_tone(sample: f32, sample_rate: f32, low_gain: f32, high_gain: f32, state: &mut ToneState) -> f32 {
    let low = one_pole_lowpass_step(sample, sample_rate.max(1.0), TONE_CROSSOVER_HZ, &mut state.lp);
    let high = sample - low;
    low * low_gain + high * high_gain
}

/// Wszystko, co wpływa na brzmienie wyjścia w jednym callbacku, czytane raz
/// z `SharedAudioAnalysis` (patrz [`read_output_settings`]).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OutputSettings {
    ear: MonoEar,
    /// Wzmocnienie liniowe (`mono_boost_db`).
    gain: f32,
    cue_style: MonoCueStyle,
    /// Wzmocnienie składowej niskiej i wysokiej korekcji tonu — `1.0` obie = bez zmian.
    tone_low_gain: f32,
    tone_high_gain: f32,
    /// Style kierunkowe z oddechem: głębokość oddechu i siła cieńszej barwy (obie 0–1).
    breath_depth: f32,
    breath_timbre: f32,
    /// „Oddech + brzęczenie”: siła lekkiego brzęczenia (0–1) i wyrównanie jego głośności.
    buzz_depth: f32,
    buzz_makeup: f32,
    /// Styl „Wg warstw”: styl dla strzałów, kroków i reszty (kolejność `layers::Layer::ALL`),
    /// każdy z [`MonoCueStyle::LAYER_CHOICES`].
    layer_styles: [MonoCueStyle; 3],
    /// Styl „Wg warstw”: głośność każdej warstwy (mnożnik liniowy) i intensywność wskazówki
    /// (0–1: udział efektu względem czystego dźwięku), w tej samej kolejności.
    layer_gains: [f32; 3],
    layer_intensity: [f32; 3],
    /// Styl „Wg kierunku”: styl, głośność (mnożnik liniowy) i intensywność wskazówki (0–1) dla
    /// strony przeciwnej, przodu i tyłu (kolejność [`CueDirection::ALL`]).
    direction_styles: [MonoCueStyle; 3],
    direction_gains: [f32; 3],
    direction_intensity: [f32; 3],
    /// Strona przeciwna w stylach kierunkowych (patrz [`OppositeSettings`]): wskazówki tylko
    /// „w dół”, współczynnik γ grzebienia, poziom (mnożnik), ustępowanie (dB), cień głowy (0–0,7),
    /// siła wskazówki po początku dźwięku (0–1) i poziom ogona (mnożnik).
    subtractive: bool,
    comb_gamma: f32,
    opposite_gain: f32,
    duck_db: f32,
    shadow: f32,
    cue_tail: f32,
    tail_gain: f32,
}

/// Ustawienia strony przeciwnej w stylach kierunkowych, tak jak w configu (dB i ułamki).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OppositeSettings {
    /// Wskazówki tylko „w dół”: strona przeciwna nigdzie nie jest głośniej niż bez wskazówki.
    subtractive: bool,
    /// Głębokość dołków grzebienia w trybie „w dół” (dB, ujemna).
    comb_notch_db: f32,
    /// Poziom całej oznaczonej strony przeciwnej (dB).
    level_db: f32,
    /// Ile dB strona przeciwna ustępuje w paśmie, w którym gra strona własna.
    duck_db: f32,
    /// Udział cienia głowy (0 = wyłączony).
    shadow: f32,
    /// Siła wskazówki po początku dźwięku (1 = bez wycofywania).
    tail: f32,
    /// Poziom ogona po wycofaniu wskazówki (dB, ≤ 0).
    tail_level_db: f32,
}

impl Default for OppositeSettings {
    fn default() -> Self {
        Self {
            subtractive: true,
            comb_notch_db: directional::DEFAULT_COMB_NOTCH_DB,
            level_db: directional::DEFAULT_OPPOSITE_LEVEL_DB,
            duck_db: directional::DEFAULT_DUCK_DB,
            shadow: 0.0,
            tail: directional::DEFAULT_CUE_TAIL,
            tail_level_db: directional::DEFAULT_TAIL_LEVEL_DB,
        }
    }
}

impl OppositeSettings {
    /// Wszystko wyłączone: strona przeciwna jak przed tymi poprawkami (wyrównana głośność,
    /// pełny poziom, bez ustępowania, cienia i wycofywania).
    #[cfg(test)]
    fn neutral() -> Self {
        Self {
            subtractive: false,
            comb_notch_db: directional::DEFAULT_COMB_NOTCH_DB,
            level_db: 0.0,
            duck_db: 0.0,
            shadow: 0.0,
            tail: 1.0,
            tail_level_db: 0.0,
        }
    }
}

impl OutputSettings {
    fn new(ear: MonoEar, gain: f32, cue_style: MonoCueStyle, tone_db: f32) -> Self {
        Self {
            ear,
            gain,
            cue_style,
            tone_low_gain: db_to_linear(-tone_db * 0.5),
            tone_high_gain: db_to_linear(tone_db * 0.5),
            breath_depth: directional::DEFAULT_DEPTH,
            breath_timbre: directional::DEFAULT_TIMBRE,
            buzz_depth: directional::DEFAULT_BUZZ_DEPTH,
            buzz_makeup: am_makeup(directional::DEFAULT_BUZZ_DEPTH),
            layer_styles: MonoLayerStyles::default().in_layer_order(),
            layer_gains: [1.0; 3],
            layer_intensity: [1.0; 3],
            direction_styles: MonoDirectionStyles::default().in_order(),
            direction_gains: [1.0; 3],
            direction_intensity: [1.0; 3],
            subtractive: true,
            comb_gamma: 0.0,
            opposite_gain: 1.0,
            duck_db: 0.0,
            shadow: 0.0,
            cue_tail: 1.0,
            tail_gain: 1.0,
        }
        .with_opposite(OppositeSettings::default())
    }

    /// Ustawia stronę przeciwną dla stylów kierunkowych, przyciętą do dozwolonych zakresów.
    fn with_opposite(mut self, opposite: OppositeSettings) -> Self {
        let notch_db = opposite.comb_notch_db.clamp(directional::MIN_COMB_NOTCH_DB, directional::MAX_COMB_NOTCH_DB);
        self.subtractive = opposite.subtractive;
        self.comb_gamma = 1.0 - db_to_linear(notch_db);
        self.opposite_gain = db_to_linear(
            opposite.level_db.clamp(directional::MIN_OPPOSITE_LEVEL_DB, directional::MAX_OPPOSITE_LEVEL_DB),
        );
        self.duck_db = opposite.duck_db.clamp(0.0, directional::MAX_DUCK_DB);
        self.shadow = opposite.shadow.clamp(0.0, directional::MAX_SHADOW);
        self.cue_tail = opposite.tail.clamp(0.0, 1.0);
        self.tail_gain = db_to_linear(opposite.tail_level_db.clamp(directional::MIN_TAIL_LEVEL_DB, 0.0));
        self
    }

    /// Parametry rozdzielania kierunkowego dla tej próbki.
    fn split_params(&self) -> SplitParams {
        SplitParams {
            timbre: self.breath_timbre,
            subtractive: self.subtractive,
            duck_db: self.duck_db,
            shadow: self.shadow,
        }
    }

    /// Ustawia parametry stylów z oddechem, przycięte do dozwolonych zakresów.
    fn with_breath(mut self, depth: f32, timbre: f32, buzz_depth: f32) -> Self {
        self.breath_depth = depth.clamp(0.0, directional::MAX_DEPTH);
        self.breath_timbre = timbre.clamp(0.0, 1.0);
        self.buzz_depth = buzz_depth.clamp(0.0, 1.0);
        self.buzz_makeup = am_makeup(self.buzz_depth);
        self
    }

    /// Ustawia style, głośności i intensywności warstw dla stylu „Wg warstw”.
    fn with_layers(mut self, layers: &MonoLayerStyles) -> Self {
        self.layer_styles = layers.in_layer_order();
        self.layer_gains = layers.volume_db_in_layer_order().map(db_to_linear);
        self.layer_intensity = layers.intensity_in_layer_order();
        self
    }

    /// Ustawia style, głośności i intensywności kierunków dla stylu „Wg kierunku”.
    fn with_directions(mut self, directions: &MonoDirectionStyles) -> Self {
        self.direction_styles = directions.in_order();
        self.direction_gains = directions.volume_db_in_order().map(db_to_linear);
        self.direction_intensity = directions.intensity_in_order();
        self
    }

    /// Ustawienia bez korekcji tonu — do testów i ścieżek, którym tonu nie trzeba.
    #[cfg(test)]
    fn plain(ear: MonoEar, gain: f32, cue_style: MonoCueStyle) -> Self {
        Self::new(ear, gain, cue_style, 0.0)
    }
}

// ── Uchwyt do strumienia wyjściowego ─────────────────────────────────────────

/// Trzymaj ten obiekt przy życiu tak długo, jak ma grać routing.
/// Drop zatrzymuje strumień wyjściowy cpal.
pub struct MonoRouter {
    _output_stream: cpal::Stream,
}

// ── Budowanie strumienia wyjściowego ─────────────────────────────────────────

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    analysis: SharedAudioAnalysis,
) -> Result<cpal::Stream> {
    let mut stream_config: StreamConfig = config.clone().into();
    // Mały bufor = mniejsze opóźnienie między dźwiękiem wejściowym a
    // usłyszanym cue w słuchawkach — patrz doc-komentarz
    // `devices::low_latency_buffer_size`, użyty tu dla symetrii z
    // przechwytywaniem wejściowym.
    stream_config.buffer_size = crate::sound::devices::low_latency_buffer_size(config.buffer_size());
    let out_channels = stream_config.channels as usize;
    let sample_rate = stream_config.sample_rate.0 as f32;
    let err_fn = |e| crate::dlog!("[mono_router] output error: {e}");

    // Wyjście obsługujemy tylko jako F32 — konwertujemy jeśli potrzeba.
    // Sterowanie lewym/prawym uchem przez `analysis.mono_ear`. `cue_state`
    // (patrz [`CueState`]) to stan efektu nakładanego na stronę przeciwną
    // (patrz `fill_output_f32`/`apply_cue`) — musi przetrwać między kolejnymi
    // wywołaniami callbacku, żeby fala/filtr nie resetowały się i nie
    // trzaskały na granicy buforów.
    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let mut cue_state = CueState::new(sample_rate);
            device.build_output_stream(
                &stream_config,
                move |out: &mut [f32], _| {
                    let frames = out.len() / out_channels.max(1);
                    let (pan, tags, zones) = analysis.read_mono_frames(frames, sample_rate);
                    let settings = read_output_settings(&analysis);
                    fill_output_zoned(out, &pan, &tags, &zones, out_channels, &settings, sample_rate, &mut cue_state);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut cue_state = CueState::new(sample_rate);
            device.build_output_stream(
                &stream_config,
                move |out: &mut [i16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let (pan, tags, zones) = analysis.read_mono_frames(frames, sample_rate);
                    let settings = read_output_settings(&analysis);
                    // Konwersja f32 → i16
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_zoned(&mut tmp, &pan, &tags, &zones, out_channels, &settings, sample_rate, &mut cue_state);
                    for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                        *dst = (src * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut cue_state = CueState::new(sample_rate);
            device.build_output_stream(
                &stream_config,
                move |out: &mut [u16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let (pan, tags, zones) = analysis.read_mono_frames(frames, sample_rate);
                    let settings = read_output_settings(&analysis);
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_zoned(&mut tmp, &pan, &tags, &zones, out_channels, &settings, sample_rate, &mut cue_state);
                    for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                        // f32 (-1..1) → u16 (0..65535), środek = 32768
                        *dst = ((src * 0.5 + 0.5) * u16::MAX as f32)
                            .clamp(0.0, u16::MAX as f32) as u16;
                    }
                },
                err_fn,
                None,
            )?
        }
        fmt => bail!("[mono_router] unsupported output format: {fmt}"),
    };

    Ok(stream)
}

/// Konwertuje decybele na liniowy mnożnik amplitudy (0 dB = 1.0×, +6 dB ≈
/// 2×, -6 dB ≈ 0.5×) — standardowy wzór `10^(dB/20)`.
fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Próg (amplituda), poniżej którego [`soft_limit`] nie rusza sygnału wcale
/// — normalne, nieprzesterowane poziomy (w tym wszystkie dotychczasowe
/// testy w tym pliku) muszą wyjść niezmienione.
const SOFT_LIMIT_KNEE: f32 = 0.8;

/// Miękki limiter amplitudy: identyczność poniżej [`SOFT_LIMIT_KNEE`],
/// powyżej płynne (tanh) dogięcie do `±1.0` zamiast twardego `clamp`.
///
/// `frame[0] = (left + opposite) * gain` w [`fill_output_f32`] regularnie
/// przekracza `±1.0` — zarówno gdy oba kanały grają jednocześnie głośno, jak
/// i przy `mono_boost_db > 0`. Twarde obcięcie w tym miejscu spłaszcza
/// szczyt fali (ostra nieciągłość pochodnej), co słychać jako trzaski/ścinki
/// ("zminimalizuj ścinki dźwięku przy mono"); dodatkowo te produkty
/// zniekształceń mieszają się z oscylatorem `Buzz`/`Squeaky` z [`apply_cue`]
/// i wzmacniają jego nieprzyjemną, piskliwą barwę. Miękkie dogięcie
/// eliminuje tę nieciągłość, więc oba objawy znikają razem.
///
/// Próg i kształt zostały przy zmianie skali bufora mono na stałą skalę
/// stereo (`devices::MONO_PAN_SCALE`): przy VB-Cable 7.1 bufor jest 12 dB
/// głośniejszy, ale `mono_boost_db` ze starego configu jest obniżane o tyle
/// samo (`RadarConfig::migrate_mono_level`), więc do limitera trafia ten sam
/// poziom co wcześniej.
#[inline]
fn soft_limit(x: f32) -> f32 {
    let mag = x.abs();
    if mag <= SOFT_LIMIT_KNEE {
        return x;
    }
    let span = 1.0 - SOFT_LIMIT_KNEE;
    let over = (mag - SOFT_LIMIT_KNEE) / span;
    let compressed = SOFT_LIMIT_KNEE + span * over.tanh();
    x.signum() * compressed
}

/// Wyprzedzenie limitera z pierwszeństwem w próbkach (co najmniej 1).
fn priority_lookahead_samples(sample_rate: f32) -> usize {
    ((PRIORITY_LOOKAHEAD_MS * 0.001 * sample_rate.max(1.0)).round() as usize).max(1)
}

/// Stałe opóźnienie stylów kierunkowych (ms przy 48 kHz): rozdzielanie i wyprzedzenie limitera.
pub(crate) fn directional_latency_ms() -> f32 {
    (directional::LATENCY + priority_lookahead_samples(48_000.0)) as f32 / 48.0
}

/// Limiter z pierwszeństwem dla strony własnej (style kierunkowe).
///
/// Wspólny [`soft_limit`] na sumie ściszał całość, więc głośny dźwięk z przeciwnej strony
/// chwilowo ściszał też stronę własną. Tu, zanim suma dojdzie do progu, ścisza się najpierw
/// tylko strona przeciwna: `g = (próg − obwiednia własnej) / obwiednia przeciwnej`, od 1 w dół
/// do [`PRIORITY_MIN_GAIN_DB`]. Obwiednie liczone są z [`PRIORITY_LOOKAHEAD_MS`] wyprzedzenia
/// (oba sygnały idą przez krótką linię opóźnienia), więc ściszenie zdąży przed szczytem.
/// `soft_limit` na końcu zostaje jako ostatnie zabezpieczenie (np. gdy sama strona własna
/// przekracza próg).
#[derive(Debug, Clone)]
struct PriorityLimiter {
    own_delay: Vec<f32>,
    opp_delay: Vec<f32>,
    pos: usize,
    env_own: f32,
    env_opp: f32,
    gain: f32,
    min_gain: f32,
    env_release: f32,
    attack: f32,
    release: f32,
}

impl PriorityLimiter {
    fn new(sample_rate: f32) -> Self {
        let lookahead = priority_lookahead_samples(sample_rate);
        Self {
            own_delay: vec![0.0; lookahead],
            opp_delay: vec![0.0; lookahead],
            pos: 0,
            env_own: 0.0,
            env_opp: 0.0,
            gain: 1.0,
            min_gain: db_to_linear(PRIORITY_MIN_GAIN_DB),
            env_release: smoothing_per_sample(PRIORITY_ENV_RELEASE_MS, sample_rate),
            attack: smoothing_per_sample(PRIORITY_ATTACK_MS, sample_rate),
            release: smoothing_per_sample(PRIORITY_RELEASE_MS, sample_rate),
        }
    }

    fn reset(&mut self) {
        self.own_delay.fill(0.0);
        self.opp_delay.fill(0.0);
        self.pos = 0;
        self.env_own = 0.0;
        self.env_opp = 0.0;
        self.gain = 1.0;
    }

    /// Obwiednia szczytowa: od razu w górę, w dół z [`PRIORITY_ENV_RELEASE_MS`].
    #[inline]
    fn follow(env: &mut f32, x: f32, release: f32) {
        let mag = x.abs();
        if mag > *env {
            *env = mag;
        } else {
            *env += (mag - *env) * release;
        }
    }

    /// Strona własna i przeciwna (po tonie i wzmocnieniu) na jedną próbkę wyjścia, opóźnioną
    /// o wyprzedzenie.
    #[inline]
    fn process(&mut self, own: f32, opp: f32) -> f32 {
        Self::follow(&mut self.env_own, own, self.env_release);
        Self::follow(&mut self.env_opp, opp, self.env_release);
        let target = if self.env_opp > 1e-9 {
            ((SOFT_LIMIT_KNEE - self.env_own) / self.env_opp).clamp(self.min_gain, 1.0)
        } else {
            1.0
        };
        let speed = if target < self.gain { self.attack } else { self.release };
        self.gain += (target - self.gain) * speed;

        let own_late = std::mem::replace(&mut self.own_delay[self.pos], own);
        let opp_late = std::mem::replace(&mut self.opp_delay[self.pos], opp);
        self.pos = (self.pos + 1) % self.own_delay.len();
        soft_limit(own_late + self.gain * opp_late)
    }
}

/// Czyta `mono_ear`, `mono_boost_db`, `mono_cue_style` i `mono_tone_db` jednym
/// lockiem i od razu zamienia dB na mnożniki liniowe — wołane raz na każdy
/// callback wyjściowy. Korekcja tonu rozkłada dB symetrycznie: `+x` dB na
/// wysokich to `+x/2` na wysokich i `-x/2` na niskich (pochylenie, a nie
/// sama głośność).
fn read_output_settings(analysis: &SharedAudioAnalysis) -> OutputSettings {
    /// Same liczby z configu, kopiowane pod zamkiem (bez klonowania całego `RadarConfig`,
    /// które alokowałoby w callbacku audio).
    struct Snapshot {
        ear: MonoEar,
        boost_db: f32,
        cue_style: MonoCueStyle,
        tone_db: f32,
        class_gain: bool,
        class_gain_db: f32,
        scout: bool,
        scout_amount: f32,
        breath_depth: f32,
        breath_timbre: f32,
        buzz_depth: f32,
        layers: MonoLayerStyles,
        directions: MonoDirectionStyles,
        opposite: OppositeSettings,
    }
    let s = analysis
        .lock()
        .map(|g| {
            let c = &g.radar_config;
            Snapshot {
                ear: g.mono_ear,
                boost_db: c.mono_boost_db,
                cue_style: c.mono_cue_style,
                tone_db: c.mono_tone_db,
                class_gain: c.mono_class_gain,
                class_gain_db: c.mono_class_gain_db,
                scout: c.mono_scout,
                scout_amount: c.mono_scout_amount,
                breath_depth: c.mono_breath_depth,
                breath_timbre: c.mono_breath_timbre,
                buzz_depth: c.mono_buzz_depth,
                layers: c.mono_layer_styles,
                directions: c.mono_direction_styles,
                opposite: OppositeSettings {
                    subtractive: c.mono_cue_subtractive,
                    comb_notch_db: c.mono_comb_notch_db,
                    level_db: c.mono_opposite_level_db,
                    duck_db: c.mono_opposite_duck_db,
                    shadow: c.mono_opposite_shadow,
                    tail: c.mono_cue_tail,
                    tail_level_db: c.mono_cue_tail_level_db,
                },
            }
        })
        .unwrap_or(Snapshot {
            ear: MonoEar::Off,
            boost_db: 0.0,
            cue_style: MonoCueStyle::Off,
            tone_db: 0.0,
            class_gain: false,
            class_gain_db: 0.0,
            scout: false,
            scout_amount: 0.0,
            breath_depth: directional::DEFAULT_DEPTH,
            breath_timbre: directional::DEFAULT_TIMBRE,
            buzz_depth: directional::DEFAULT_BUZZ_DEPTH,
            layers: MonoLayerStyles::default(),
            directions: MonoDirectionStyles::default(),
            opposite: OppositeSettings::default(),
        });
    // Przełączniki i siły idą do bufora; zastosują się od następnego callbacku.
    analysis.set_mono_class_gain(s.class_gain, s.class_gain_db);
    analysis.set_mono_scout(s.scout, s.scout_amount);
    analysis.set_mono_layer_styles(s.cue_style == MonoCueStyle::PerLayer);
    OutputSettings::new(s.ear, db_to_linear(s.boost_db), s.cue_style, s.tone_db)
        .with_breath(s.breath_depth, s.breath_timbre, s.buzz_depth)
        .with_layers(&s.layers)
        .with_directions(&s.directions)
        .with_opposite(s.opposite)
}

// ── Wypełnianie bufora wyjściowego ───────────────────────────────────────────

/// Jedna próbka do granego ucha w trybie jednousznym jako para `(własna, przeciwna)`: swoja
/// strona i strona `opposite` oznaczona wybraną wskazówką. Suma pary to próbka do ucha.
///
/// Style kanałowe modulują cały KANAŁ przeciwny ([`apply_cue`]) i oddają wszystko jako
/// „własną” (przeciwna = 0), bo kanału nie da się podzielić na strony.
///
/// Style kierunkowe biorą oba kanały naraz: `DirectionalSplit` oddziela dźwięk, który przychodzi
/// z przeciwnej strony (już z ustępowaniem i cieniem głowy), i tylko on dostaje efekt. Środek
/// i swoja strona zostają czyste (opóźnienie rozdzielania, ok. 5,3 ms). Oznaczona strona przeciwna
/// dostaje jeszcze poziom `opposite_gain`, a style bez wolnej modulacji wycofują wskazówkę po
/// początku dźwięku ([`fade_after_onset`]).
///
/// „Wg warstw” wybiera efekt z `tags`: warstwa rozpoznana po przeciwnej stronie (strzały,
/// kroki) dostaje swój styl, głośność i intensywność, a gdy nic nie rozpoznano, obowiązują
/// ustawienia reszty. Przejścia są płynne ([`LAYER_FADE_MS`]).
#[inline]
fn mix_for_ear(
    left: f32,
    right: f32,
    tags: SideLayers,
    opposite: Opposite,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) -> (f32, f32) {
    let style = settings.cue_style;
    if !style.is_directional() {
        let mixed = match opposite {
            Opposite::Right => left + apply_cue(style, right, sample_rate, &mut cue_state.effect),
            Opposite::Left => right + apply_cue(style, left, sample_rate, &mut cue_state.effect),
        };
        return (mixed, 0.0);
    }
    let split = cue_state.directional.process(left, right, opposite, settings.split_params());
    cue_state.fresh += (split.fresh - cue_state.fresh) * cue_state.fresh_smooth;
    let fresh = cue_state.fresh;
    // Znaczniki zapisujemy zawsze, żeby po przełączeniu na „Wg warstw” były od razu zgrane.
    let tags = cue_state.delayed_tags(tags);
    if style != MonoCueStyle::PerLayer {
        // „Wg kierunku”: przód i tył są już oznaczone w parze (patrz [`apply_zones`]), tu dochodzi
        // styl strony przeciwnej z jej głośnością i intensywnością.
        let (style, gain, strength) = if style == MonoCueStyle::PerDirection {
            let slot = CueDirection::Opposite.index();
            (settings.direction_styles[slot], settings.direction_gains[slot], settings.direction_intensity[slot])
        } else {
            (style, 1.0, 1.0)
        };
        let effect = directional_effect(style, split, settings, sample_rate, &mut cue_state.effect);
        let (intensity, level) = fade_after_onset(style, fresh, settings);
        let marked = gain * level * (split.part + strength * intensity * (effect - split.part));
        return (split.rest, settings.opposite_gain * marked);
    }
    let layer = match opposite {
        Opposite::Left => tags.left,
        Opposite::Right => tags.right,
    }
    .unwrap_or(Layer::Rest);
    let fade = cue_state.layer_fade;
    let mut marked = 0.0;
    for (slot, &slot_layer) in Layer::ALL.iter().enumerate() {
        let target = if slot_layer == layer { 1.0 } else { 0.0 };
        cue_state.layer_weights[slot] += (target - cue_state.layer_weights[slot]) * fade;
        let slot_style = settings.layer_styles[slot];
        let effect = directional_effect(slot_style, split, settings, sample_rate, &mut cue_state.layer_effects[slot]);
        // Intensywność miesza efekt z czystym dźwiękiem z przeciwnej strony, głośność mnoży całość.
        let (fade_intensity, level) = fade_after_onset(slot_style, fresh, settings);
        let intensity = settings.layer_intensity[slot] * fade_intensity;
        let blended = split.part + intensity * (effect - split.part);
        marked += cue_state.layer_weights[slot] * settings.layer_gains[slot] * level * blended;
    }
    (split.rest, settings.opposite_gain * marked)
}

/// Próbka do granego ucha: ton, wzmocnienie i limiter. Style kierunkowe idą przez limiter
/// z pierwszeństwem dla strony własnej, kanałowe przez wspólny [`soft_limit`].
#[inline]
fn ear_sample(
    (own, opp): (f32, f32),
    directional: bool,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) -> f32 {
    let OutputSettings { gain, tone_low_gain, tone_high_gain, .. } = *settings;
    let tone = |x: f32, state: &mut ToneState| apply_tone(x, sample_rate, tone_low_gain, tone_high_gain, state);
    if directional {
        let own = tone(own, &mut cue_state.tone) * gain;
        let opp = tone(opp, &mut cue_state.tone_opp) * gain;
        cue_state.limiter.process(own, opp)
    } else {
        soft_limit(tone(own + opp, &mut cue_state.tone) * gain)
    }
}

/// Zapisuje `pan` (pary "ważonego mono" `(left, right)`, patrz
/// [`crate::sound::devices::SharedAudioAnalysis::read_stereo_pan_tagged`]) do
/// bufora wyjściowego interleaved, ze wzmocnieniem `gain` (mnożnik liniowy,
/// patrz [`db_to_linear`]) — dociśnięte do `-1.0..=1.0`, żeby duży boost dawał
/// głośniej, a nie zawinięcie/trzaski przy konwersji do i16/u16 niżej w
/// łańcuchu. Kanały poza `out_channels` są zerowane.
///
/// Korekcja tonu (`settings.tone_*_gain`, patrz [`apply_tone`]) działa na sumie
/// trafiającej do słuchawki, przed wzmocnieniem i miękkim limiterem.
///
/// `ear` decyduje który kanał stereo dostaje sygnał — w trybach jednousznych
/// (`Left`/`Right`) strona PRZECIWNA do granego ucha nie jest po prostu
/// domiksowywana "na płasko": dostaje efekt wg `cue_style` (patrz
/// [`apply_cue`]/[`MonoCueStyle`], `cue_state` trzymany przez wywołującego
/// między callbackami), żeby dało się usłyszeć w jednej słuchawce, że dźwięk
/// oryginalnie był z drugiej strony — bez tego cała informacja L/R ginęłaby
/// w trybie jednousznym. `MonoCueStyle::Off` (checkbox/wybór w UI,
/// `RadarConfig::mono_cue_style`) wraca do starego zachowania: strona
/// przeciwna gra na pełnej głośności, tak samo jak swoja, bez żadnego
/// oznaczenia.
///
/// Style kierunkowe nie działają na kanale, tylko na dźwięku z przeciwnej
/// strony: patrz [`mix_for_ear`] i `sound::directional`. Obie strony idą
/// osobno przez ton i wzmocnienie, a potem przez [`PriorityLimiter`], który
/// przy głośnej sumie ścisza najpierw stronę przeciwną (+1 ms opóźnienia).
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
fn fill_output_f32(
    out: &mut [f32],
    pan: &[(f32, f32)],
    tags: &[SideLayers],
    out_channels: usize,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) {
    fill_output_zoned(out, pan, tags, &[], out_channels, settings, sample_rate, cue_state);
}

/// Jak [`fill_output_f32`], razem z przodem i tyłem każdej ramki (`zones`, patrz
/// `devices::ZoneFrame`) dla stylu „Wg kierunku”. Brakujące strefy to zera: dźwięk bez przodu
/// i tyłu, czyli tylko styl strony przeciwnej.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fill_output_zoned(
    out: &mut [f32],
    pan: &[(f32, f32)],
    tags: &[SideLayers],
    zones: &[ZoneFrame],
    out_channels: usize,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) {
    let ch = out_channels.max(1);
    let OutputSettings { ear, gain, tone_low_gain, tone_high_gain, .. } = *settings;

    // Styl kierunkowy trzyma okno FFT z poprzednich buforów. Gdy wraca po przerwie (styl
    // kanałowy, inne ucho), zaczyna od czystego okna zamiast wypuścić resztki sprzed przełączenia.
    let directional = settings.cue_style.is_directional() && matches!(ear, MonoEar::Left | MonoEar::Right);
    if directional && !cue_state.directional_active {
        cue_state.reset_directional();
    }
    cue_state.directional_active = directional;
    let per_direction = directional && settings.cue_style == MonoCueStyle::PerDirection;

    for (frame_idx, frame) in out.chunks_exact_mut(ch).enumerate() {
        let (left, right) = pan.get(frame_idx).copied().unwrap_or((0.0, 0.0));
        let layers = tags.get(frame_idx).copied().unwrap_or_default();
        let (left, right) = if per_direction {
            let zone = zones.get(frame_idx).copied().unwrap_or_default();
            apply_zones(left, right, zone, settings, sample_rate, cue_state)
        } else {
            (left, right)
        };

        for s in frame.iter_mut() {
            *s = 0.0;
        }
        match ear {
            MonoEar::Off => { /* cisza */ }
            MonoEar::Left => {
                let mixed = mix_for_ear(left, right, layers, Opposite::Right, settings, sample_rate, cue_state);
                frame[0] = ear_sample(mixed, directional, settings, sample_rate, cue_state);
            }
            MonoEar::Right => {
                let mixed = mix_for_ear(left, right, layers, Opposite::Left, settings, sample_rate, cue_state);
                // Urządzenie z jednym kanałem: gramy w nim.
                let slot = if ch >= 2 { 1 } else { 0 };
                frame[slot] = ear_sample(mixed, directional, settings, sample_rate, cue_state);
            }
            MonoEar::Both => {
                // Oba uszy już słyszą osobno lewą/prawą stronę (stereo
                // wyjście) — bez efektu cue, ale tonem objęty jest cały miks.
                let sample = soft_limit(
                    apply_tone(left + right, sample_rate, tone_low_gain, tone_high_gain, &mut cue_state.tone) * gain,
                );
                frame[0] = sample;
                if ch >= 2 {
                    frame[1] = sample;
                }
            }
        }
    }
}

// ── Publiczne API ─────────────────────────────────────────────────────────────

/// Uruchamia routing mono na wybranym urządzeniu wyjściowym (słuchawki),
/// czytając próbki "ważonego mono" z bufora, do którego pisze strumień
/// wejściowy zasilający radar (patrz komentarz na górze pliku — MUSI on już
/// działać, inaczej `read_stereo_pan_tagged` będzie zwracać ciszę).
///
/// - `output_device_name` — nazwa fizycznego urządzenia wyjściowego (np. słuchawki)
/// - `analysis`           — współdzielony stan; musi mieć ustawione `mono_ear`
///
/// Zwraca `MonoRouter`. Drop zatrzymuje strumień.
pub fn start_mono_output(
    output_device_name: &str,
    analysis: SharedAudioAnalysis,
) -> Result<MonoRouter> {
    let host = cpal::default_host();

    let output_device = host
        .output_devices()?
        .find(|d| d.name().map(|n| n == output_device_name).unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("Output device not found: {}", output_device_name))?;

    let output_config = output_device.default_output_config()?;
    crate::dlog!(
        "[mono_router] output: {:?} | {}ch | {}Hz | {:?}",
        output_device_name,
        output_config.channels(),
        output_config.sample_rate().0,
        output_config.sample_format(),
    );

    let output_stream = build_output_stream(&output_device, &output_config, analysis)?;
    output_stream.play()?;

    crate::dlog!("[mono_router] routing active");

    Ok(MonoRouter { _output_stream: output_stream })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kilka klatek stałej pary `(left, right)`, z domyślnym stylem `Buzz`
    /// (wystarczająco klatek, żeby oscylator zdążył przejść przez pełny
    /// cykl i test złapał zarówno szczyt, jak i dolinę fali).
    fn run_frames(pan: &(f32, f32), n: usize, out_channels: usize, ear: MonoEar) -> Vec<f32> {
        run_frames_with_style(pan, n, out_channels, ear, MonoCueStyle::Buzz)
    }

    fn run_frames_with_style(
        pan: &(f32, f32),
        n: usize,
        out_channels: usize,
        ear: MonoEar,
        cue_style: MonoCueStyle,
    ) -> Vec<f32> {
        let pan_frames = vec![*pan; n];
        let mut out = vec![0.0f32; n * out_channels];
        let mut cue_state = CueState::default();
        let settings = OutputSettings::plain(ear, 1.0, cue_style);
        fill_output_f32(&mut out, &pan_frames, &[], out_channels, &settings, 48_000.0, &mut cue_state);
        out
    }

    #[test]
    fn left_ear_passes_left_side_unmodulated() {
        // Czysty dźwięk z lewej (right=0) — kanał 0 (lewa słuchawka) ma
        // zawsze dokładnie tę samą wartość co wejście, niezależnie od fazy
        // ring modulation (bo modulujemy tylko `right`, a ten jest zerowy).
        let out = run_frames(&(0.7, 0.0), 64, 2, MonoEar::Left);
        for frame in out.chunks_exact(2) {
            assert!((frame[0] - 0.7).abs() < 1e-6);
            assert_eq!(frame[1], 0.0, "prawy kanał wyjścia ma być cichy w trybie Left");
        }
    }

    #[test]
    fn buzz_ring_modulates_the_right_side_instead_of_silencing_it() {
        // Czysty dźwięk z prawej (left=0) grany w trybie Left: nie może być
        // ani ciszą (informacja by zginęła), ani stałą wartością (nie dałoby
        // się jej odróżnić od dźwięku z lewej) — musi się wahać z ring
        // modulation.
        let out = run_frames(&(0.0, 0.7), 64, 2, MonoEar::Left);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        assert!(left_channel.iter().any(|&s| s.abs() > 1e-6), "sygnał z prawej nie może zniknąć");
        let max = left_channel.iter().cloned().fold(f32::MIN, f32::max);
        let min = left_channel.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max - min > 0.1, "ring modulation powinien wyraźnie wahać amplitudę: min={min} max={max}");
    }

    #[test]
    fn squeaky_oscillates_faster_than_buzz() {
        // `Squeaky` to ten sam mechanizm co `Buzz`, tylko szybszy — w tej
        // samej liczbie klatek powinien przejść więcej pełnych cykli, czyli
        // więcej razy przeciąć zero.
        let count_zero_crossings = |samples: &[f32]| {
            samples.windows(2).filter(|w| (w[0] - 0.35).signum() != (w[1] - 0.35).signum()).count()
        };
        let buzz = run_frames_with_style(&(0.0, 0.7), 256, 2, MonoEar::Left, MonoCueStyle::Buzz);
        let squeaky = run_frames_with_style(&(0.0, 0.7), 256, 2, MonoEar::Left, MonoCueStyle::Squeaky);
        let buzz_left: Vec<f32> = buzz.chunks_exact(2).map(|f| f[0]).collect();
        let squeaky_left: Vec<f32> = squeaky.chunks_exact(2).map(|f| f[0]).collect();
        assert!(
            count_zero_crossings(&squeaky_left) > count_zero_crossings(&buzz_left),
            "Squeaky powinien oscylować wyraźnie szybciej niż Buzz"
        );
    }

    #[test]
    fn squeaky_never_dips_as_deep_as_buzz_so_it_sounds_gentler() {
        // Regresja dla "popraw piskliwy na przyjemniejszy": `Buzz` (pełna
        // głębokość modulacji) schodzi na dnie cyklu do dokładnej ciszy;
        // `Squeaky` po dostrojeniu ma zawsze zostawiać słyszalną resztę
        // sygnału (`RING_MOD_SQUEAKY_DEPTH` < 1.0), więc brzmi jak łagodniejsze
        // "trzepotanie", a nie ostre bramkowanie — mimo to nadal wyraźnie
        // różne od Buzz (test wyżej: szybsze) i od ciszy/stałego dźwięku.
        // Pełny cykl `Buzz` (45 Hz @ 48 kHz) to ~1067 ramek — trzeba ich
        // wygenerować z zapasem, żeby złapać dno fali.
        let buzz = run_frames_with_style(&(0.0, 0.7), 2_000, 2, MonoEar::Left, MonoCueStyle::Buzz);
        let squeaky = run_frames_with_style(&(0.0, 0.7), 2_000, 2, MonoEar::Left, MonoCueStyle::Squeaky);
        let buzz_min = buzz.chunks_exact(2).map(|f| f[0]).fold(f32::MAX, f32::min);
        let squeaky_min = squeaky.chunks_exact(2).map(|f| f[0]).fold(f32::MAX, f32::min);
        assert!(buzz_min < 1e-3, "Buzz powinien wciąż schodzić blisko zera: {buzz_min}");
        assert!(
            squeaky_min > 0.3,
            "Squeaky nie powinien już schodzić blisko ciszy jak Buzz: {squeaky_min}"
        );
        let squeaky_left: Vec<f32> = squeaky.chunks_exact(2).map(|f| f[0]).collect();
        assert!(
            squeaky_left.iter().any(|&s| (s - 0.7).abs() > 0.05),
            "Squeaky ma nadal wyraźnie modulować amplitudę, nie tylko lekko drżeć: {squeaky_left:?}"
        );
    }

    #[test]
    fn bright_leaves_a_steady_signal_unchanged_once_settled() {
        // Przy STAŁYM sygnale składowa wysokotonowa (dopełnienie lowpassu)
        // zanika do zera, więc po ustabilizowaniu się filtra `Bright` nie
        // powinien nic dokładać — inaczej cichy, ciągły dźwięk brzmiałby
        // głośniej niż w rzeczywistości.
        let sample_rate = 48_000.0;
        let mut state = EffectState::default();
        let mut out = 0.0;
        for _ in 0..2000 {
            out = apply_cue(MonoCueStyle::Bright, 0.7, sample_rate, &mut state);
        }
        assert!((out - 0.7).abs() < 1e-3, "przy stałym sygnale składowa wysokotonowa powinna zaniknąć: {out}");
    }

    #[test]
    fn bright_amplifies_a_fast_alternating_signal_beyond_its_own_amplitude() {
        // Sygnał zmieniający znak co próbkę to najwyższa możliwa
        // częstotliwość dla danej częstotliwości próbkowania (Nyquist) —
        // idealny przypadek do sprawdzenia podbicia wysokich tonów.
        let sample_rate = 48_000.0;
        let mut state = EffectState::default();
        let mut max_abs = 0.0f32;
        for i in 0..64 {
            let sample = if i % 2 == 0 { 0.5 } else { -0.5 };
            let out = apply_cue(MonoCueStyle::Bright, sample, sample_rate, &mut state);
            max_abs = max_abs.max(out.abs());
        }
        assert!(
            max_abs > 0.5,
            "Bright powinien wzmacniać szybkozmienny sygnał ponad jego własną amplitudę: {max_abs}"
        );
    }

    #[test]
    fn off_style_plays_the_opposite_side_at_full_unmodulated_strength() {
        // `MonoCueStyle::Off`: strona przeciwna wraca do dawnego, prostego
        // zachowania sprzed tej funkcji — pełna głośność, stała wartość, bez
        // wahania — zamiast jakiegokolwiek efektu.
        let out = run_frames_with_style(&(0.0, 0.7), 16, 2, MonoEar::Left, MonoCueStyle::Off);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        for &s in &left_channel {
            assert!((s - 0.7).abs() < 1e-6, "przy Off sygnał z prawej ma grać niezmieniony: {s}");
        }
    }

    /// Wahanie obwiedni (dB) sinusa 1 kHz w blokach po 10 ms, po odcięciu rozbiegu.
    fn block_swing_db(samples: &[f32]) -> f32 {
        let rms: Vec<f32> = samples
            .chunks_exact(480)
            .map(|c| (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let max = rms.iter().cloned().fold(f32::MIN, f32::max);
        let min = rms.iter().cloned().fold(f32::MAX, f32::min);
        20.0 * (max / min.max(1e-9)).log10()
    }

    /// Ucho lewe, same wskazówki „w dół” bez poziomu, ustępowania, cienia i wycofywania:
    /// do sprawdzania kształtu samego efektu.
    fn cues_only(style: MonoCueStyle) -> OutputSettings {
        OutputSettings::plain(MonoEar::Left, 1.0, style).with_opposite(OppositeSettings {
            level_db: 0.0,
            duck_db: 0.0,
            shadow: 0.0,
            tail: 1.0,
            tail_level_db: 0.0,
            ..OppositeSettings::default()
        })
    }

    /// Ucho lewe, strona przeciwna jak przed poprawkami (patrz [`OppositeSettings::neutral`]).
    fn neutral(style: MonoCueStyle) -> OutputSettings {
        OutputSettings::plain(MonoEar::Left, 1.0, style).with_opposite(OppositeSettings::neutral())
    }

    /// Opóźnienie stylów kierunkowych przy 48 kHz: okno FFT i wyprzedzenie limitera.
    fn directional_latency() -> usize {
        directional::LATENCY + priority_lookahead_samples(48_000.0)
    }

    /// Pary `(L, P)` przez `fill_output_f32` w kilku buforach, jak w prawdziwym callbacku (okno
    /// FFT stylów kierunkowych ma przejść przez ich granice), ze znacznikami warstw dla każdej ramki.
    fn run_tagged(settings: OutputSettings, pan_frames: &[(f32, f32)], tags: &[SideLayers]) -> Vec<f32> {
        let mut out = vec![0.0f32; pan_frames.len() * 2];
        let mut cue_state = CueState::default();
        for (i, (out_chunk, pan_chunk)) in out.chunks_mut(441 * 2).zip(pan_frames.chunks(441)).enumerate() {
            let start = (i * 441).min(tags.len());
            let end = (start + pan_chunk.len()).min(tags.len());
            let tag_chunk = &tags[start..end];
            fill_output_f32(out_chunk, pan_chunk, tag_chunk, 2, &settings, 48_000.0, &mut cue_state);
        }
        out
    }

    fn sine(hz: f32, i: usize) -> f32 {
        0.2 * (std::f32::consts::TAU * hz * i as f32 / 48_000.0).sin()
    }

    /// Sekunda sinusa `hz` z poziomami `gain_left`/`gain_right` po obu stronach.
    fn run_directional(style: MonoCueStyle, hz: f32, gain_left: f32, gain_right: f32, ear: MonoEar) -> Vec<f32> {
        run_directional_with(OutputSettings::plain(ear, 1.0, style), hz, gain_left, gain_right)
    }

    /// Jak [`run_directional`], z własnymi ustawieniami.
    fn run_directional_with(settings: OutputSettings, hz: f32, gain_left: f32, gain_right: f32) -> Vec<f32> {
        let pan_frames: Vec<(f32, f32)> =
            (0..48_000).map(|i| (sine(hz, i) * gain_left, sine(hz, i) * gain_right)).collect();
        run_tagged(settings, &pan_frames, &[])
    }

    fn output_channel(out: &[f32], index: usize) -> Vec<f32> {
        out.chunks_exact(2).map(|f| f[index]).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn directional_breath_marks_only_the_sound_from_the_opposite_side() {
        let style = MonoCueStyle::DirectionalBreath;
        let from_right = run_directional(style, 1_000.0, 0.5, 1.0, MonoEar::Left);
        assert!(from_right.chunks_exact(2).all(|f| f[1] == 0.0), "prawy kanał wyjścia ma być cichy");
        let swing = block_swing_db(&output_channel(&from_right, 0)[12_000..]);
        // Domyślnie „w dół” z głębokością 0,44: od 0 do −5 dB.
        assert!(swing > 4.0, "dźwięk z prawej (+6 dB) ma oddychać w lewym uchu: {swing:.2} dB");

        let from_left = run_directional(style, 1_000.0, 1.0, 0.5, MonoEar::Left);
        let swing = block_swing_db(&output_channel(&from_left, 0)[12_000..]);
        assert!(swing < 0.5, "dźwięk ze swojej strony ma zostać czysty: {swing:.2} dB");

        let center = run_directional(style, 1_000.0, 1.0, 1.0, MonoEar::Left);
        let swing = block_swing_db(&output_channel(&center, 0)[12_000..]);
        assert!(swing < 0.5, "środek (np. własna broń) ma zostać czysty: {swing:.2} dB");
    }

    #[test]
    fn directional_breath_follows_the_right_ear_too() {
        let from_left = run_directional(MonoCueStyle::DirectionalBreath, 1_000.0, 1.0, 0.5, MonoEar::Right);
        assert!(from_left.chunks_exact(2).all(|f| f[0] == 0.0), "lewy kanał wyjścia ma być cichy");
        let swing = block_swing_db(&output_channel(&from_left, 1)[12_000..]);
        assert!(swing > 4.0, "przy prawym uchu ma oddychać dźwięk z lewej: {swing:.2} dB");
    }

    #[test]
    fn directional_styles_leave_center_and_own_side_untouched() {
        // Środek i swoja strona trafiają w całości do reszty, więc każdy styl kierunkowy oddaje
        // je co do próbki, tylko opóźnione o rozdzielanie i wyprzedzenie limitera. 1,1 kHz, żeby
        // opóźnienie nie wypadało w całych okresach i błąd opóźnienia był widoczny.
        let latency = directional_latency();
        for &style in MonoCueStyle::DIRECTIONAL {
            for (gain_left, gain_right) in [(1.0, 1.0), (1.0, 0.5)] {
                let out = output_channel(&run_directional(style, 1_100.0, gain_left, gain_right, MonoEar::Left), 0);
                for (t, &y) in out.iter().enumerate().skip(12_000) {
                    let expected = sine(1_100.0, t - latency) * (gain_left + gain_right);
                    assert!((y - expected).abs() < 1e-3, "{style:?} L={gain_left} P={gain_right}, próbka {t}");
                }
            }
        }
    }

    #[test]
    fn directional_styles_mark_the_sound_from_the_opposite_side() {
        // Kształt samych wskazówek „w dół” (bez poziomu i wycofywania).
        use crate::sound::directional::test_signals::modulation_index;
        let from_right = |style: MonoCueStyle, hz: f32| {
            output_channel(&run_directional_with(cues_only(style), hz, 0.5, 1.0), 0)[12_000..].to_vec()
        };
        // Oddech 0,44 „w dół”: obwiednia 1 − 0,44·lfo, czyli m = 0,22 / 0,78 ≈ 0,28.
        let breath = from_right(MonoCueStyle::DirectionalBreath, 1_000.0);
        let m = modulation_index(&breath, 4.0);
        assert!(m > 0.24, "Oddech kierunkowy: oddech 4 Hz, m = {m:.2}");

        let combo = from_right(MonoCueStyle::BreathBuzz, 1_000.0);
        let (m_breath, m_buzz) = (modulation_index(&combo, 4.0), modulation_index(&combo, 45.0));
        assert!(m_breath > 0.2 && m_buzz > 0.2, "Oddech + brzęczenie: oddech {m_breath:.2}, brzęczenie {m_buzz:.2}");

        let buzz = from_right(MonoCueStyle::DirectionalBuzz, 1_000.0);
        assert!(modulation_index(&buzz, 45.0) > 0.8, "Brzęczenie kierunkowe: pełne wahanie 45 Hz");

        let squeaky = from_right(MonoCueStyle::DirectionalSqueaky, 1_000.0);
        assert!(modulation_index(&squeaky, 140.0) > 0.3, "Piskliwy kierunkowy: wahanie 140 Hz");

        // Grzebień „w dół”: pierwszy dołek (1 / 2,4 ms ≈ 417 Hz) ma głębokość z suwaka (−14 dB),
        // a szczyt (833 Hz) zostaje na poziomie oryginału.
        let notch_hz = 0.5 / COMB_DELAY_S;
        let plain_at = |hz: f32| rms(&(12_000..48_000).map(|i| sine(hz, i) * 1.5).collect::<Vec<_>>());
        let notch_db = 20.0 * (rms(&from_right(MonoCueStyle::DirectionalComb, notch_hz)) / plain_at(notch_hz)).log10();
        assert!(
            (notch_db - directional::DEFAULT_COMB_NOTCH_DB).abs() < 1.5,
            "Grzebień kierunkowy: dołek przy {notch_hz:.0} Hz, jest {notch_db:.1} dB"
        );
        let peak_hz = 2.0 * notch_hz;
        let peak_db = 20.0 * (rms(&from_right(MonoCueStyle::DirectionalComb, peak_hz)) / plain_at(peak_hz)).log10();
        assert!(peak_db.abs() < 0.5, "Grzebień kierunkowy: szczyt przy {peak_hz:.0} Hz, jest {peak_db:+.1} dB");

        // Jasny: góra wyraźnie głośniej niż dół.
        let high = rms(&from_right(MonoCueStyle::DirectionalBright, 8_000.0));
        let low = rms(&from_right(MonoCueStyle::DirectionalBright, 100.0));
        let tilt_db = 20.0 * (high / low).log10();
        assert!(tilt_db > 6.0, "Jasny kierunkowy: 8 kHz względem 100 Hz, jest {tilt_db:.1} dB");
    }

    #[test]
    fn subtractive_cues_never_raise_the_opposite_side_above_the_original() {
        // Wskazówki „w dół”: żaden styl, w żadnym paśmie ani fazie modulacji, nie daje dźwięku
        // z przeciwnej strony głośniej niż bez wskazówki (szczyt sinusa najwyżej jak oryginał).
        let original_peak = 0.2 * 1.5;
        for &style in MonoCueStyle::DIRECTIONAL.iter().filter(|&&s| s != MonoCueStyle::PerLayer) {
            for hz in [100.0, 417.0, 833.0, 1_000.0, 3_000.0, 8_000.0] {
                let out = output_channel(&run_directional_with(cues_only(style), hz, 0.5, 1.0), 0);
                let peak = out[12_000..].iter().fold(0.0f32, |m, &y| m.max(y.abs()));
                assert!(
                    peak <= original_peak * 1.03,
                    "{style:?} {hz} Hz: szczyt {:+.2} dB względem oryginału",
                    20.0 * (peak / original_peak).log10()
                );
            }
        }
    }

    /// Szum różowy z prawej (ILD +9 dB) przez styl: zmiana głośności dla ucha (ważenie A, dB)
    /// względem samej sumy kanałów.
    fn loudness_change_db(settings: OutputSettings) -> f32 {
        use crate::sound::directional::test_signals::{a_weighted_db, pan, pink};
        let (l, r) = pan(&pink(96_000, 21), 9.0);
        let pan_frames: Vec<(f32, f32)> = l.iter().copied().zip(r.iter().copied()).collect();
        let plain: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        let reference = a_weighted_db(&plain[24_000 - directional_latency()..96_000 - directional_latency()]);
        let out = output_channel(&run_tagged(settings, &pan_frames, &[]), 0);
        a_weighted_db(&out[24_000..]) - reference
    }

    #[test]
    fn loudness_neutral_mode_keeps_the_loudness_of_the_opposite_side() {
        // Tryb z wyrównaną głośnością (poprawki wyłączone): zmienia się charakter, nie głośność.
        for &style in MonoCueStyle::DIRECTIONAL {
            let diff = loudness_change_db(neutral(style));
            assert!(diff.abs() < 1.5, "{style:?}: głośność strony przeciwnej zmieniona o {diff:+.2} dB");
        }
    }

    #[test]
    fn subtractive_cues_only_make_the_opposite_side_quieter() {
        // „W dół”: strona przeciwna nigdy głośniej niż oryginał, ale też nie znika.
        for &style in MonoCueStyle::DIRECTIONAL {
            let diff = loudness_change_db(cues_only(style));
            assert!((-10.0..=0.3).contains(&diff), "{style:?}: zmiana głośności {diff:+.2} dB");
        }
        // Z domyślnym poziomem (−3 dB) i wycofywaniem: co najmniej o tyle ciszej.
        for &style in MonoCueStyle::DIRECTIONAL {
            let diff = loudness_change_db(OutputSettings::plain(MonoEar::Left, 1.0, style));
            assert!(
                diff <= directional::DEFAULT_OPPOSITE_LEVEL_DB + 0.3,
                "{style:?}: z domyślnymi ustawieniami zmiana głośności {diff:+.2} dB"
            );
        }
    }

    #[test]
    fn opposite_level_scales_only_the_opposite_side() {
        // „Wg warstw” bez wskazówek: zostaje sam poziom strony przeciwnej.
        let layers = MonoLayerStyles {
            shots: MonoCueStyle::Off,
            steps: MonoCueStyle::Off,
            rest: MonoCueStyle::Off,
            ..MonoLayerStyles::default()
        };
        let level = |level_db: f32, gain_left: f32, gain_right: f32| {
            let settings = OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::PerLayer)
                .with_layers(&layers)
                .with_opposite(OppositeSettings { level_db, duck_db: 0.0, ..OppositeSettings::default() });
            rms(&output_channel(&run_directional_with(settings, 1_000.0, gain_left, gain_right), 0)[12_000..])
        };
        let opposite_db = 20.0 * (level(-3.0, 0.5, 1.0) / level(0.0, 0.5, 1.0)).log10();
        assert!((opposite_db + 3.0).abs() < 0.2, "strona przeciwna −3 dB, jest {opposite_db:+.2} dB");
        let own_db = 20.0 * (level(-3.0, 1.0, 0.5) / level(0.0, 1.0, 0.5)).log10();
        assert!(own_db.abs() < 0.05, "swoja strona bez zmian, jest {own_db:+.2} dB");
    }

    #[test]
    fn priority_limiter_keeps_the_own_side_when_the_opposite_side_is_loud() {
        // Cichy dźwięk ze swojej strony (1,1 kHz) na tle głośnego z przeciwnej (200 Hz, szczyt 1,2):
        // wspólny limiter ściszał go razem z całością, limiter z pierwszeństwem ścisza tylko
        // stronę przeciwną.
        use crate::sound::directional::test_signals::tone_level;
        let n = 96_000;
        let own: Vec<f32> = (0..n).map(|i| 0.02 * (std::f32::consts::TAU * 1_100.0 * i as f32 / 48_000.0).sin()).collect();
        let opp: Vec<f32> = (0..n).map(|i| 1.2 * (std::f32::consts::TAU * 200.0 * i as f32 / 48_000.0).sin()).collect();
        let mut limiter = PriorityLimiter::new(48_000.0);
        let priority: Vec<f32> = own.iter().zip(&opp).map(|(&a, &b)| limiter.process(a, b)).collect();
        let shared: Vec<f32> = own.iter().zip(&opp).map(|(&a, &b)| soft_limit(a + b)).collect();
        let reference = tone_level(&own[48_000..], 1_100.0);
        let lookahead = priority_lookahead_samples(48_000.0);
        let kept_db = 20.0 * (tone_level(&priority[48_000 + lookahead..n], 1_100.0) / reference).log10();
        let shared_db = 20.0 * (tone_level(&shared[48_000..], 1_100.0) / reference).log10();
        assert!(kept_db > -1.0, "limiter z pierwszeństwem: swoja strona {kept_db:+.1} dB");
        assert!(shared_db < -3.0, "wspólny limiter: swoja strona {shared_db:+.1} dB");
        assert!(priority.iter().all(|y| y.abs() <= 1.0), "wyjście w zakresie");
    }

    #[test]
    fn priority_limiter_leaves_normal_levels_alone() {
        // Poniżej progu nic nie ściszamy: wyjście to wejście opóźnione o wyprzedzenie.
        let mut limiter = PriorityLimiter::new(48_000.0);
        let lookahead = priority_lookahead_samples(48_000.0);
        let input: Vec<(f32, f32)> = (0..4_800).map(|i| (sine(700.0, i), sine(300.0, i) * 1.5)).collect();
        for (t, &(own, opp)) in input.iter().enumerate() {
            let y = limiter.process(own, opp);
            let expected = input.get(t.wrapping_sub(lookahead)).map_or(0.0, |&(a, b)| a + b);
            assert!((y - expected).abs() < 1e-6, "próbka {t}: {y} zamiast {expected}");
        }
    }

    /// Ćwierć sekundy ciszy, potem dźwięk z prawej (+6 dB) z `source`: wyjście lewego ucha
    /// i numer próbki wyjścia, od której słychać dźwięk.
    fn run_onset(settings: OutputSettings, source: impl Fn(usize) -> f32) -> (Vec<f32>, usize) {
        let start = 12_000;
        let pan_frames: Vec<(f32, f32)> = (0..72_000)
            .map(|i| if i < start { (0.0, 0.0) } else { let s = source(i - start); (s * 0.5, s) })
            .collect();
        (output_channel(&run_tagged(settings, &pan_frames, &[]), 0), start + directional_latency())
    }

    #[test]
    fn comb_cue_fades_after_the_onset_without_getting_louder() {
        use crate::sound::directional::test_signals::noise;
        let settings = OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::DirectionalComb);
        let ms = |x: usize| x * 48;

        // Na dołku grzebienia wskazówka jest wyraźna zaraz po początku i słabnie po ok. 0,5 s.
        let notch_hz = 0.5 / COMB_DELAY_S;
        let (out, at) = run_onset(settings, |i| 0.2 * (std::f32::consts::TAU * notch_hz * i as f32 / 48_000.0).sin());
        let early = rms(&out[at + ms(40)..at + ms(170)]);
        let late = rms(&out[at + ms(700)..at + ms(1_000)]);
        let fade_db = 20.0 * (late / early).log10();
        assert!(fade_db > 5.0, "dołek ma się spłycić po początku dźwięku: {fade_db:+.1} dB");

        // Szum (płaskie widmo, więc RMS dobrze oddaje średnie i wysokie pasma): ogon nie jest
        // głośniejszy niż oznaczony początek.
        let white = noise(60_000, 5);
        let (out, at) = run_onset(settings, |i| 0.3 * white[i]);
        let db = |x: &[f32]| 20.0 * rms(x).log10();
        let early = db(&out[at + ms(40)..at + ms(170)]);
        let late = db(&out[at + ms(700)..at + ms(1_000)]);
        assert!(late <= early + 0.5, "ogon {late:.1} dB, początek {early:.1} dB");

        // 100 %: bez wycofywania.
        let steady = OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::DirectionalComb)
            .with_opposite(OppositeSettings { tail: 1.0, ..OppositeSettings::default() });
        let (out, at) = run_onset(steady, |i| 0.2 * (std::f32::consts::TAU * notch_hz * i as f32 / 48_000.0).sin());
        let early = rms(&out[at + ms(40)..at + ms(170)]);
        let late = rms(&out[at + ms(700)..at + ms(1_000)]);
        let fade_db = 20.0 * (late / early).log10();
        assert!(fade_db.abs() < 0.5, "przy 100 % wskazówka się nie zmienia: {fade_db:+.1} dB");
    }

    #[test]
    fn breath_cue_keeps_breathing_after_the_onset() {
        use crate::sound::directional::test_signals::modulation_index;
        let settings = OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::DirectionalBreath);
        let (out, at) = run_onset(settings, |i| sine(1_000.0, i));
        let late = &out[at + 24_000..at + 48_000];
        let m = modulation_index(late, 4.0);
        assert!(m > 0.24, "oddech nie wycofuje się po początku dźwięku: m = {m:.2}");
    }

    /// Dwie sekundy sinusa 1 kHz z prawej (+6 dB) przez „Wg warstw” z domyślnymi stylami
    /// (strzały: grzebień, kroki: brzęczenie, reszta: oddech). W drugiej sekundzie znacznik kroków
    /// stoi po stronie wybranej przez `tag`. Strona przeciwna z wyrównaną głośnością i bez
    /// wycofywania, żeby sprawdzać samo przełączanie warstw.
    fn run_per_layer(tag: fn(Option<Layer>) -> SideLayers) -> Vec<f32> {
        run_per_layer_with(&MonoLayerStyles::default(), tag)
    }

    /// Jak [`run_per_layer`], z własnymi ustawieniami warstw.
    fn run_per_layer_with(layers: &MonoLayerStyles, tag: fn(Option<Layer>) -> SideLayers) -> Vec<f32> {
        run_per_layer_settings(neutral(MonoCueStyle::PerLayer).with_layers(layers), tag)
    }

    /// Jak [`run_per_layer`], z pełnymi ustawieniami.
    fn run_per_layer_settings(settings: OutputSettings, tag: fn(Option<Layer>) -> SideLayers) -> Vec<f32> {
        let n = 96_000;
        let pan_frames: Vec<(f32, f32)> = (0..n).map(|i| (sine(1_000.0, i) * 0.5, sine(1_000.0, i))).collect();
        let tags: Vec<SideLayers> =
            (0..n).map(|i| tag(if i >= n / 2 { Some(Layer::Steps) } else { None })).collect();
        output_channel(&run_tagged(settings, &pan_frames, &tags), 0)
    }

    #[test]
    fn per_layer_fades_only_layers_with_styles_that_fade() {
        // Domyślnie: reszta (oddech) oddycha cały czas, kroki (brzęczenie) wycofują wskazówkę.
        // Znacznik kroków wchodzi po sekundzie, ale dźwięk trwa od początku, więc brzęczenie
        // trafia już na ogon (40 %).
        use crate::sound::directional::test_signals::modulation_index;
        let out = run_per_layer_settings(
            OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::PerLayer).with_layers(&MonoLayerStyles::default()),
            steps_from_the_right,
        );
        let rest = modulation_index(&out[24_000..46_000], 4.0);
        assert!(rest > 0.24, "reszta: oddech bez wycofywania, m = {rest:.2}");
        // Pełne brzęczenie to obwiednia 0,5 + 0,5·sin. Przy intensywności I: 1 − I/2 + (I/2)·sin,
        // czyli m = (I/2) / (1 − I/2) (0,25 dla ogona 40 %).
        let steps = modulation_index(&out[60_000..94_000], 45.0);
        let half = directional::DEFAULT_CUE_TAIL * 0.5;
        let expected = half / (1.0 - half);
        assert!((steps / expected - 1.0).abs() < 0.2, "kroki: brzęczenie na ogonie, m = {steps:.2}");
        let full = modulation_index(&run_per_layer(steps_from_the_right)[60_000..94_000], 45.0);
        assert!(full > 0.8, "bez wycofywania pełne brzęczenie, m = {full:.2}");
    }

    fn steps_from_the_right(layer: Option<Layer>) -> SideLayers {
        SideLayers { left: None, right: layer }
    }

    #[test]
    fn per_layer_volume_changes_only_its_own_layer() {
        // Wszystkie warstwy bez wskazówki, kroki +6 dB: głośniej robi się tylko odcinek z krokami.
        let layers = MonoLayerStyles {
            shots: MonoCueStyle::Off,
            steps: MonoCueStyle::Off,
            rest: MonoCueStyle::Off,
            steps_volume_db: 6.0,
            ..MonoLayerStyles::default()
        };
        let out = run_per_layer_with(&layers, steps_from_the_right);
        let gain_db = 20.0 * (rms(&out[60_000..94_000]) / rms(&out[12_000..46_000])).log10();
        assert!((gain_db - 6.0).abs() < 0.3, "kroki +6 dB, reszta bez zmian: {gain_db:.2} dB");
    }

    #[test]
    fn per_layer_intensity_blends_the_cue_with_the_plain_sound() {
        use crate::sound::directional::test_signals::modulation_index;
        let buzz_at = |intensity: f32| {
            let layers = MonoLayerStyles { steps_intensity: intensity, ..MonoLayerStyles::default() };
            modulation_index(&run_per_layer_with(&layers, steps_from_the_right)[60_000..94_000], 45.0)
        };
        let (none, half, full) = (buzz_at(0.0), buzz_at(0.5), buzz_at(1.0));
        assert!(none < 0.05, "intensywność 0 %: bez brzęczenia, jest {none:.2}");
        assert!(half > 0.3 && half < 0.6, "intensywność 50 %: połowa brzęczenia, jest {half:.2}");
        assert!(full > 0.8, "intensywność 100 %: pełne brzęczenie, jest {full:.2}");
    }

    #[test]
    fn per_layer_style_follows_the_layer_heard_on_the_opposite_side() {
        use crate::sound::directional::test_signals::modulation_index;
        let out = run_per_layer(|layer| SideLayers { left: None, right: layer });
        // Nic nie rozpoznano: styl reszty, czyli oddech 4 Hz bez brzęczenia.
        let rest = &out[12_000..46_000];
        assert!(modulation_index(rest, 4.0) > 0.24, "reszta: oddech (m ≈ 0,28 przy głębokości 0,44)");
        assert!(modulation_index(rest, 45.0) < 0.05, "reszta: bez brzęczenia");
        // Kroki z prawej (przeciwnej) strony: brzęczenie 45 Hz bez oddechu.
        let steps = &out[60_000..94_000];
        assert!(modulation_index(steps, 45.0) > 0.8, "kroki: brzęczenie");
        assert!(modulation_index(steps, 4.0) < 0.05, "kroki: bez oddechu");
    }

    #[test]
    fn per_layer_ignores_layers_heard_on_the_own_side() {
        use crate::sound::directional::test_signals::modulation_index;
        // Kroki rozpoznane po lewej (swojej) stronie nie zmieniają stylu dźwięku z prawej.
        let out = run_per_layer(|layer| SideLayers { left: layer, right: None });
        let late = &out[60_000..94_000];
        assert!(modulation_index(late, 4.0) > 0.24, "dalej styl reszty");
        assert!(modulation_index(late, 45.0) < 0.05, "kroki ze swojej strony nie dają brzęczenia");
    }

    #[test]
    fn per_layer_switch_fades_instead_of_clicking() {
        // Sinus 1 kHz o amplitudzie ok. 0,5 (z oddechem i wyrównaniem głośności) zmienia się
        // najwyżej o ok. 0,07 na próbkę. Twarde przełączenie stylu dawałoby skok do ok. 0,5.
        let out = run_per_layer(|layer| SideLayers { left: None, right: layer });
        let switch_at = 48_000 + directional::LATENCY;
        let max_step = out[switch_at - 2_000..switch_at + 2_000]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step < 0.1, "skok przy przełączeniu stylu: {max_step}");
    }

    #[test]
    fn both_ears_mode_matches_flat_sum_without_modulation() {
        let out = run_frames(&(0.3, 0.2), 8, 2, MonoEar::Both);
        for frame in out.chunks_exact(2) {
            assert!((frame[0] - 0.5).abs() < 1e-6);
            assert!((frame[1] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn off_ear_produces_silence() {
        let out = run_frames(&(0.9, 0.9), 8, 2, MonoEar::Off);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn soft_limit_leaves_normal_levels_untouched() {
        // Poniżej progu (`SOFT_LIMIT_KNEE`) limiter nie może nic zmieniać —
        // inaczej wszystkie testy powyżej (0.7, 0.5, ...) by się posypały.
        assert_eq!(soft_limit(0.5), 0.5);
        assert_eq!(soft_limit(-0.7), -0.7);
        assert_eq!(soft_limit(0.0), 0.0);
    }

    #[test]
    fn soft_limit_bends_loud_peaks_towards_one_instead_of_flattening_them() {
        // Głośny sygnał z boostem (`mono_boost_db`) albo suma lewej strony i
        // zmodulowanej prawej łatwo przekracza ±1.0 — twardy `clamp` spłaszczał
        // szczyt (ścinki/trzaski), miękki limiter ma się do niego zbliżać bez
        // nagłej nieciągłości i nigdy go nie przekraczać.
        let just_over = soft_limit(1.05);
        assert!(just_over < 1.0, "nawet lekkie przesterowanie ma zostać dogięte poniżej 1.0: {just_over}");
        assert!(just_over > 0.8, "dogięcie ma być łagodne, nie nagłe: {just_over}");

        let way_over = soft_limit(3.0);
        assert!(way_over <= 1.0 && way_over > 0.95, "przy dużym przesterowaniu limiter ma zbliżać się do 1.0: {way_over}");

        assert_eq!(soft_limit(-3.0), -soft_limit(3.0), "limiter ma być symetryczny dla ujemnych amplitud");
    }

    #[test]
    fn flat_tone_leaves_the_signal_unchanged() {
        // 0 dB (oba wzmocnienia 1.0) ma być tożsamością — domyślna konfiguracja
        // nie może zmieniać brzmienia.
        let mut state = ToneState::default();
        for i in 0..256 {
            let x = if i % 2 == 0 { 0.5 } else { -0.3 };
            let y = apply_tone(x, 48_000.0, 1.0, 1.0, &mut state);
            assert!((y - x).abs() < 1e-6, "przy 0 dB sygnał ma przejść bez zmian: {x} -> {y}");
        }
    }

    #[test]
    fn positive_tone_favours_treble_and_negative_favours_bass() {
        // Stały sygnał to składowa niska (wychodzi z wagą `low_gain`), naprzemienny
        // przy Nyquist to wysokie (waga `high_gain`). Po ustabilizowaniu filtra
        // porównujemy oba.
        let settle = |low_gain: f32, high_gain: f32| {
            let mut dc_state = ToneState::default();
            let mut nyq_state = ToneState::default();
            let mut dc_out = 0.0f32;
            let mut nyq_max = 0.0f32;
            for i in 0..2000 {
                dc_out = apply_tone(0.5, 48_000.0, low_gain, high_gain, &mut dc_state);
                let x = if i % 2 == 0 { 0.5 } else { -0.5 };
                let y = apply_tone(x, 48_000.0, low_gain, high_gain, &mut nyq_state);
                if i > 1000 {
                    nyq_max = nyq_max.max(y.abs());
                }
            }
            (dc_out, nyq_max)
        };
        let (bright_dc, bright_nyq) = settle(db_to_linear(-3.0), db_to_linear(3.0));
        let (dark_dc, dark_nyq) = settle(db_to_linear(3.0), db_to_linear(-3.0));
        assert!(bright_nyq > bright_dc, "jasny ton ma wzmacniać wysokie: nyq={bright_nyq} dc={bright_dc}");
        assert!(dark_dc > dark_nyq, "ciemny ton ma wzmacniać bas: dc={dark_dc} nyq={dark_nyq}");
        assert!(bright_nyq > dark_nyq, "wysokie przy jasnym tonie muszą być głośniejsze niż przy ciemnym");
    }

    #[test]
    fn boosted_gain_no_longer_hard_clips_the_combined_signal() {
        // Regresja dla "ścinki dźwięku przy mono": zsumowany sygnał (własna
        // strona + zmodulowana przeciwna) z boostem łatwo szedł powyżej
        // ±1.0. Przedtem trafiał w twardy `clamp` (płaski szczyt na każdej
        // klatce, gdzie suma przekraczała próg); teraz ma się płynnie wyginać.
        let pan_frames = vec![(0.35, 0.35); 32];
        let mut out = vec![0.0f32; pan_frames.len() * 2];
        let mut cue_state = CueState::default();
        // ~2.0x, so (0.35+0.35)*2.0 ≈ 1.4 — over the old ±1.0 clamp
        let boosted = OutputSettings::plain(MonoEar::Left, db_to_linear(6.0), MonoCueStyle::Off);
        fill_output_f32(&mut out, &pan_frames, &[], 2, &boosted, 48_000.0, &mut cue_state);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        assert!(left_channel.iter().all(|&s| s <= 1.0 && s >= -1.0), "limiter nie może przekroczyć zakresu wyjścia");
        assert!(
            left_channel.iter().any(|&s| s < 1.0),
            "przy miękkim dogięciu przesterowany sygnał NIE powinien lądować dokładnie na płaskim 1.0 co klatkę: {left_channel:?}"
        );
    }

    // ── Styl „Wg kierunku” ──────────────────────────────────────────────────

    /// Lewy kanał wyjścia (ucho lewe) dla pary i stref, w kawałkach jak z callbacku.
    fn run_zoned(settings: OutputSettings, pan_frames: &[(f32, f32)], zones: &[ZoneFrame]) -> Vec<f32> {
        let mut out = vec![0.0f32; pan_frames.len() * 2];
        let mut cue_state = CueState::default();
        for (i, (out_chunk, pan_chunk)) in out.chunks_mut(441 * 2).zip(pan_frames.chunks(441)).enumerate() {
            let start = (i * 441).min(zones.len());
            let end = (start + pan_chunk.len()).min(zones.len());
            fill_output_zoned(out_chunk, pan_chunk, &[], &zones[start..end], 2, &settings, 48_000.0, &mut cue_state);
        }
        output_channel(&out, 0)
    }

    fn per_direction(directions: MonoDirectionStyles) -> OutputSettings {
        neutral(MonoCueStyle::PerDirection).with_directions(&directions)
    }

    /// Ton 1 kHz z różnicą stron `ild_db` (dodatnia = z prawej), 2 s.
    fn tone_pair(ild_db: f32) -> Vec<(f32, f32)> {
        use crate::sound::directional::test_signals::pan;
        let src: Vec<f32> = (0..96_000).map(|i| 0.5 * sine(1_000.0, i)).collect();
        let (l, r) = pan(&src, ild_db);
        l.into_iter().zip(r).collect()
    }

    /// Cała para jako przód albo tył.
    fn all_in_zone(pan_frames: &[(f32, f32)], back: bool) -> Vec<ZoneFrame> {
        pan_frames.iter().map(|&(l, r)| if back { [0.0, 0.0, l, r] } else { [l, r, 0.0, 0.0] }).collect()
    }

    #[test]
    fn per_direction_without_zones_is_just_the_opposite_style() {
        // Strumień stereo nie ma przodu ani tyłu: zostaje sam styl strony przeciwnej.
        use crate::sound::directional::test_signals::{noise, pan};
        let (l, r) = pan(&noise(24_000, 31), 9.0);
        let pan_frames: Vec<(f32, f32)> = l.into_iter().zip(r).collect();
        let directions = MonoDirectionStyles { opposite: MonoCueStyle::DirectionalBuzz, ..MonoDirectionStyles::default() };
        let by_direction = run_zoned(per_direction(directions), &pan_frames, &[]);
        let single = run_zoned(neutral(MonoCueStyle::DirectionalBuzz), &pan_frames, &[]);
        assert_eq!(by_direction, single);
    }

    #[test]
    fn back_zone_gets_its_style_and_the_rest_stays_clean() {
        use crate::sound::directional::test_signals::modulation_index;
        // Dźwięk ze środka: strona przeciwna go nie rusza, więc słychać tylko styl strefy.
        let pan_frames = tone_pair(0.0);
        let directions = MonoDirectionStyles {
            front: MonoCueStyle::Off,
            back: MonoCueStyle::DirectionalBreath,
            ..MonoDirectionStyles::default()
        };
        let settings = per_direction(directions);
        let from_back = run_zoned(settings, &pan_frames, &all_in_zone(&pan_frames, true));
        let from_front = run_zoned(settings, &pan_frames, &all_in_zone(&pan_frames, false));
        let from_side = run_zoned(settings, &pan_frames, &[]);
        let breath = |x: &[f32]| modulation_index(&x[24_000..], TREMOLO_HZ);
        assert!(breath(&from_back) > 0.15, "tył ma oddychać: {}", breath(&from_back));
        assert!(breath(&from_front) < 0.02, "przód bez stylu zostaje czysty: {}", breath(&from_front));
        assert!(breath(&from_side) < 0.02, "dźwięk spoza stref zostaje czysty: {}", breath(&from_side));
        // Przód bez stylu to dokładnie suma kanałów, opóźniona jak w każdym stylu kierunkowym.
        let latency = directional_latency();
        for t in 12_000..pan_frames.len() {
            let expected = pan_frames[t - latency].0 + pan_frames[t - latency].1;
            assert!((from_front[t] - expected).abs() < 1e-3, "próbka {t}: {} zamiast {expected}", from_front[t]);
        }
    }

    #[test]
    fn back_sound_on_the_opposite_side_gets_both_styles() {
        use crate::sound::directional::test_signals::modulation_index;
        let pan_frames = tone_pair(9.0);
        let directions = |back| MonoDirectionStyles {
            opposite: MonoCueStyle::DirectionalBuzz,
            front: MonoCueStyle::Off,
            back,
            ..MonoDirectionStyles::default()
        };
        let zones = all_in_zone(&pan_frames, true);
        let both = run_zoned(per_direction(directions(MonoCueStyle::DirectionalBreath)), &pan_frames, &zones);
        let side_only = run_zoned(per_direction(directions(MonoCueStyle::Off)), &pan_frames, &zones);
        let at = |x: &[f32], hz: f32| modulation_index(&x[24_000..], hz);
        assert!(at(&both, RING_MOD_BUZZ_HZ) > 0.5, "brzęczenie strony przeciwnej: {}", at(&both, RING_MOD_BUZZ_HZ));
        assert!(at(&both, TREMOLO_HZ) > 0.15, "oddech tyłu: {}", at(&both, TREMOLO_HZ));
        assert!(at(&side_only, RING_MOD_BUZZ_HZ) > 0.5, "bez stylu tyłu brzęczenie zostaje");
        assert!(at(&side_only, TREMOLO_HZ) < 0.05, "bez stylu tyłu nie ma oddechu: {}", at(&side_only, TREMOLO_HZ));
    }

    #[test]
    fn zone_volume_and_intensity_scale_only_that_zone() {
        let pan_frames = tone_pair(0.0);
        let level = |directions: MonoDirectionStyles, back: bool| {
            rms(&run_zoned(per_direction(directions), &pan_frames, &all_in_zone(&pan_frames, back))[24_000..])
        };
        let base = MonoDirectionStyles { front: MonoCueStyle::Off, back: MonoCueStyle::Off, ..MonoDirectionStyles::default() };
        let quieter = MonoDirectionStyles { back_volume_db: -6.0, ..base };
        let change_db = |a: f32, b: f32| 20.0 * (a / b).log10();
        let back = change_db(level(quieter, true), level(base, true));
        let front = change_db(level(quieter, false), level(base, false));
        assert!((back + 6.0).abs() < 0.1, "tył o 6 dB ciszej: {back:+.2} dB");
        assert!(front.abs() < 0.01, "przód bez zmian: {front:+.2} dB");
        // Intensywność 0 % wyłącza efekt stylu.
        let muted = MonoDirectionStyles { back: MonoCueStyle::DirectionalBuzz, back_intensity: 0.0, ..base };
        let diff = change_db(level(muted, true), level(base, true));
        assert!(diff.abs() < 0.01, "intensywność 0 % to czysty dźwięk: {diff:+.2} dB");
    }
}
