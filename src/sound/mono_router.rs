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

use crate::sound::devices::{MonoCueStyle, MonoEar, SharedAudioAnalysis, SideLayers};
use crate::sound::directional::{self, DirectionalSplit, Opposite, SplitSample};
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
    /// Znaczniki warstw opóźnione o okno FFT, żeby zgadzały się z wyjściem rozdzielania.
    tag_delay: Vec<SideLayers>,
    tag_pos: usize,
    /// Współczynnik wygładzania wag warstw na jedną próbkę.
    layer_fade: f32,
    /// Stan korekcji tonu sumy trafiającej do słuchawki (patrz [`apply_tone`]).
    tone: ToneState,
}

/// Wagi warstw na starcie: cały dźwięk z przeciwnej strony ma styl reszty.
const REST_ONLY: [f32; 3] = [0.0, 0.0, 1.0];

impl CueState {
    fn new(sample_rate: f32) -> Self {
        Self {
            effect: EffectState::default(),
            directional: DirectionalSplit::new(sample_rate),
            directional_active: false,
            layer_effects: [EffectState::default(), EffectState::default(), EffectState::default()],
            layer_weights: REST_ONLY,
            tag_delay: vec![SideLayers::default(); directional::FFT_LEN],
            tag_pos: 0,
            layer_fade: 1.0 - (-1.0 / (LAYER_FADE_MS * 0.001 * sample_rate.max(1.0))).exp(),
            tone: ToneState::default(),
        }
    }

    /// Zeruje stan stylów kierunkowych po przerwie (inny styl, inne ucho).
    fn reset_directional(&mut self) {
        self.directional.reset();
        self.tag_delay.fill(SideLayers::default());
        self.tag_pos = 0;
        self.layer_weights = REST_ONLY;
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

/// Wyrównanie głośności stylu kierunkowego: dźwięk z przeciwnej strony dostaje cały efekt,
/// a ma brzmieć tak samo głośno jak bez niego (zmienia się tylko charakter, nie głośność).
fn directional_makeup(style: MonoCueStyle) -> f32 {
    match style {
        MonoCueStyle::DirectionalBuzz => am_makeup(1.0),
        MonoCueStyle::DirectionalSqueaky => am_makeup(RING_MOD_SQUEAKY_DEPTH),
        MonoCueStyle::DirectionalBright => BRIGHT_DIRECTIONAL_MAKEUP,
        MonoCueStyle::DirectionalComb => COMB_DIRECTIONAL_MAKEUP,
        _ => 1.0,
    }
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
    match style {
        MonoCueStyle::DirectionalBreath => {
            split.part_timbre * directional::breath_gain(settings.breath_depth, sample_rate, &mut state.breath_phase)
        }
        MonoCueStyle::BreathBuzz => {
            // Oddech oznacza długie dźwięki, lekkie brzęczenie (45 Hz, kilka cykli na krok) krótkie.
            let breathed = split.part_timbre
                * directional::breath_gain(settings.breath_depth, sample_rate, &mut state.breath_phase);
            am_modulate(breathed, RING_MOD_BUZZ_HZ, settings.buzz_depth, sample_rate, &mut state.phase)
                * settings.buzz_makeup
        }
        MonoCueStyle::DirectionalBuzz
        | MonoCueStyle::DirectionalSqueaky
        | MonoCueStyle::DirectionalBright
        | MonoCueStyle::DirectionalComb => {
            apply_cue(style, split.part, sample_rate, state) * directional_makeup(style)
        }
        _ => split.part,
    }
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
            layer_styles: crate::sound::devices::MonoLayerStyles::default().in_layer_order(),
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

    /// Ustawia style warstw dla stylu „Wg warstw”.
    fn with_layer_styles(mut self, styles: [MonoCueStyle; 3]) -> Self {
        self.layer_styles = styles.map(MonoCueStyle::as_layer_choice);
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
                    let (pan, tags) = analysis.read_stereo_pan_tagged(frames);
                    let settings = read_output_settings(&analysis);
                    fill_output_f32(out, &pan, &tags, out_channels, &settings, sample_rate, &mut cue_state);
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
                    let (pan, tags) = analysis.read_stereo_pan_tagged(frames);
                    let settings = read_output_settings(&analysis);
                    // Konwersja f32 → i16
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, &tags, out_channels, &settings, sample_rate, &mut cue_state);
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
                    let (pan, tags) = analysis.read_stereo_pan_tagged(frames);
                    let settings = read_output_settings(&analysis);
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, &tags, out_channels, &settings, sample_rate, &mut cue_state);
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
        layer_styles: [MonoCueStyle; 3],
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
                layer_styles: c.mono_layer_styles.in_layer_order(),
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
            layer_styles: crate::sound::devices::MonoLayerStyles::default().in_layer_order(),
        });
    // Przełączniki i siły idą do bufora; zastosują się od następnego callbacku.
    analysis.set_mono_class_gain(s.class_gain, s.class_gain_db);
    analysis.set_mono_scout(s.scout, s.scout_amount);
    analysis.set_mono_layer_styles(s.cue_style == MonoCueStyle::PerLayer);
    OutputSettings::new(s.ear, db_to_linear(s.boost_db), s.cue_style, s.tone_db)
        .with_breath(s.breath_depth, s.breath_timbre, s.buzz_depth)
        .with_layer_styles(s.layer_styles)
}

// ── Wypełnianie bufora wyjściowego ───────────────────────────────────────────

/// Jedna próbka do granego ucha w trybie jednousznym: swoja strona plus strona `opposite`
/// oznaczona wybraną wskazówką.
///
/// Style kanałowe modulują cały KANAŁ przeciwny ([`apply_cue`]). Style kierunkowe biorą oba
/// kanały naraz: `DirectionalSplit` oddziela dźwięk, który przychodzi z przeciwnej strony,
/// i tylko on dostaje efekt, z wyrównaną głośnością. Środek i swoja strona zostają czyste
/// (opóźnienie o okno FFT, ok. 11 ms).
///
/// „Wg warstw” wybiera efekt z `tags`: warstwa rozpoznana po przeciwnej stronie (strzały,
/// kroki) dostaje swój styl, a gdy nic nie rozpoznano, obowiązuje styl reszty. Przejścia są
/// płynne ([`LAYER_FADE_MS`]).
#[inline]
fn mix_for_ear(
    left: f32,
    right: f32,
    tags: SideLayers,
    opposite: Opposite,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) -> f32 {
    let style = settings.cue_style;
    if !style.is_directional() {
        return match opposite {
            Opposite::Right => left + apply_cue(style, right, sample_rate, &mut cue_state.effect),
            Opposite::Left => right + apply_cue(style, left, sample_rate, &mut cue_state.effect),
        };
    }
    let split = cue_state.directional.process(left, right, opposite, settings.breath_timbre);
    // Znaczniki zapisujemy zawsze, żeby po przełączeniu na „Wg warstw” były od razu zgrane.
    let tags = cue_state.delayed_tags(tags);
    if style != MonoCueStyle::PerLayer {
        return split.rest + directional_effect(style, split, settings, sample_rate, &mut cue_state.effect);
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
        let effect =
            directional_effect(settings.layer_styles[slot], split, settings, sample_rate, &mut cue_state.layer_effects[slot]);
        marked += cue_state.layer_weights[slot] * effect;
    }
    split.rest + marked
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
/// strony: patrz [`mix_for_ear`] i `sound::directional`.
#[inline]
fn fill_output_f32(
    out: &mut [f32],
    pan: &[(f32, f32)],
    tags: &[SideLayers],
    out_channels: usize,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
) {
    let ch = out_channels.max(1);
    let OutputSettings { ear, gain, tone_low_gain, tone_high_gain, .. } = *settings;
    let tone = |x: f32, state: &mut ToneState| apply_tone(x, sample_rate, tone_low_gain, tone_high_gain, state);

    // Styl kierunkowy trzyma okno FFT z poprzednich buforów. Gdy wraca po przerwie (styl
    // kanałowy, inne ucho), zaczyna od czystego okna zamiast wypuścić resztki sprzed przełączenia.
    let directional = settings.cue_style.is_directional() && matches!(ear, MonoEar::Left | MonoEar::Right);
    if directional && !cue_state.directional_active {
        cue_state.reset_directional();
    }
    cue_state.directional_active = directional;

    for (frame_idx, frame) in out.chunks_exact_mut(ch).enumerate() {
        let (left, right) = pan.get(frame_idx).copied().unwrap_or((0.0, 0.0));
        let layers = tags.get(frame_idx).copied().unwrap_or_default();

        for s in frame.iter_mut() {
            *s = 0.0;
        }
        match ear {
            MonoEar::Off => { /* cisza */ }
            MonoEar::Left => {
                let mixed = mix_for_ear(left, right, layers, Opposite::Right, settings, sample_rate, cue_state);
                frame[0] = soft_limit(tone(mixed, &mut cue_state.tone) * gain);
            }
            MonoEar::Right if ch >= 2 => {
                let mixed = mix_for_ear(left, right, layers, Opposite::Left, settings, sample_rate, cue_state);
                frame[1] = soft_limit(tone(mixed, &mut cue_state.tone) * gain);
            }
            MonoEar::Right => {
                // mono device fallback
                let mixed = mix_for_ear(left, right, layers, Opposite::Left, settings, sample_rate, cue_state);
                frame[0] = soft_limit(tone(mixed, &mut cue_state.tone) * gain);
            }
            MonoEar::Both => {
                // Oba uszy już słyszą osobno lewą/prawą stronę (stereo
                // wyjście) — bez efektu cue, ale tonem objęty jest cały miks.
                let sample = soft_limit(tone(left + right, &mut cue_state.tone) * gain);
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

    /// Pary `(L, P)` przez `fill_output_f32` w kilku buforach, jak w prawdziwym callbacku:
    /// okno FFT stylów kierunkowych ma przejść przez ich granice.
    fn run_pan(style: MonoCueStyle, pan_frames: &[(f32, f32)], ear: MonoEar) -> Vec<f32> {
        run_tagged(OutputSettings::plain(ear, 1.0, style), pan_frames, &[])
    }

    /// Jak [`run_pan`], z ustawieniami i znacznikami warstw dla każdej ramki.
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
        let pan_frames: Vec<(f32, f32)> =
            (0..48_000).map(|i| (sine(hz, i) * gain_left, sine(hz, i) * gain_right)).collect();
        run_pan(style, &pan_frames, ear)
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
        assert!(swing > 5.0, "dźwięk z prawej (+6 dB) ma oddychać w lewym uchu: {swing:.2} dB");

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
        assert!(swing > 5.0, "przy prawym uchu ma oddychać dźwięk z lewej: {swing:.2} dB");
    }

    #[test]
    fn directional_styles_leave_center_and_own_side_untouched() {
        // Środek i swoja strona trafiają w całości do reszty, więc każdy styl kierunkowy oddaje
        // je co do próbki, tylko opóźnione o okno FFT.
        let latency = directional::FFT_LEN;
        for &style in MonoCueStyle::DIRECTIONAL {
            for (gain_left, gain_right) in [(1.0, 1.0), (1.0, 0.5)] {
                let out = output_channel(&run_directional(style, 1_000.0, gain_left, gain_right, MonoEar::Left), 0);
                for (t, &y) in out.iter().enumerate().skip(12_000) {
                    let expected = sine(1_000.0, t - latency) * (gain_left + gain_right);
                    assert!((y - expected).abs() < 1e-3, "{style:?} L={gain_left} P={gain_right}, próbka {t}");
                }
            }
        }
    }

    #[test]
    fn directional_styles_mark_the_sound_from_the_opposite_side() {
        use crate::sound::directional::test_signals::modulation_index;
        let from_right = |style: MonoCueStyle, hz: f32| {
            output_channel(&run_directional(style, hz, 0.5, 1.0, MonoEar::Left), 0)[12_000..].to_vec()
        };
        let breath = from_right(MonoCueStyle::DirectionalBreath, 1_000.0);
        assert!(modulation_index(&breath, 4.0) > 0.3, "Oddech kierunkowy: oddech 4 Hz");

        let combo = from_right(MonoCueStyle::BreathBuzz, 1_000.0);
        let (m_breath, m_buzz) = (modulation_index(&combo, 4.0), modulation_index(&combo, 45.0));
        assert!(m_breath > 0.25 && m_buzz > 0.2, "Oddech + brzęczenie: oddech {m_breath:.2}, brzęczenie {m_buzz:.2}");

        let buzz = from_right(MonoCueStyle::DirectionalBuzz, 1_000.0);
        assert!(modulation_index(&buzz, 45.0) > 0.8, "Brzęczenie kierunkowe: pełne wahanie 45 Hz");

        let squeaky = from_right(MonoCueStyle::DirectionalSqueaky, 1_000.0);
        assert!(modulation_index(&squeaky, 140.0) > 0.3, "Piskliwy kierunkowy: wahanie 140 Hz");

        // Grzebień: pierwszy dołek (1 / 2,4 ms ≈ 417 Hz) prawie wygasza dźwięk z przeciwnej strony.
        let notch_hz = 0.5 / COMB_DELAY_S;
        let comb = from_right(MonoCueStyle::DirectionalComb, notch_hz);
        let plain = rms(&(12_000..48_000).map(|i| sine(notch_hz, i) * 1.5).collect::<Vec<_>>());
        let notch_db = 20.0 * (rms(&comb) / plain).log10();
        assert!(notch_db < -12.0, "Grzebień kierunkowy: dołek przy {notch_hz:.0} Hz, jest {notch_db:.1} dB");

        // Jasny: góra wyraźnie głośniej niż dół.
        let high = rms(&from_right(MonoCueStyle::DirectionalBright, 8_000.0));
        let low = rms(&from_right(MonoCueStyle::DirectionalBright, 100.0));
        let tilt_db = 20.0 * (high / low).log10();
        assert!(tilt_db > 6.0, "Jasny kierunkowy: 8 kHz względem 100 Hz, jest {tilt_db:.1} dB");
    }

    #[test]
    fn directional_styles_keep_the_loudness_of_the_opposite_side() {
        // Dźwięk z przeciwnej strony (szum różowy, ILD +9 dB) ma brzmieć dla ucha tak samo
        // głośno jak bez wskazówki: zmienia się charakter, nie głośność.
        use crate::sound::directional::test_signals::{a_weighted_db, pan, pink};
        let (l, r) = pan(&pink(96_000, 21), 9.0);
        let pan_frames: Vec<(f32, f32)> = l.iter().copied().zip(r.iter().copied()).collect();
        let plain: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        let reference = a_weighted_db(&plain[24_000 - directional::FFT_LEN..]);
        for &style in MonoCueStyle::DIRECTIONAL {
            let out = output_channel(&run_pan(style, &pan_frames, MonoEar::Left), 0);
            let diff = a_weighted_db(&out[24_000..]) - reference;
            assert!(diff.abs() < 1.5, "{style:?}: głośność strony przeciwnej zmieniona o {diff:+.2} dB");
        }
    }

    /// Dwie sekundy sinusa 1 kHz z prawej (+6 dB) przez „Wg warstw” z domyślnymi stylami
    /// (strzały: grzebień, kroki: brzęczenie, reszta: oddech). W drugiej sekundzie znacznik kroków
    /// stoi po stronie wybranej przez `tag`.
    fn run_per_layer(tag: fn(Option<Layer>) -> SideLayers) -> Vec<f32> {
        let n = 96_000;
        let pan_frames: Vec<(f32, f32)> = (0..n).map(|i| (sine(1_000.0, i) * 0.5, sine(1_000.0, i))).collect();
        let tags: Vec<SideLayers> =
            (0..n).map(|i| tag(if i >= n / 2 { Some(Layer::Steps) } else { None })).collect();
        let settings = OutputSettings::plain(MonoEar::Left, 1.0, MonoCueStyle::PerLayer);
        output_channel(&run_tagged(settings, &pan_frames, &tags), 0)
    }

    #[test]
    fn per_layer_style_follows_the_layer_heard_on_the_opposite_side() {
        use crate::sound::directional::test_signals::modulation_index;
        let out = run_per_layer(|layer| SideLayers { left: None, right: layer });
        // Nic nie rozpoznano: styl reszty, czyli oddech 4 Hz bez brzęczenia.
        let rest = &out[12_000..46_000];
        assert!(modulation_index(rest, 4.0) > 0.3, "reszta: oddech");
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
        assert!(modulation_index(late, 4.0) > 0.3, "dalej styl reszty");
        assert!(modulation_index(late, 45.0) < 0.05, "kroki ze swojej strony nie dają brzęczenia");
    }

    #[test]
    fn per_layer_switch_fades_instead_of_clicking() {
        // Sinus 1 kHz o amplitudzie ok. 0,5 (z oddechem i wyrównaniem głośności) zmienia się
        // najwyżej o ok. 0,07 na próbkę. Twarde przełączenie stylu dawałoby skok do ok. 0,5.
        let out = run_per_layer(|layer| SideLayers { left: None, right: layer });
        let switch_at = 48_000 + directional::FFT_LEN;
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
}
