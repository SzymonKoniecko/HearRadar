//! Mono router — odtwarza w jednej słuchawce (albo obu) zmiksowany sygnał
//! ze wszystkich kanałów, korzystając z TEGO SAMEGO strumienia wejściowego,
//! który już zasila radar. Nie otwiera własnego, niezależnego przechwytywania
//! wejścia — to tylko strumień WYJŚCIOWY, czytający gotowe próbki "ważonego
//! mono" z bufora kołowego w `SharedAudioAnalysis` (patrz
//! `push_stereo_pan`/`read_stereo_pan` w `sound::devices`), do którego pisze
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

use crate::sound::devices::{MonoCueStyle, MonoEar, SharedAudioAnalysis};

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

/// Częstotliwość (Hz) rytmicznego wyciszania/odciszania dla
/// [`MonoCueStyle::Pulse`] — na tyle wolno, żeby słychać było osobne
/// "mrugnięcia", a nie ciągłe brzęczenie.
const PULSE_HZ: f32 = 8.0;

/// Częstotliwość odcięcia (Hz) filtra dolnoprzepustowego dla
/// [`MonoCueStyle::Muffled`] — poniżej typowego pasma mowy, więc strona
/// przeciwna brzmi wyraźnie "przytłumiona"/zza ściany.
const MUFFLED_CUTOFF_HZ: f32 = 800.0;

/// Jak [`MUFFLED_CUTOFF_HZ`], ale niższe — dla [`MonoCueStyle::Dark`], który
/// ma brzmieć wyraźnie bardziej głucho niż zwykły `Muffled`.
const DARK_CUTOFF_HZ: f32 = 300.0;

/// Odcięcie (Hz) filtra dolnoprzepustowego, którego DOPEŁNIENIE
/// (`sample - lowpass`) daje składową wysokotonową dla [`MonoCueStyle::Bright`]
/// — te same pasmo co `Muffled`, tylko użyte w drugą stronę (dodane, nie
/// odjęte od oryginału).
const BRIGHT_HIGHPASS_CUTOFF_HZ: f32 = 800.0;

/// O ile wzmocnić składową wysokotonową w [`MonoCueStyle::Bright`] — na tyle
/// mocno, żeby zmiana barwy była jednoznacznie słyszalna, ale bez ostrego
/// przesterowania przy typowych, niegłośnych sygnałach z radaru.
const BRIGHT_BOOST_GAIN: f32 = 1.5;

/// Stan efektu DSP nakładanego na stronę przeciwną (patrz
/// [`apply_cue`]/[`fill_output_f32`]), trzymany przez wywołującego MIĘDZY
/// kolejnymi wywołaniami callbacku wyjściowego — inaczej faza oscylatora
/// (`phase`) i stan filtra (`lp`) resetowałyby się co bufor, dając trzaski na
/// granicach zamiast płynnego efektu.
#[derive(Debug, Default, Clone, Copy)]
struct CueState {
    /// Faza oscylatora (0.0..=1.0) dla `Buzz`/`Squeaky`/`Pulse`.
    phase: f32,
    /// Stan filtra dolnoprzepustowego jednobiegunowego dla
    /// `Muffled`/`Dark`/`Bright` (ten ostatni używa go pośrednio, przez
    /// [`highpass_component`]).
    lp: f32,
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
/// zmian (stare zachowanie sprzed tej funkcji).
#[inline]
fn apply_cue(style: MonoCueStyle, sample: f32, sample_rate: f32, state: &mut CueState) -> f32 {
    let sample_rate = sample_rate.max(1.0);
    match style {
        MonoCueStyle::Off => sample,
        MonoCueStyle::Buzz => am_modulate(sample, RING_MOD_BUZZ_HZ, 1.0, sample_rate, &mut state.phase),
        MonoCueStyle::Squeaky => {
            am_modulate(sample, RING_MOD_SQUEAKY_HZ, RING_MOD_SQUEAKY_DEPTH, sample_rate, &mut state.phase)
        }
        MonoCueStyle::Pulse => {
            state.phase = (state.phase + PULSE_HZ / sample_rate).fract();
            let gate = if (state.phase * std::f32::consts::TAU).sin() > 0.0 { 1.0 } else { 0.0 };
            sample * gate
        }
        MonoCueStyle::Muffled => one_pole_lowpass_step(sample, sample_rate, MUFFLED_CUTOFF_HZ, &mut state.lp),
        MonoCueStyle::Dark => one_pole_lowpass_step(sample, sample_rate, DARK_CUTOFF_HZ, &mut state.lp),
        MonoCueStyle::Bright => {
            // Dopełnienie filtra dolnoprzepustowego (`sample - lowpass`) to
            // składowa wysokotonowa — dodajemy ją z powrotem do oryginału ze
            // wzmocnieniem, więc wysokie tony stają się głośniejsze względem
            // reszty widma zamiast tłumić resztę (jak w `Muffled`/`Dark`).
            let low = one_pole_lowpass_step(sample, sample_rate, BRIGHT_HIGHPASS_CUTOFF_HZ, &mut state.lp);
            let high = sample - low;
            sample + high * BRIGHT_BOOST_GAIN
        }
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
}

impl OutputSettings {
    /// Ustawienia bez korekcji tonu — do testów i ścieżek, którym tonu nie trzeba.
    #[cfg(test)]
    fn plain(ear: MonoEar, gain: f32, cue_style: MonoCueStyle) -> Self {
        Self { ear, gain, cue_style, tone_low_gain: 1.0, tone_high_gain: 1.0 }
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
            let mut cue_state = CueState::default();
            let mut tone_state = ToneState::default();
            device.build_output_stream(
                &stream_config,
                move |out: &mut [f32], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let settings = read_output_settings(&analysis);
                    fill_output_f32(out, &pan, out_channels, &settings, sample_rate, &mut cue_state, &mut tone_state);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut cue_state = CueState::default();
            let mut tone_state = ToneState::default();
            device.build_output_stream(
                &stream_config,
                move |out: &mut [i16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let settings = read_output_settings(&analysis);
                    // Konwersja f32 → i16
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, out_channels, &settings, sample_rate, &mut cue_state, &mut tone_state);
                    for (dst, &src) in out.iter_mut().zip(tmp.iter()) {
                        *dst = (src * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut cue_state = CueState::default();
            let mut tone_state = ToneState::default();
            device.build_output_stream(
                &stream_config,
                move |out: &mut [u16], _| {
                    let frames = out.len() / out_channels.max(1);
                    let pan = analysis.read_stereo_pan(frames);
                    let settings = read_output_settings(&analysis);
                    let mut tmp = vec![0.0f32; out.len()];
                    fill_output_f32(&mut tmp, &pan, out_channels, &settings, sample_rate, &mut cue_state, &mut tone_state);
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
    let (ear, boost_db, cue_style, tone_db) = analysis
        .lock()
        .map(|g| {
            (
                g.mono_ear,
                g.radar_config.mono_boost_db,
                g.radar_config.mono_cue_style,
                g.radar_config.mono_tone_db,
            )
        })
        .unwrap_or((MonoEar::Off, 0.0, MonoCueStyle::Off, 0.0));
    OutputSettings {
        ear,
        gain: db_to_linear(boost_db),
        cue_style,
        tone_low_gain: db_to_linear(-tone_db * 0.5),
        tone_high_gain: db_to_linear(tone_db * 0.5),
    }
}

// ── Wypełnianie bufora wyjściowego ───────────────────────────────────────────

/// Zapisuje `pan` (pary "ważonego mono" `(left, right)`, patrz
/// [`crate::sound::devices::SharedAudioAnalysis::read_stereo_pan`]) do
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
#[inline]
fn fill_output_f32(
    out: &mut [f32],
    pan: &[(f32, f32)],
    out_channels: usize,
    settings: &OutputSettings,
    sample_rate: f32,
    cue_state: &mut CueState,
    tone_state: &mut ToneState,
) {
    let ch = out_channels.max(1);
    let OutputSettings { ear, gain, cue_style, tone_low_gain, tone_high_gain } = *settings;
    let tone = |x: f32, state: &mut ToneState| apply_tone(x, sample_rate, tone_low_gain, tone_high_gain, state);

    for (frame_idx, frame) in out.chunks_exact_mut(ch).enumerate() {
        let (left, right) = pan.get(frame_idx).copied().unwrap_or((0.0, 0.0));

        for s in frame.iter_mut() {
            *s = 0.0;
        }
        match ear {
            MonoEar::Off => { /* cisza */ }
            MonoEar::Left => {
                let opposite = apply_cue(cue_style, right, sample_rate, cue_state);
                frame[0] = soft_limit(tone(left + opposite, tone_state) * gain);
            }
            MonoEar::Right if ch >= 2 => {
                let opposite = apply_cue(cue_style, left, sample_rate, cue_state);
                frame[1] = soft_limit(tone(right + opposite, tone_state) * gain);
            }
            MonoEar::Right => {
                // mono device fallback
                let opposite = apply_cue(cue_style, left, sample_rate, cue_state);
                frame[0] = soft_limit(tone(right + opposite, tone_state) * gain);
            }
            MonoEar::Both => {
                // Oba uszy już słyszą osobno lewą/prawą stronę (stereo
                // wyjście) — bez efektu cue, ale tonem objęty jest cały miks.
                let sample = soft_limit(tone(left + right, tone_state) * gain);
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
/// działać, inaczej `read_stereo_pan` będzie zwracać ciszę).
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
        let mut tone_state = ToneState::default();
        let settings = OutputSettings::plain(ear, 1.0, cue_style);
        fill_output_f32(&mut out, &pan_frames, out_channels, &settings, 48_000.0, &mut cue_state, &mut tone_state);
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
    fn pulse_fully_mutes_during_gaps_instead_of_smoothly_fading() {
        // `Pulse` to bramkowanie 0/1, więc w przeciwieństwie do `Buzz`
        // (płynna fala 0.5..1.0) powinny się pojawić próbki DOKŁADNIE zerowe.
        // Przy 48 kHz i `PULSE_HZ` = 8 Hz jeden pełny cykl to 6000 klatek —
        // trzeba ich wygenerować z zapasem, żeby test złapał obie fazy bramki.
        let out = run_frames_with_style(&(0.0, 0.7), 7_000, 2, MonoEar::Left, MonoCueStyle::Pulse);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        assert!(left_channel.iter().any(|&s| s == 0.0), "Pulse powinien mieć momenty całkowitej ciszy");
        assert!(left_channel.iter().any(|&s| (s - 0.7).abs() < 1e-6), "Pulse powinien mieć momenty pełnej głośności");
    }

    #[test]
    fn muffled_smooths_a_step_instead_of_oscillating() {
        // `Muffled` to filtr dolnoprzepustowy: przy stałym wejściu wartość
        // powinna monotonicznie zbliżać się do sygnału, NIE oscylować jak
        // pozostałe style.
        let out = run_frames_with_style(&(0.0, 0.7), 64, 2, MonoEar::Left, MonoCueStyle::Muffled);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        for pair in left_channel.windows(2) {
            assert!(pair[1] + 1e-6 >= pair[0], "filtr dolnoprzepustowy powinien narastać monotonicznie: {left_channel:?}");
        }
        assert!(left_channel.last().unwrap() > &0.3, "po 64 próbkach filtr powinien zdążyć zauważalnie narosnąć");
    }

    #[test]
    fn dark_has_a_lower_cutoff_than_muffled_so_it_rises_slower() {
        // Ten sam test co `muffled_smooths_a_step_instead_of_oscillating`,
        // ale porównawczy: przy identycznym skoku wejścia i tej samej liczbie
        // próbek `Dark` (niższe odcięcie) powinien być dalej od celu niż
        // `Muffled` — czyli brzmieć wyraźniej "głucho".
        let sample_rate = 48_000.0;
        let mut muffled_state = CueState::default();
        let mut dark_state = CueState::default();
        let mut muffled_out = 0.0;
        let mut dark_out = 0.0;
        for _ in 0..64 {
            muffled_out = apply_cue(MonoCueStyle::Muffled, 0.7, sample_rate, &mut muffled_state);
            dark_out = apply_cue(MonoCueStyle::Dark, 0.7, sample_rate, &mut dark_state);
        }
        assert!(
            dark_out < muffled_out,
            "Dark (niższe odcięcie) powinien narastać wolniej niż Muffled: dark={dark_out} muffled={muffled_out}"
        );
    }

    #[test]
    fn bright_leaves_a_steady_signal_unchanged_once_settled() {
        // Przy STAŁYM sygnale składowa wysokotonowa (dopełnienie lowpassu)
        // zanika do zera, więc po ustabilizowaniu się filtra `Bright` nie
        // powinien nic dokładać — inaczej cichy, ciągły dźwięk brzmiałby
        // głośniej niż w rzeczywistości.
        let sample_rate = 48_000.0;
        let mut state = CueState::default();
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
        let mut state = CueState::default();
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
        let mut tone_state = ToneState::default();
        // ~2.0x, so (0.35+0.35)*2.0 ≈ 1.4 — over the old ±1.0 clamp
        let boosted = OutputSettings::plain(MonoEar::Left, db_to_linear(6.0), MonoCueStyle::Off);
        fill_output_f32(&mut out, &pan_frames, 2, &boosted, 48_000.0, &mut cue_state, &mut tone_state);
        let left_channel: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        assert!(left_channel.iter().all(|&s| s <= 1.0 && s >= -1.0), "limiter nie może przekroczyć zakresu wyjścia");
        assert!(
            left_channel.iter().any(|&s| s < 1.0),
            "przy miękkim dogięciu przesterowany sygnał NIE powinien lądować dokładnie na płaskim 1.0 co klatkę: {left_channel:?}"
        );
    }
}
