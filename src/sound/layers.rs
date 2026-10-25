//! Warstwy dźwięku dla radaru „3+3”: sześć słupków, po trzy na stronę (lewo/prawo),
//! ułożonych jak cebula. Od środka: strzały i wybuchy, kroki, reszta. Obok nich pasek „ja”
//! dla własnych dźwięków gracza.
//!
//! Zdarzenia wykrywa [`LayerTracker`] z poziomu ramek 10 ms, tą samą regułą co
//! `HearAnalytics-ML/hearanalytics/detect_events.py`. Każde zdarzenie dostaje klasę A–D z poziomu
//! szczytu i jasności widma (progi jak w `features.py`). Klasa wskazuje warstwę. Granat to klasa D
//! z krótkim odstępem od poprzedniego zdarzenia, więc liczy się jako strzał. Kierunek to balans L/P
//! w ramce szczytu, ta sama definicja co `balans_lp`. Sprawdzają to `radar --classify-clips`
//! i `radar --classify-session`.
//!
//! Mapowanie klasa → warstwa jest tymczasowe (patrz [`layer_for`]).

use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Progi klas (dBFS i dB), te same co w `features.py`.
pub const CLASS_LOUD_DBFS: f32 = -22.0;
pub const CLASS_MID_DBFS: f32 = -33.0;
pub const CLASS_BRIGHT_DB: f32 = -12.0;

/// Zakres poziomu słupka: poniżej `FLOOR` słupek jest pusty, od `CEIL` pełny.
const LEVEL_FLOOR_DBFS: f32 = -50.0;
const LEVEL_CEIL_DBFS: f32 = -5.0;

/// Długość ramki analizy (jak `FRAME_S` w `detect_events.py`).
const FRAME_S: f32 = 0.01;
/// Reguła wykrywania zdarzeń, ta sama co w `detect_events.py`: poziom musi wzrosnąć o 9 dB
/// ponad tło, czyli minimum z ostatnich 0,5 s (50 ramek), i być co najmniej −40 dBFS.
const DETECT_THRESHOLD_DB: f32 = 9.0;
const DETECT_MIN_LEVEL_DBFS: f32 = -40.0;
const BASELINE_FRAMES: usize = 50;
/// Onset krótszy niż to od poprzedniego zdarzenia to ten sam dźwięk (jak `refractory_s` w `detect_events.py`).
const EVENT_REFRACTORY_S: f32 = 0.15;
/// Szczyt szukany w tym oknie po onsecie (jak w `detect_events.py`).
const PEAK_SEARCH_S: f32 = 0.1;
/// Okno jasności wokół szczytu: 10 ms przed nim, 40 ms długości.
const BRIGHTNESS_BEFORE_S: f32 = 0.01;
const BRIGHTNESS_LENGTH_S: f32 = 0.04;
const LOW_CUTOFF_HZ: f32 = 500.0;
const HIGH_CUTOFF_HZ: f32 = 2000.0;
/// Najkrótsze okno, dla którego liczymy jasność (jak `MIN_SAMPLES` w Pythonie).
const MIN_BRIGHTNESS_SAMPLES: usize = 64;
/// Ile sekund próbek trzymamy w buforze. Okno decyzji to ok. 0,35 s.
const HISTORY_S: f32 = 1.0;
/// Klasa D z krótszym odstępem od poprzedniego zdarzenia to granat, czyli strzał/wybuch
/// (jak `GRENADE_MAX_GAP_S` w `features.py`).
pub const GRENADE_MAX_GAP_S: f32 = 0.5;

/// Balans L/P, przy którym wychylenie jest pełne. Zwykłe dźwięki mają balans rzędu 0,03.
const PAN_FULL_SCALE: f32 = 0.3;
/// Poniżej tej sumy wysokości słupków żadna strona nie dominuje (cisza albo szum).
const DOMINANT_MIN_HEIGHT: f32 = 0.05;
/// Dźwięk bliżej środka niż to i z warstwy strzał/krok to kandydat na własny dźwięk gracza.
pub const OWN_MAX_PAN: f32 = 0.2;

/// Ile trafień trzymamy na jednym słupku.
const MAX_HITS_PER_BAR: usize = 16;
/// Wykładnik zaniku słupka. Taki sam jak zanik kropek w `radar_egui`.
const BAR_FADE_EXPONENT: f32 = 1.6;

/// Klasa zdarzenia według progów z `features.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Głośne (≥ −22 dBFS).
    A,
    /// Średnie (od −33 do −22 dBFS).
    B,
    /// Ciche i ciemne.
    C,
    /// Ciche i jasne.
    D,
}

/// Warstwa na słupkach. Indeks to pozycja od środka: 0 najbliżej, 2 najdalej.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Strzały i wybuchy (najważniejsze, najbliżej środka).
    Shots = 0,
    /// Kroki.
    Steps = 1,
    /// Reszta (najbardziej zewnętrzny słupek).
    Rest = 2,
}

impl Layer {
    pub const ALL: [Layer; 3] = [Layer::Shots, Layer::Steps, Layer::Rest];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn label(self) -> &'static str {
        match self {
            Layer::Shots => "strzały",
            Layer::Steps => "kroki",
            Layer::Rest => "reszta",
        }
    }
}

/// Wynik klasyfikacji jednego zdarzenia.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerHit {
    pub layer: Layer,
    /// `None`, gdy zdarzenie jest ciche i nie da się policzyć jasności.
    pub class: Option<Class>,
    /// Proporcja lewo/prawo: −1 pełne lewo, 0 środek, +1 pełne prawo.
    pub pan: f32,
    /// Poziom szczytu (dBFS), jak `peak_dbfs` w `events.csv`.
    pub peak_dbfs: f32,
    pub brightness_db: Option<f32>,
    /// Odstęp od poprzedniego zdarzenia (s), `None` dla pierwszego.
    pub gap_s: Option<f32>,
    /// Chwila onsetu od początku strumienia (s).
    pub onset_s: f64,
}

impl LayerHit {
    /// Czy zdarzenie może być własnym dźwiękiem gracza: blisko środka i ze strzałów albo kroków.
    pub fn is_own_candidate(&self) -> bool {
        self.pan.abs() < OWN_MAX_PAN && matches!(self.layer, Layer::Shots | Layer::Steps)
    }
}

/// Proporcja lewo/prawo −1..1 z poziomów RMS obu kanałów (dodatnia = prawo). Ta sama definicja
/// co `balans_lp` w `detect_events.py`, przeskalowana tak, że mocne wychylenie daje pełną wartość.
pub fn pan_from_rms(left_rms: f32, right_rms: f32) -> f32 {
    let total = left_rms + right_rms;
    if total <= 0.0 {
        return 0.0;
    }
    ((right_rms - left_rms) / total / PAN_FULL_SCALE).clamp(-1.0, 1.0)
}

/// Klasa A–D z poziomu szczytu i jasności. `None`, gdy zdarzenie jest ciche i nie ma jasności.
pub fn class_auto(peak_dbfs: f32, brightness_db: Option<f32>) -> Option<Class> {
    if peak_dbfs >= CLASS_LOUD_DBFS {
        return Some(Class::A);
    }
    if peak_dbfs >= CLASS_MID_DBFS {
        return Some(Class::B);
    }
    let brightness = brightness_db?;
    Some(if brightness > CLASS_BRIGHT_DB { Class::D } else { Class::C })
}

/// Warstwa dla klasy i odstępu. A i B to strzały/wybuchy. D to kroki, chyba że odstęp jest
/// krótki: wtedy to granat, czyli strzały. C i brak klasy to reszta. Mapowanie wynika z
/// `sesja_trening_v2` i jest zapisane w `features.layer_auto`. Zmień tu, gdy odsłuch pokaże inaczej.
pub fn layer_for(class: Option<Class>, gap_s: Option<f32>, previous: Option<Class>) -> Layer {
    match class {
        Some(Class::A) | Some(Class::B) => Layer::Shots,
        // Granat: klasa D krótko po strzale albo wybuchu, a nie po przypadkowym szumie.
        Some(Class::D)
            if gap_s.is_some_and(|gap| gap < GRENADE_MAX_GAP_S)
                && matches!(previous, Some(Class::A) | Some(Class::B)) =>
        {
            Layer::Shots
        }
        Some(Class::D) => Layer::Steps,
        Some(Class::C) | None => Layer::Rest,
    }
}

/// Poziom słupka 0..1 z poziomu szczytu w dBFS.
pub fn level_from_dbfs(peak_dbfs: f32) -> f32 {
    ((peak_dbfs - LEVEL_FLOOR_DBFS) / (LEVEL_CEIL_DBFS - LEVEL_FLOOR_DBFS)).clamp(0.0, 1.0)
}

/// Stosunek energii powyżej 2 kHz do energii poniżej 500 Hz w dB, liczony na podanym oknie.
/// Ta sama definicja co `features.brightness_db`.
pub fn brightness_db(mono: &[f32], rate: f32) -> Option<f32> {
    if mono.len() < MIN_BRIGHTNESS_SAMPLES || rate <= 0.0 {
        return None;
    }
    let low = one_pole_lowpass(mono, LOW_CUTOFF_HZ, rate);
    let low_2k = one_pole_lowpass(mono, HIGH_CUTOFF_HZ, rate);
    let energy_low: f32 = low.iter().map(|y| y * y).sum();
    let energy_high: f32 = mono.iter().zip(&low_2k).map(|(x, y)| (x - y) * (x - y)).sum();
    if energy_low <= 0.0 || energy_high <= 0.0 {
        return None;
    }
    Some(10.0 * (energy_high / energy_low).log10())
}

fn one_pole_lowpass(samples: &[f32], cutoff_hz: f32, rate: f32) -> Vec<f32> {
    let alpha = 1.0 - (-TAU * cutoff_hz / rate).exp();
    let mut y = 0.0;
    samples
        .iter()
        .map(|&x| {
            y += alpha * (x - y);
            y
        })
        .collect()
}

/// Splata kanały (każdy `Vec` to jeden kanał) w jedną tablicę: L, P, L, P, …
pub fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * channels.len());
    for frame in 0..frames {
        for channel in channels {
            out.push(channel[frame]);
        }
    }
    out
}

/// Długość ramki w próbkach przy danej częstotliwości (jak `hop` w `detect_events.py`).
fn frame_samples(rate: f32) -> u64 {
    (FRAME_S * rate).round() as u64
}

/// Jedna próbka stereo: lewy i prawy kanał.
#[derive(Clone, Copy, Debug, Default)]
struct Frame {
    left: f32,
    right: f32,
}

impl Frame {
    /// Średnia L/P, jak mono w `features.py`.
    fn mono(self) -> f32 {
        0.5 * (self.left + self.right)
    }

    /// Średnia z kwadratów obu kanałów, jak moc w `detect_events.frame_levels`.
    fn power(self) -> f32 {
        0.5 * (self.left * self.left + self.right * self.right)
    }
}

/// Zdarzenie czekające, aż minie okno szczytu i jasności.
#[derive(Clone, Copy, Debug)]
struct PendingOnset {
    at_frame: u64,
    gap_s: Option<f32>,
}

/// Bufor ostatniej sekundy próbek stereo, wykrywanie zdarzeń z ramek 10 ms i kolejka
/// zdarzeń czekających na decyzję. Reguła wykrywania jest taka sama jak w `detect_events.py`,
/// więc wyniki Rusta i Pythona są porównywalne. Pierwsze dwa kanały to lewa i prawa strona.
/// Decyzja zapada ok. 0,35 s po zdarzeniu, więc słupek pojawia się z tym opóźnieniem.
#[derive(Debug)]
pub struct LayerTracker {
    rate: f32,
    frames: VecDeque<Frame>,
    /// Numer (od początku strumienia) najstarszej próbki w buforze.
    first_frame: u64,
    /// Ile próbek tracker widział od początku.
    total_frames: u64,
    /// Suma mocy próbek bieżącej ramki 10 ms.
    frame_power: f32,
    /// Poziomy (dBFS) ostatnich ramek, do liczenia tła.
    levels: VecDeque<f32>,
    /// Przyrost nad tło z poprzedniej ramki, do wykrycia przekroczenia.
    prev_rise: f32,
    pending: VecDeque<PendingOnset>,
    /// Początek ostatnio przyjętego zdarzenia (w próbkach), do liczenia odstępu.
    last_event_frame: Option<u64>,
    /// Klasa ostatnio podjętej decyzji, do reguły granatu.
    last_class: Option<Class>,
    /// Gdy `true`, zdarzenia zgłasza zewnętrzny, szybki detektor (`add_onset`), a reguła z Pythona
    /// jest wyłączona. Tak działa mikser na żywo.
    external_onsets: bool,
}

impl Default for LayerTracker {
    fn default() -> Self {
        Self {
            rate: 48_000.0,
            frames: VecDeque::new(),
            first_frame: 0,
            total_frames: 0,
            frame_power: 0.0,
            levels: VecDeque::new(),
            prev_rise: 0.0,
            pending: VecDeque::new(),
            last_event_frame: None,
            last_class: None,
            external_onsets: false,
        }
    }
}

impl LayerTracker {
    /// Dopisuje próbki interleaved. Przy mono obie strony są takie same. Każda pełna ramka 10 ms
    /// przechodzi przez wykrywanie zdarzeń.
    pub fn push_frames(&mut self, interleaved: &[f32], channels: usize, rate: f32) {
        if channels == 0 || rate <= 0.0 {
            return;
        }
        self.rate = rate;
        let capacity = (HISTORY_S * rate) as usize;
        let hop = frame_samples(rate);
        for chunk in interleaved.chunks_exact(channels) {
            let (left, right) = if channels >= 2 {
                (chunk[0], chunk[1])
            } else {
                (chunk[0], chunk[0])
            };
            let frame = Frame { left, right };
            self.frame_power += frame.power();
            self.frames.push_back(frame);
            self.total_frames += 1;
            if self.frames.len() > capacity {
                self.frames.pop_front();
                self.first_frame += 1;
            }
            if self.total_frames % hop == 0 {
                let level = 10.0 * (self.frame_power / hop as f32).max(1e-18).log10();
                self.frame_power = 0.0;
                self.on_frame(level, self.total_frames - hop);
            }
        }
    }

    /// Jedna pełna ramka zaczynająca się w `start`. Reguła z `detect_events.py`: przekroczenie
    /// tła o 9 dB (przy wejściu w przekroczenie, nie przez cały czas), poziom co najmniej
    /// −40 dBFS i odstęp co najmniej 0,15 s od poprzedniego zdarzenia.
    fn on_frame(&mut self, level: f32, start: u64) {
        let base = self.levels.iter().copied().reduce(f32::min).unwrap_or(level);
        let rise = level - base;
        let crossed = rise >= DETECT_THRESHOLD_DB && self.prev_rise < DETECT_THRESHOLD_DB;
        self.prev_rise = rise;
        self.levels.push_back(level);
        if self.levels.len() > BASELINE_FRAMES {
            self.levels.pop_front();
        }
        // Przy zewnętrznym detektorze reguła z Pythona nie zgłasza zdarzeń, tylko tło jest liczone.
        if self.external_onsets {
            return;
        }
        if !crossed || level < DETECT_MIN_LEVEL_DBFS {
            return;
        }
        self.register_onset(start);
    }

    /// Przyjmuje zdarzenie zaczynające się w `start`. Zdarzenie krótsze niż `EVENT_REFRACTORY_S`
    /// od poprzedniego to ten sam dźwięk i jest pomijane. Odstęp od poprzedniego zapisuje się
    /// razem ze zdarzeniem.
    fn register_onset(&mut self, start: u64) {
        let refractory = (EVENT_REFRACTORY_S * self.rate) as u64;
        let gap_s = match self.last_event_frame {
            Some(last) if start < last + refractory => return,
            Some(last) => Some((start - last) as f32 / self.rate),
            None => None,
        };
        self.last_event_frame = Some(start);
        self.pending.push_back(PendingOnset {
            at_frame: start,
            gap_s,
        });
    }

    /// Czy zewnętrzny onset przechodzi regułę z Pythona: poziom ostatniej pełnej ramki co najmniej
    /// −40 dBFS i 9 dB ponad minimum z 0,5 s przed nią. Odrzuca szum przy starcie i ogon dźwięku.
    pub fn onset_allowed(&self) -> bool {
        let Some(&level) = self.levels.back() else {
            return false;
        };
        let Some(base) = self.levels.iter().rev().skip(1).copied().reduce(f32::min) else {
            return false;
        };
        level >= DETECT_MIN_LEVEL_DBFS && level - base >= DETECT_THRESHOLD_DB
    }

    /// Tryb dla miksera na żywo: zdarzenia zgłasza szybki detektor kanałów przez `add_onset`,
    /// a reguła z Pythona jest wyłączona. Reszta (klasy, odstęp, balans) działa tak samo.
    pub fn with_external_onsets(mut self) -> Self {
        self.external_onsets = true;
        self
    }

    /// Czy zdarzenia zgłasza zewnętrzny detektor.
    pub fn has_external_onsets(&self) -> bool {
        self.external_onsets
    }

    /// Zgłasza zdarzenie z zewnętrznego detektora w bieżącej chwili (koniec ostatnio dopisanych
    /// próbek). Dla trybu z `with_external_onsets`.
    pub fn add_onset(&mut self) {
        let at = self.total_frames;
        self.register_onset(at);
    }

    /// Zwraca decyzje dla zdarzeń, którym minęło już okno szczytu i jasności.
    pub fn take_ready(&mut self) -> Vec<LayerHit> {
        let delay = ((PEAK_SEARCH_S + BRIGHTNESS_LENGTH_S + FRAME_S) * self.rate) as u64;
        let mut hits = Vec::new();
        while let Some(&onset) = self.pending.front() {
            if self.total_frames < onset.at_frame + delay {
                break;
            }
            self.pending.pop_front();
            if let Some(hit) = self.decide(onset, self.last_class) {
                self.last_class = hit.class;
                hits.push(hit);
            }
        }
        hits
    }

    /// Szuka szczytu w oknie 0,3 s po zdarzeniu (ramki 10 ms, pierwszy najgłośniejszy),
    /// liczy jasność w oknie szczytu i balans L/P w ramce szczytu, a potem przydziela warstwę.
    /// Tak samo jak `detect_events.py` i `features.py`.
    fn decide(&self, onset: PendingOnset, previous: Option<Class>) -> Option<LayerHit> {
        let hop = frame_samples(self.rate);
        let steps = (PEAK_SEARCH_S / FRAME_S).round() as u64;
        let mut peak: Option<(u64, f32)> = None;
        for k in 0..=steps {
            let start = onset.at_frame + k * hop;
            let level = self.frame_level_dbfs(start, hop as usize)?;
            if peak.map_or(true, |(_, best)| level > best) {
                peak = Some((start, level));
            }
        }
        let (peak_start, peak_dbfs) = peak?;
        let window_start = peak_start.checked_sub((BRIGHTNESS_BEFORE_S * self.rate).round() as u64)?;
        let mono = self.mono_window(window_start, (BRIGHTNESS_LENGTH_S * self.rate).round() as usize)?;
        let brightness = brightness_db(&mono, self.rate);
        let class = class_auto(peak_dbfs, brightness);
        let pan = self.pan_at(peak_start);
        Some(LayerHit {
            layer: layer_for(class, onset.gap_s, previous),
            class,
            pan,
            peak_dbfs,
            brightness_db: brightness,
            gap_s: onset.gap_s,
            onset_s: onset.at_frame as f64 / self.rate as f64,
        })
    }

    /// Próbki od `start` (numer od początku strumienia), jeśli wszystkie są w buforze.
    fn slice(&self, start: u64, len: usize) -> Option<impl Iterator<Item = &Frame>> {
        if start < self.first_frame || start + len as u64 > self.total_frames {
            return None;
        }
        let from = (start - self.first_frame) as usize;
        Some(self.frames.range(from..from + len))
    }

    /// Balans L/P w ramce 10 ms od `start`, przez [`pan_from_rms`].
    fn pan_at(&self, start: u64) -> f32 {
        let Some(frames) = self.slice(start, frame_samples(self.rate) as usize) else {
            return 0.0;
        };
        let (left, right) = frames.fold((0.0f32, 0.0f32), |(left, right), frame| {
            (left + frame.left * frame.left, right + frame.right * frame.right)
        });
        pan_from_rms(left.sqrt(), right.sqrt())
    }

    fn frame_level_dbfs(&self, start: u64, len: usize) -> Option<f32> {
        let (sum, count) = self
            .slice(start, len)?
            .fold((0.0f32, 0usize), |(sum, count), frame| (sum + frame.power(), count + 1));
        if count == 0 {
            return None;
        }
        Some(10.0 * (sum / count as f32).max(1e-18).log10())
    }

    fn mono_window(&self, start: u64, len: usize) -> Option<Vec<f32>> {
        let mono: Vec<f32> = self.slice(start, len)?.map(|frame| frame.mono()).collect();
        (mono.len() >= MIN_BRIGHTNESS_SAMPLES).then_some(mono)
    }
}

/// Jedno trafienie na słupku: czas UI i poziom 0..1 po podziale na strony.
#[derive(Clone, Copy, Debug)]
struct BarHit {
    at: f64,
    level: f32,
}

/// Zanik jednego trafienia w chwili `now`: 1 świeże, 0 wygasłe.
fn fade(age: f64, decay_secs: f32) -> f32 {
    let life = 1.0 - (age as f32) / decay_secs;
    if age < 0.0 || life <= 0.0 {
        0.0
    } else {
        life.powf(BAR_FADE_EXPONENT)
    }
}

/// Stan radaru 3+3: sześć słupków (strona × warstwa) i pasek „ja”.
#[derive(Debug, Default)]
pub struct LayerBars {
    hits: [Vec<BarHit>; 6],
    /// Indeks wybranej pary w `Layer::ALL`. Zawsze jedna z trzech.
    highlight: usize,
}

impl LayerBars {
    /// Zapisuje trafienie z chwilą `at` (s UI). Poziom dzieli się między strony według proporcji:
    /// to, co mocniej z prawej, wypełnia prawy słupek bardziej. Środek idzie po połowie na oba.
    pub fn record(&mut self, at: f64, hit: LayerHit) {
        let level = level_from_dbfs(hit.peak_dbfs);
        let left_share = (1.0 - hit.pan) * 0.5;
        let right_share = (1.0 + hit.pan) * 0.5;
        self.push(0, hit.layer, BarHit { at, level: level * left_share });
        self.push(1, hit.layer, BarHit { at, level: level * right_share });
    }

    fn push(&mut self, side: usize, layer: Layer, hit: BarHit) {
        let list = &mut self.hits[side * Layer::ALL.len() + layer.index()];
        if list.len() == MAX_HITS_PER_BAR {
            list.remove(0);
        }
        list.push(hit);
    }

    /// Wysokość słupka 0..1 w chwili `now`: najmocniejsze jeszcze widoczne trafienie z zanikiem.
    /// `side` to 0 dla lewej, 1 dla prawej.
    pub fn height(&self, side: usize, layer: Layer, now: f64, decay_secs: f32) -> f32 {
        self.hits[side * Layer::ALL.len() + layer.index()]
            .iter()
            .map(|hit| hit.level * fade(now - hit.at, decay_secs))
            .fold(0.0, f32::max)
    }

    /// Warstwa podświetlonej pary (L i P tej samej warstwy). Zawsze jedna z trzech.
    pub fn highlighted_layer(&self) -> Layer {
        Layer::ALL[self.highlight]
    }

    /// Strona wyższego słupka wybranej pary: `0` lewa, `1` prawa. `None`, gdy słupki są równe
    /// albo oba są pod progiem (cisza).
    pub fn focus_side(&self, now: f64, decay_secs: f32) -> Option<usize> {
        let layer = self.highlighted_layer();
        let left = self.height(0, layer, now, decay_secs);
        let right = self.height(1, layer, now, decay_secs);
        if left.max(right) < DOMINANT_MIN_HEIGHT || left == right {
            None
        } else if left > right {
            Some(0)
        } else {
            Some(1)
        }
    }

    /// Wybiera parę z największą aktywnością, ale tylko gdy któraś jest wyraźnie aktywna.
    /// Dzięki temu w ciszy zostaje ostatnio wybrana para.
    fn update_highlight(&mut self, now: f64, decay_secs: f32) {
        let mut best = (self.highlight, DOMINANT_MIN_HEIGHT);
        for (index, &layer) in Layer::ALL.iter().enumerate() {
            let activity = self
                .height(0, layer, now, decay_secs)
                .max(self.height(1, layer, now, decay_secs));
            if activity > best.1 {
                best = (index, activity);
            }
        }
        self.highlight = best.0;
    }


    /// Usuwa trafienia, które już wygasły.
    pub fn prune(&mut self, now: f64, decay_secs: f32) {
        for list in &mut self.hits {
            list.retain(|hit| now - hit.at <= decay_secs as f64);
        }
        self.update_highlight(now, decay_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 48_000.0;

    fn sine(freq: f32, amp: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| amp * (TAU * freq * i as f32 / RATE).sin()).collect()
    }

    /// Biały szum (xorshift) w zakresie −amp..amp.
    fn noise(amp: f32, n: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// Dźwięk zaczynający się w 0,2 s, trwający tyle, ile `body`. Całość 1 s.
    fn burst_at_onset(body: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; RATE as usize];
        let onset = (0.2 * RATE) as usize;
        out[onset..onset + body.len()].copy_from_slice(body);
        out
    }

    /// Jak na żywo: kawałkami po 10 ms, decyzje zbierane po drodze (bufor trzyma tylko sekundę).
    fn classify_stereo(left: &[f32], right: &[f32]) -> Vec<LayerHit> {
        let mut tracker = LayerTracker::default();
        let mut hits = Vec::new();
        let stereo: Vec<f32> = left.iter().zip(right).flat_map(|(&l, &r)| [l, r]).collect();
        for part in stereo.chunks(480 * 2) {
            tracker.push_frames(part, 2, RATE);
            hits.extend(tracker.take_ready());
        }
        hits
    }

    fn classify_mono(mono: &[f32]) -> Vec<LayerHit> {
        classify_stereo(mono, mono)
    }

    /// Cichy szum (klasa D, bo jasny) o poziomie ok. −36 dBFS, po 0,1 s od każdego zdarzenia.
    /// Całość 2 s.
    fn quiet_noise_at(onsets_s: &[f64]) -> Vec<f32> {
        let mut out = vec![0.0f32; 2 * RATE as usize];
        let body = noise(0.027, 4_800);
        for &onset in onsets_s {
            let start = (onset * RATE as f64) as usize;
            out[start..start + body.len()].copy_from_slice(&body);
        }
        out
    }

    fn hit(layer: Layer, peak_dbfs: f32, pan: f32) -> LayerHit {
        LayerHit {
            layer,
            class: None,
            pan,
            peak_dbfs,
            brightness_db: None,
            gap_s: None,
            onset_s: 0.0,
        }
    }

    #[test]
    fn class_follows_level_then_brightness() {
        assert_eq!(class_auto(-8.3, None), Some(Class::A));
        assert_eq!(class_auto(-22.0, None), Some(Class::A), "granica należy do A");
        assert_eq!(class_auto(-30.0, None), Some(Class::B));
        assert_eq!(class_auto(-37.6, Some(-22.6)), Some(Class::C));
        assert_eq!(class_auto(-38.2, Some(-7.6)), Some(Class::D));
        assert_eq!(class_auto(-38.2, None), None, "cichy bez jasności nie ma klasy");
    }

    #[test]
    fn layers_follow_classes_and_gap() {
        assert_eq!(layer_for(Some(Class::A), None, None), Layer::Shots);
        assert_eq!(layer_for(Some(Class::B), Some(2.0), None), Layer::Shots);
        assert_eq!(layer_for(Some(Class::D), Some(0.3), Some(Class::B)), Layer::Shots, "granat: krótki odstęp po strzale");
        assert_eq!(layer_for(Some(Class::D), Some(0.3), Some(Class::D)), Layer::Steps, "szum przed krokiem to nie strzał");
        assert_eq!(layer_for(Some(Class::D), Some(0.72), Some(Class::B)), Layer::Steps, "krok: dłuższy odstęp");
        assert_eq!(layer_for(Some(Class::D), None, None), Layer::Steps, "pierwsze zdarzenie");
        assert_eq!(layer_for(Some(Class::C), Some(0.1), Some(Class::A)), Layer::Rest);
        assert_eq!(layer_for(None, None, None), Layer::Rest);
    }

    #[test]
    fn pan_follows_the_level_balance_with_a_full_scale() {
        assert_eq!(pan_from_rms(1.0, 1.0), 0.0, "równo: środek");
        assert_eq!(pan_from_rms(1.0, 0.0), -1.0, "tylko lewy: pełne lewo");
        assert_eq!(pan_from_rms(0.0, 1.0), 1.0, "tylko prawy: pełne prawo");
        assert!((pan_from_rms(0.9, 1.0) - 0.175).abs() < 1e-3, "lekkie wychylenie w prawo");
        assert_eq!(pan_from_rms(0.0, 0.0), 0.0, "cisza: środek");
    }

    #[test]
    fn external_onsets_must_rise_above_the_background() {
        let mut tracker = LayerTracker::default().with_external_onsets();
        assert!(!tracker.onset_allowed(), "na starcie nie ma tła");
        let silence = vec![0.0f32; 2 * 28_800];
        tracker.push_frames(&silence, 2, RATE);
        let loud: Vec<f32> = noise(0.2, 4_800).iter().flat_map(|&s| [s, s]).collect();
        tracker.push_frames(&loud, 2, RATE);
        assert!(tracker.onset_allowed(), "głośny dźwięk nad ciszą przechodzi");
        let quiet: Vec<f32> = noise(0.001, 4_800).iter().flat_map(|&s| [s, s]).collect();
        tracker.push_frames(&quiet, 2, RATE);
        assert!(!tracker.onset_allowed(), "poniżej −40 dBFS odrzucamy");
    }

    #[test]
    fn external_onsets_are_classified_without_the_python_rule() {
        let mono = burst_at_onset(&noise(0.027, 4_800));
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let mut tracker = LayerTracker::default().with_external_onsets();
        let mut hits = Vec::new();
        for (index, frame) in stereo.chunks(480 * 2).enumerate() {
            if index == 20 {
                tracker.add_onset(); // zdarzenie w 0,2 s zgłoszone z zewnątrz
            }
            tracker.push_frames(frame, 2, RATE);
            hits.extend(tracker.take_ready());
        }
        assert_eq!(hits.len(), 1);
        assert!((hits[0].onset_s - 0.2).abs() < 0.02, "onset: {}", hits[0].onset_s);
        assert_eq!(hits[0].class, Some(Class::D));
    }

    #[test]
    fn brightness_is_high_for_treble_and_low_for_bass() {
        let treble = brightness_db(&sine(5_000.0, 0.5, 1920), RATE).unwrap();
        let bass = brightness_db(&sine(100.0, 0.5, 1920), RATE).unwrap();
        assert!(treble > 10.0, "wysokie tony: {treble}");
        assert!(bass < -10.0, "niskie tony: {bass}");
    }

    #[test]
    fn brightness_needs_a_long_enough_window() {
        assert_eq!(brightness_db(&[0.1; 10], RATE), None);
    }

    #[test]
    fn level_maps_the_dbfs_range_onto_the_bar() {
        assert_eq!(level_from_dbfs(-50.0), 0.0);
        assert_eq!(level_from_dbfs(-5.0), 1.0);
        assert!((level_from_dbfs(-27.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn interleave_puts_channels_side_by_side() {
        assert_eq!(interleave(&[vec![1.0, 2.0], vec![3.0, 4.0]]), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn loud_burst_is_a_shot_and_pans_to_the_louder_side() {
        let left = burst_at_onset(&sine(1_000.0, 0.2, 4_800)); // ok. −17 dBFS
        let right: Vec<f32> = left.iter().map(|s| s * 0.8).collect(); // prawy ciszej
        let hits = classify_stereo(&left, &right);
        assert_eq!(hits.len(), 1, "jedno zdarzenie, jedna decyzja");
        assert!((hits[0].onset_s - 0.2).abs() < 0.01, "zdarzenie w 0,2 s: {}", hits[0].onset_s);
        assert_eq!(hits[0].layer, Layer::Shots);
        assert_eq!(hits[0].class, Some(Class::A));
        assert!(hits[0].pan < -0.3 && hits[0].pan > -0.45, "lewy głośniejszy: {}", hits[0].pan);
    }

    #[test]
    fn quiet_bright_burst_is_a_step() {
        let hits = classify_mono(&burst_at_onset(&noise(0.027, 4_800)));
        assert_eq!(hits[0].class, Some(Class::D));
        assert_eq!(hits[0].layer, Layer::Steps);
    }

    #[test]
    fn quiet_dark_burst_is_rest() {
        let hits = classify_mono(&burst_at_onset(&sine(100.0, 0.0224, 4_800)));
        assert_eq!(hits[0].class, Some(Class::C));
        assert_eq!(hits[0].layer, Layer::Rest);
    }

    /// Strzał w 0,2 s, potem cichy dźwięk w 0,4 s i w 1,2 s. Całość 2 s.
    fn shot_then_quiet_sounds() -> Vec<f32> {
        let mut out = vec![0.0f32; 2 * RATE as usize];
        let shot = sine(1_000.0, 0.2, 4_800);
        let quiet = noise(0.027, 4_800);
        for (start, body) in [(9_600usize, &shot), (19_200, &quiet), (57_600, &quiet)] {
            out[start..start + body.len()].copy_from_slice(body);
        }
        out
    }

    #[test]
    fn a_quiet_sound_right_after_a_shot_is_a_grenade() {
        // Strzał, potem cichy dźwięk 0,2 s później → granat (strzały). Kolejny po 0,8 s → kroki.
        let hits = classify_mono(&shot_then_quiet_sounds());
        let layers: Vec<Layer> = hits.iter().map(|h| h.layer).collect();
        assert_eq!(layers, vec![Layer::Shots, Layer::Shots, Layer::Steps]);
        assert!((hits[1].gap_s.unwrap() - 0.2).abs() < 1e-3);
        assert!((hits[2].gap_s.unwrap() - 0.8).abs() < 1e-3);
    }

    #[test]
    fn quiet_noise_before_a_step_does_not_make_a_grenade() {
        // Dwa ciche dźwięki 0,2 s od siebie: to kroki, bo przed nimi nie było strzału.
        let onsets = [0.2, 0.4];
        let hits = classify_mono(&quiet_noise_at(&onsets));
        let layers: Vec<Layer> = hits.iter().map(|h| h.layer).collect();
        assert_eq!(layers, vec![Layer::Steps, Layer::Steps]);
    }

    #[test]
    fn a_second_click_within_the_refractory_period_is_ignored() {
        // Dwa krótkie kliknięcia 0,06 s od siebie: w `detect_events.py` to jedno zdarzenie.
        let mut mono = vec![0.0f32; 2 * RATE as usize];
        let click = noise(0.027, 960); // 20 ms
        for start in [9_600, 12_480] {
            mono[start..start + click.len()].copy_from_slice(&click);
        }
        assert_eq!(classify_mono(&mono).len(), 1);
    }

    #[test]
    fn decision_waits_for_the_peak_window() {
        let mono = burst_at_onset(&noise(0.027, 4_800));
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let mut tracker = LayerTracker::default();
        tracker.push_frames(&stereo[..2 * 13_200], 2, RATE); // do 0,275 s
        assert!(tracker.take_ready().is_empty(), "okno szczytu jeszcze nie minęło");
        tracker.push_frames(&stereo[2 * 13_200..2 * 19_200], 2, RATE); // do 0,4 s
        assert_eq!(tracker.take_ready().len(), 1);
    }

    #[test]
    fn bar_height_follows_the_pan_and_fades_out() {
        let mut bars = LayerBars::default();
        bars.record(1.0, hit(Layer::Shots, -14.0, -1.0));
        let fresh = bars.height(0, Layer::Shots, 1.0, 1.5);
        let later = bars.height(0, Layer::Shots, 1.5, 1.5);
        assert!(fresh > later && later > 0.0, "zanik: {fresh} → {later}");
        assert_eq!(bars.height(1, Layer::Shots, 1.0, 1.5), 0.0, "prawy słupek pusty przy pełnym wychyleniu w lewo");
        assert_eq!(bars.height(0, Layer::Shots, 2.6, 1.5), 0.0, "po zaniku pusty");
    }

    #[test]
    fn stronger_side_fills_more_and_gets_the_frame() {
        let mut bars = LayerBars::default();
        bars.record(0.0, hit(Layer::Steps, -20.0, 0.6));
        bars.prune(0.0, 1.5);
        let left = bars.height(0, Layer::Steps, 0.0, 1.5);
        let right = bars.height(1, Layer::Steps, 0.0, 1.5);
        assert!(right > left && left > 0.0, "mocniej z prawej → prawy bardziej wypełniony: {left} / {right}");
        assert_eq!(bars.highlighted_layer(), Layer::Steps, "wybrana para to ta, która świeci");
        assert_eq!(bars.focus_side(0.0, 1.5), Some(1), "prawy słupek dostaje obwódkę");
    }

    #[test]
    fn one_pair_is_always_highlighted_and_sticks_in_silence() {
        let mut bars = LayerBars::default();
        assert_eq!(bars.highlighted_layer(), Layer::Shots, "na starcie pierwsza para");
        bars.record(0.0, hit(Layer::Steps, -20.0, 0.0));
        bars.prune(0.0, 1.5);
        assert_eq!(bars.highlighted_layer(), Layer::Steps);
        bars.prune(10.0, 1.5);
        assert_eq!(bars.highlighted_layer(), Layer::Steps, "w ciszy zostaje ostatnio wybrana para");
    }

    #[test]
    fn centre_hit_lights_both_sides_equally() {
        let mut bars = LayerBars::default();
        bars.record(0.0, hit(Layer::Rest, -20.0, 0.0));
        let left = bars.height(0, Layer::Rest, 0.0, 1.5);
        let right = bars.height(1, Layer::Rest, 0.0, 1.5);
        assert!(left > 0.0);
        assert!((left - right).abs() < 1e-6);
        bars.prune(0.0, 1.5);
        assert_eq!(bars.focus_side(0.0, 1.5), None, "równe słupki: bez obwódki");
    }

    #[test]
    fn own_candidates_are_centred_shots_and_steps_only() {
        assert!(hit(Layer::Shots, -14.0, 0.05).is_own_candidate(), "strzał na środku to własny dźwięk");
        assert!(!hit(Layer::Rest, -20.0, 0.0).is_own_candidate(), "reszta to nie własny dźwięk");
        assert!(!hit(Layer::Shots, -14.0, -0.6).is_own_candidate(), "mocno z boku to nie ja");
    }
}
