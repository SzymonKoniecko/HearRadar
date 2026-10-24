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
    /// Kąt kropki na tarczy radaru.
    ///
    /// Tu każdy kanał ma WŁASNY, odrębny kierunek — front, side i rear są
    /// wizualnie rozróżnialne (istotne dla radaru 7.1). Konwencja: `0°` = przód/góra ekranu,
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
    /// RMS (root-mean-square) każdego kanału.
    pub rms: Vec<f32>,
}

/// Oblicza RMS dla każdego kanału na podstawie znormalizowanych,
/// przeplecionych (interleaved) próbek.
///
/// Kluczowe: w przeciwieństwie do poprzedniej implementacji, ta funkcja
/// analizuje WSZYSTKIE kanały, nie tylko pierwsze dwa — dzięki temu działa
/// poprawnie dla 5.1 i 7.1.
pub fn analyze_channels(samples: &[f32], channels: usize) -> ChannelStats {
    let channels = channels.max(1);
    let mut sum_sq = vec![0.0f32; channels];
    let mut frame_count = 0usize;

    for frame in samples.chunks_exact(channels) {
        frame_count += 1;
        for (i, &sample) in frame.iter().enumerate() {
            sum_sq[i] += sample * sample;
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

    ChannelStats { rms }
}

/// Mnożnik skalujący średnią energię do zakresu intensywności 0.0..=1.0.
/// Dobrany tak, by dla 2.0 zachować zgodność z poprzednim zachowaniem
/// (`(rms_l + rms_r) * 4.0`), ponieważ `avg_energy * 8.0 == (rms_l+rms_r) * 4.0`
/// gdy kanałów jest dwa.
pub(crate) const INTENSITY_SCALE: f32 = 8.0;

/// Zawija kąt w stopniach do zakresu `(-180, 180]`.
fn wrap_angle_deg(angle: f32) -> f32 {
    ((angle + 180.0).rem_euclid(360.0)) - 180.0
}

// ── Detekcja niezależnych zdarzeń dźwiękowych (onsetów) ────────────────────
//
// W przeciwieństwie do JEDNEGO, sumarycznego wektora dla
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

/// Próg odległości kątowej (stopnie) sąsiedniego kanału, którego energia
/// wpływa na [`onset_angle_deg`] danego kanału — WYŁĄCZNIE ŚCIŚLE PONIŻEJ tej
/// wartości (`<`, nie `<=`): kanały leżące PO PRZECIWNEJ stronie tarczy
/// (`>= 90°`) nigdy nie powinny przeciągać kąta na swoją stronę. Ostry próg
/// jest tu kluczowy — `FrontLeft`/`FrontRight` leżą dokładnie 90° od siebie,
/// więc przy `<=` ta para (czyli zwykłe stereo!) wzajemnie się uśredniała,
/// zamiast pozostać wyraźnie rozdzielona.
const NEIGHBOR_MAX_ANGLE_DIFF_DEG: f32 = 90.0;

/// Próg mnożnikowy nagłego skoku energii W POJEDYNCZYM PAŚMIE kolorowania
/// (patrz [`crate::sound::spectrum::ColorRule`]) nad jego WŁASNĄ średnią
/// kroczącą — niższy niż [`TRANSIENT_RATIO`], bo pojedyncze, węższe pasmo z
/// natury ma wyższy stosunek sygnał/tło przy realnym zdarzeniu niż całe
/// szerokopasmowe RMS kanału (nie miesza się z energią spoza swojego zakresu).
///
/// To rozwiązuje ten sam problem co `ColorRule::weight` (patrz `spectrum.rs`),
/// ale dla SAMEGO WYKRYCIA onsetu, nie tylko jego koloru: cichy, ale
/// charakterystyczny dźwięk (alarm, dzwonek) w jednym paśmie dotąd w ogóle
/// nie generował markera, bo ginął w porównaniu z szerokopasmowym RMS kanału
/// zdominowanym przez głośniejsze pasmo (np. bas). Sprawdzając KAŻDE pasmo
/// osobno przeciw JEGO WŁASNEMU tłu (patrz [`detect_channel_activity`]),
/// taki dźwięk generuje onset nawet gdy energia szerokopasmowa kanału ledwo
/// drgnie.
pub const BAND_TRANSIENT_RATIO: f32 = 2.0;

/// Jak [`ONSET_MIN_ENERGY`], ale dla energii pojedynczego pasma — niższy, bo
/// jedno wąskie pasmo z natury niesie mniej energii niż cały sygnał
/// szerokopasmowy tego samego kanału.
const BAND_ONSET_MIN_ENERGY: f32 = 0.004;

/// Frakcja średniej energii KANAŁÓW KIERUNKOWYCH (bez LFE) odejmowana od
/// energii każdego z nich (i szerokopasmowo, i per pasmo) przed sprawdzeniem
/// onsetu — analogia do "contrast" z SoundRadar (podobny, konkurencyjny
/// projekt, patrz analiza w dokumentacji zmian tego modułu). Włączane
/// opcjonalnie przez [`crate::sound::devices::RadarConfig::direction_contrast_enabled`].
///
/// Matematycznie: gdy WSZYSTKIE kanały kierunkowe rosną RÓWNO (naprawdę
/// niekierunkowy dźwięk — otoczenie, muzyka w tle), kontrast skaluje w dół
/// zarówno bieżący odczyt, jak i historyczną średnią kroczącą tego kanału
/// (zbudowaną w tych samych, wcześniej też jednolitych warunkach) o DOKŁADNIE
/// ten sam czynnik — więc próg mnożnikowy ([`TRANSIENT_RATIO`]/
/// [`BAND_TRANSIENT_RATIO`]) wypada identycznie jak bez kontrastu (neutralne,
/// nie pogarsza wykrywania jednolitych zdarzeń). Ale gdy TYLKO JEDEN kanał
/// jest głośniejszy niż jego (niezmienieni) sąsiedzi, kontrast odejmuje od
/// niego dużo MNIEJ (bo średnia wszystkich kanałów ledwo drgnęła), a jego
/// WŁASNA historyczna średnia (zbudowana z czasów, gdy wszystko było cicho i
/// równo) jest obniżona kontrastem tak samo jak wtedy — efektywnie obniżając
/// próg potrzebny, by TEN kanał onsetował, względem dawnego wymogu
/// przebicia [`TRANSIENT_RATIO`]x WŁASNEJ historii. Innymi słowy: kontrast
/// nigdy nie pogarsza wykrywania jednolitego "tła", a ułatwia wykrycie
/// wyraźnie jednostronnego zdarzenia na tle takiego tła.
const CONTRAST_AMOUNT: f32 = 0.5;

/// Minimalna liczba kanałów KIERUNKOWYCH (bez LFE), od której [`CONTRAST_AMOUNT`]
/// zaczyna działać. Przy zwykłym stereo (2.0) L≈R to NORMALNY sposób zapisu
/// wyśrodkowanego/mono dźwięku (dialog, głos), nie oznaka otoczenia — a przy
/// bardzo cichych sygnałach blisko [`ONSET_MIN_ENERGY`] (stały próg, NIE
/// skalujący się razem z kontrastem) odjęcie połowy energii mogłoby zgasić
/// dźwięk, który bez kontrastu jeszcze by przebił próg (dokładnie scenariusz
/// z testu `detect_onsets_sensitivity_lets_a_too_quiet_jump_cross_the_threshold`).
/// Surround (5.1/7.1, gdzie ta funkcja jest realnie użyteczna) ma zawsze
/// więcej niż 2 kanały kierunkowe, więc to nie ogranicza jej praktycznie.
const MIN_CHANNELS_FOR_CONTRAST: usize = 3;

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

/// Stan JEDNEGO kanału (z kierunkiem) w bieżącej ramce analizy — to samo, co
/// bada [`detect_onsets`] wewnętrznie, ale dla KAŻDEGO kanału, nie tylko tych,
/// które przebiły próg onsetu. Używane m.in. przez logger próbek
/// (`crate::sound::sample_logger`), żeby dało się ręcznie sprawdzić dźwięki,
/// które były słyszalne (`energy`/`scaled_energy` > 0), ale nigdy nie
/// wygenerowały markera na radarze (`is_onset == false`) — patrz doc-komentarz
/// [`detect_onsets`] o tym, dlaczego to się zdarza.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelActivity {
    /// Indeks kanału (w `channel_rms`/`channel_layout`).
    pub channel_index: usize,
    /// Poziom RMS kanału w tej ramce (0.0..=1.0), nieprzeskalowany `sensitivity`.
    pub energy: f32,
    /// `energy * sensitivity`, obcięte do 0.0..=1.0 — to porównujemy z progiem
    /// onsetu (szerokopasmowym; per-pasmowe porównanie, patrz
    /// [`detect_channel_activity`], liczone jest osobno i wpływa tylko na
    /// `is_onset`, nie na tę wartość).
    pub scaled_energy: f32,
    /// Płynny kąt (patrz [`onset_angle_deg`]) — liczony niezależnie od tego,
    /// czy kanał akurat jest onsetem, żeby logger miał kąt nawet dla
    /// "cichych" próbek.
    pub angle_deg: f32,
    /// `true`, gdy ten kanał w tej ramce przebił próg onsetu i realnie
    /// wygenerował [`ChannelOnset`] (a więc i marker na radarze).
    pub is_onset: bool,
}

/// Wykrywa nagłe skoki energii (onsety) na każdym kanale osobno, porównując
/// bieżący RMS ze średnią kroczącą tego kanału (stan przekazywany z zewnątrz
/// w `ema_state`, żeby funkcja zostawała czystą/testowalną bez żywego audio —
/// wywołujący jest odpowiedzialny za trzymanie tego samego `Vec` między
/// kolejnymi wywołaniami). Kanały bez kierunku (LFE) są pomijane.
///
/// `ema_state` jest automatycznie dopasowywany do rozmiaru `channel_rms`,
/// jeśli liczba kanałów strumienia się zmieni (np. przełączenie urządzenia).
///
/// `sensitivity` to ten sam mnożnik co [`crate::sound::devices::RadarConfig::radar_sensitivity`]
/// (suwak "Czułość radaru" w ustawieniach) — bez niego progi [`ONSET_MIN_ENERGY`]
/// i [`TRANSIENT_RATIO`] są sztywne i niezależne od tego suwaka, co w praktyce
/// (patrz log diagnostyczny z prawdziwego audio) oznacza, że przy typowych,
/// niegłośnych nagraniach (RMS rzędu 0.01-0.03) detekcja odpala się raz na
/// starcie i potem prawie nigdy więcej — średnia krocząca "dogania" rosnący
/// poziom głośności szybciej, niż zdąży pojawić się coś, co przebije próg.
/// Podnosząc suwak, użytkownik podnosi też `level * sensitivity` tu, więc
/// dotąd niewystarczające skoki zaczynają przebijać próg — dokładnie tak samo
/// jak już dziś reguluje to, jak duże/daleko od środka są kropki kanałów w
/// [`crate::sound::devices::AudioAnalysis`].
pub fn detect_onsets(
    channel_rms: &[f32],
    channel_band_energies: &[Vec<f32>],
    ema_state: &mut Vec<f32>,
    band_ema_state: &mut Vec<Vec<f32>>,
    sensitivity: f32,
    contrast_enabled: bool,
) -> Vec<ChannelOnset> {
    detect_channel_activity(
        channel_rms,
        channel_band_energies,
        ema_state,
        band_ema_state,
        sensitivity,
        contrast_enabled,
    )
    .into_iter()
    .filter(|a| a.is_onset)
    .map(|a| ChannelOnset {
        channel_index: a.channel_index,
        energy: a.energy,
        angle_deg: a.angle_deg,
    })
    .collect()
}

/// Średnia wartości zwracanych przez `f` po `indices` — `0.0` dla pustej listy.
fn mean_of(indices: &[usize], f: impl Fn(usize) -> f32) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| f(i).max(0.0)).sum::<f32>() / indices.len() as f32
}

/// Jak [`detect_onsets`], ale zwraca stan KAŻDEGO kanału z kierunkiem (patrz
/// [`ChannelActivity`]), nie tylko tych, które przebiły próg onsetu —
/// `detect_onsets` jest teraz cienkim filtrem nad tą funkcją, więc oba
/// warianty zawsze widzą dokładnie te same progi/EMA.
///
/// `channel_band_energies[kanał][pasmo]` to energia w każdej regule
/// kolorowania (patrz [`crate::sound::spectrum::analyze_channel_rule_energies`])
/// — TA SAMA tablica, którą wywołujący i tak już liczy do pokolorowania
/// kropek, więc przekazanie jej tutaj nie kosztuje dodatkowej analizy audio
/// ani buforowania (a więc i dodatkowego opóźnienia). Onset kanału zapala się,
/// gdy PRZEBIJE próg ALBO szerokopasmowo (jak dawniej), ALBO w KTÓRYMKOLWIEK
/// pojedynczym paśmie ponad JEGO WŁASNĄ średnią kroczącą (patrz
/// [`BAND_TRANSIENT_RATIO`]) — dzięki temu cichy, ale charakterystyczny
/// dźwięk w jednym paśmie (np. alarm w wysokich tonach) generuje marker nawet
/// gdy energia szerokopasmowa kanału jest zdominowana przez inne, głośniejsze
/// pasmo (np. bas) i sama w sobie nie przebiłaby progu. `band_ema_state`
/// (jedna średnia krocząca na kanał×pasmo) jest automatycznie dopasowywany
/// do liczby kanałów/pasm, tak samo jak `ema_state`. Pusta
/// `channel_band_energies` (np. w testach, które nie badają tego aspektu)
/// po prostu wyłącza sprawdzanie per-pasmowe — zachowanie identyczne jak
/// przed jego dodaniem.
///
/// `contrast_enabled` włącza [`CONTRAST_AMOUNT`] (patrz doc-komentarz tam) —
/// odpowiednik checkboxa `RadarConfig::direction_contrast_enabled` w
/// ustawieniach. Domyślnie `false` w configu, ale sam parametr tu jest
/// bezwarunkowy, żeby funkcja została w pełni testowalna niezależnie od UI.
pub fn detect_channel_activity(
    channel_rms: &[f32],
    channel_band_energies: &[Vec<f32>],
    ema_state: &mut Vec<f32>,
    band_ema_state: &mut Vec<Vec<f32>>,
    sensitivity: f32,
    contrast_enabled: bool,
) -> Vec<ChannelActivity> {
    if ema_state.len() != channel_rms.len() {
        *ema_state = vec![0.0; channel_rms.len()];
    }
    if band_ema_state.len() != channel_rms.len() {
        *band_ema_state = vec![Vec::new(); channel_rms.len()];
    }

    let layout = channel_layout(channel_rms.len());
    let directional: Vec<usize> = (0..channel_rms.len())
        .filter(|&i| layout.get(i).and_then(|role| role.radar_angle_degrees()).is_some())
        .collect();
    let contrast = if contrast_enabled && directional.len() >= MIN_CHANNELS_FOR_CONTRAST {
        CONTRAST_AMOUNT
    } else {
        0.0
    };

    let broadband_mean = mean_of(&directional, |i| channel_rms[i]);
    let n_bands = channel_band_energies.iter().map(Vec::len).max().unwrap_or(0);
    let band_means: Vec<f32> = (0..n_bands)
        .map(|b| {
            mean_of(&directional, |i| {
                channel_band_energies.get(i).and_then(|v| v.get(b)).copied().unwrap_or(0.0)
            })
        })
        .collect();

    // Średnia krocząca sprzed TEJ ramki — stereo liczy kąt z tego, co właśnie
    // podniosło się ponad nią (patrz `stereo_onset_angle_deg`), a pętla niżej
    // już aktualizuje `ema_state`.
    let prev_scaled = ema_state.to_vec();
    let stereo_angle = (layout.len() == 2 && channel_rms.len() >= 2).then(|| {
        stereo_onset_angle_deg(
            [channel_rms[0], channel_rms[1]],
            [prev_scaled[0], prev_scaled[1]],
            sensitivity,
        )
    });

    let mut activity = Vec::new();

    for (i, &level) in channel_rms.iter().enumerate() {
        let level = level.max(0.0);
        let has_direction = layout
            .get(i)
            .and_then(|role| role.radar_angle_degrees())
            .is_some();

        if !has_direction {
            let scaled = (level * sensitivity).clamp(0.0, 1.0);
            ema_state[i] = ema_state[i] * (1.0 - ONSET_EMA_FACTOR) + scaled * ONSET_EMA_FACTOR;
            continue;
        }

        if band_ema_state[i].len() != n_bands {
            band_ema_state[i] = vec![0.0; n_bands];
        }

        let contrasted_level = (level - contrast * broadband_mean).max(0.0);
        let scaled = (contrasted_level * sensitivity).clamp(0.0, 1.0);
        let prev_ema = ema_state[i];
        let mut is_onset = scaled > ONSET_MIN_ENERGY && scaled > prev_ema * TRANSIENT_RATIO + 0.02;
        ema_state[i] = prev_ema * (1.0 - ONSET_EMA_FACTOR) + scaled * ONSET_EMA_FACTOR;

        if let Some(bands) = channel_band_energies.get(i) {
            for (b, &band_energy) in bands.iter().enumerate() {
                let band_mean = band_means.get(b).copied().unwrap_or(0.0);
                let contrasted_band = (band_energy.max(0.0) - contrast * band_mean).max(0.0);
                let band_scaled = (contrasted_band * sensitivity).clamp(0.0, 1.0);
                let prev_band_ema = band_ema_state[i][b];
                if band_scaled > BAND_ONSET_MIN_ENERGY
                    && band_scaled > prev_band_ema * BAND_TRANSIENT_RATIO + BAND_ONSET_MIN_ENERGY
                {
                    is_onset = true;
                }
                band_ema_state[i][b] =
                    prev_band_ema * (1.0 - ONSET_EMA_FACTOR) + band_scaled * ONSET_EMA_FACTOR;
            }
        }

        activity.push(ChannelActivity {
            // Nieskalowane RMS — `sensitivity` (i ewentualny kontrast) wpływa
            // tylko na to, KIEDY odpala się onset (bramka), nie na to, jaką
            // energię niesie (o skalowaniu do jasności/odległości markera
            // decyduje wywołujący, patrz `sound::devices::update_analysis`).
            channel_index: i,
            energy: level,
            scaled_energy: scaled,
            angle_deg: stereo_angle.unwrap_or_else(|| onset_angle_deg(channel_rms, i)),
            is_onset,
        });
    }

    activity
}

/// Płynny kąt onsetu wykrytego na `channel_index`: średnia ważona energią
/// własnego kąta kanału i kątów jego kątowych sąsiadów (odległość kątowa
/// ŚCIŚLE MNIEJSZA niż [`NEIGHBOR_MAX_ANGLE_DIFF_DEG`]`°`, NIE sąsiedztwo w
/// tablicy `channel_layout` — w układzie 2.0 jedynym sąsiadem indeksowym L
/// jest R, czyli kanał dokładnie naprzeciwko, więc głośny R przeciągałby kąt
/// L na swoją stronę, gdyby liczyć po indeksie). Dzięki temu głośny, sąsiedni
/// kanał płynnie przesuwa kąt w swoją stronę, ale kanał po przeciwnej stronie
/// tarczy nigdy nie wpływa na wynik.
///
/// Próg jest CELOWO ostry (`<`, nie `<=`): `FrontLeft`/`FrontRight` leżą
/// dokładnie [`NEIGHBOR_MAX_ANGLE_DIFF_DEG`]`°` (90°) od siebie — to DOKŁADNIE
/// granica progu. Przy `<=` (poprzednia wersja) ta para wzajemnie się
/// uśredniała: dla zwykłego stereo o podobnej energii L i R oba onsety
/// zbiegały do niemal identycznego kąta blisko `0°`, więc dwa niezależne
/// wskaźniki kierunku (patrz [`detect_onsets`]) wizualnie nakładały się na
/// siebie w jednym punkcie — dokładnie ten problem, który cała ta funkcja
/// miała rozwiązać. Znalezione w logu diagnostycznym: `L:3 R:3` dawało
/// `angle=-0.2°` dla OBU kanałów naraz, zamiast dwóch wyraźnie odrębnych
/// kątów. Przy ostrym `<` FL/FR (i inne pary dokładnie 90° od siebie: FL/RL,
/// FR/RR, C/SL, C/SR) już się nie zlewają.
pub fn onset_angle_deg(channel_rms: &[f32], channel_index: usize) -> f32 {
    let layout = channel_layout(channel_rms.len());
    let Some(own_role) = layout.get(channel_index).copied() else {
        return 0.0;
    };
    let Some(own_angle) = own_role.radar_angle_degrees() else {
        return 0.0;
    };

    if layout.len() == 2 && channel_rms.len() >= 2 {
        return stereo_onset_angle_deg([channel_rms[0], channel_rms[1]], [0.0, 0.0], 1.0);
    }

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut total_energy = 0.0f32;

    for (i, role) in layout.iter().enumerate() {
        let Some(angle_deg) = role.radar_angle_degrees() else {
            continue;
        };
        if i != channel_index && wrap_angle_deg(angle_deg - own_angle).abs() >= NEIGHBOR_MAX_ANGLE_DIFF_DEG {
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

/// Kąt onsetu dla stereo (2.0): balans L/R zamiast sztywnego kąta głośnika.
///
/// Liczy się z tego, co właśnie PODNIOSŁO się ponad średnią kroczącą
/// (`prev_scaled`), a nie z surowych poziomów — inaczej stałe tło grające po
/// jednej stronie (muzyka, ambient) przeciągałoby każdy krok w swoją stronę.
/// Gdy żaden kanał nie urósł, wraca do balansu surowych poziomów.
///
/// `pan = (P - L) / (P + L)` mieści się w `-1..=1` i jest mapowany na kąty
/// głośników FL/FR (`-45°`/`+45°`): dźwięk tylko z lewej ląduje na lewym
/// głośniku, równy poziom daje `0°`, a pośrednie balanse dają kąty pośrednie.
/// Wcześniej każdy onset stereo lądował dokładnie na głośniku, niezależnie od
/// tego, jak bardzo jeden kanał przeważał.
fn stereo_onset_angle_deg(levels: [f32; 2], prev_scaled: [f32; 2], sensitivity: f32) -> f32 {
    let scaled = |i: usize| (levels[i].max(0.0) * sensitivity).clamp(0.0, 1.0);
    let rise = |i: usize| (scaled(i) - prev_scaled[i]).max(0.0);

    let (mut left, mut right) = (rise(0), rise(1));
    if left + right < f32::EPSILON {
        left = levels[0].max(0.0);
        right = levels[1].max(0.0);
    }
    let total = left + right;
    if total < f32::EPSILON {
        return 0.0;
    }

    let speaker_angle = ChannelRole::FrontRight.radar_angle_degrees().unwrap_or(45.0);
    (right - left) / total * speaker_angle
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

    // ---------- analyze_channels (RMS per-channel) ----------

    #[test]
    fn analyze_channels_stereo_exact_rms() {
        // L: stała amplituda 1.0, R: stała amplituda 0.5
        let samples = [1.0, 0.5, -1.0, -0.5, 1.0, 0.5];
        let stats = analyze_channels(&samples, 2);
        assert_close(stats.rms[0], 1.0, "rms L");
        assert_close(stats.rms[1], 0.5, "rms R");
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

    // ---------- layout_name ----------

    #[test]
    fn layout_name_matches_known_configurations() {
        assert_eq!(layout_name(1), "1.0");
        assert_eq!(layout_name(2), "2.0");
        assert_eq!(layout_name(6), "5.1");
        assert_eq!(layout_name(8), "7.1");
        assert_eq!(layout_name(4), "4ch");
    }

    // ---------- detect_onsets / onset_angle_deg ----------

    #[test]
    fn opposite_channels_with_similar_energy_yield_two_separate_onsets_not_one_cancelled() {
        // Dokładnie scenariusz "silnik z lewej + wybuch z prawej": SL (-90°) i
        // SR (90°) leżą dokładnie naprzeciw siebie i mają podobną energię, więc
        // suma wektorowa dałaby prawie zerowy wektor. detect_onsets
        // powinien zamiast tego zgłosić DWA niezależne zdarzenia, każde na
        // swoim kanale. [FL, FR, C, LFE, RL, RR, SL, SR]
        let mut ema = vec![0.0; 8];
        let mut band_ema = Vec::new();
        let onsets = detect_onsets(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3], &[], &mut ema, &mut band_ema, 1.0, false);
        let left = onsets.iter().find(|o| o.channel_index == 6).unwrap();
        let right = onsets.iter().find(|o| o.channel_index == 7).unwrap();
        assert_close(left.angle_deg, -90.0, "onset lewego kanału (SL)");
        assert_close(right.angle_deg, 90.0, "onset prawego kanału (SR)");
    }

    #[test]
    fn detect_onsets_ignores_steady_energy_without_a_sudden_jump() {
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        // Kilka ramek o stałym poziomie — po "rozgrzaniu" średniej kroczącej
        // dalsze ramki o tym samym poziomie nie powinny być onsetami.
        for _ in 0..50 {
            detect_onsets(&[0.2, 0.2], &[], &mut ema, &mut band_ema, 1.0, false);
        }
        let onsets = detect_onsets(&[0.2, 0.2], &[], &mut ema, &mut band_ema, 1.0, false);
        assert!(onsets.is_empty(), "stały poziom nie powinien generować onsetów");
    }

    #[test]
    fn detect_onsets_silence_yields_nothing() {
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        assert!(detect_onsets(&[0.0, 0.0], &[], &mut ema, &mut band_ema, 1.0, false).is_empty());
    }

    #[test]
    fn detect_onsets_skips_lfe_channel() {
        // [FL, FR, C, LFE, RL, RR]
        let mut ema = vec![0.0; 6];
        let mut band_ema = Vec::new();
        let onsets = detect_onsets(&[0.0, 0.0, 0.0, 0.5, 0.0, 0.0], &[], &mut ema, &mut band_ema, 1.0, false);
        assert!(
            onsets.iter().all(|o| o.channel_index != 3),
            "LFE nie niesie kierunku, nie powinien generować onsetu"
        );
    }

    #[test]
    fn detect_onsets_resizes_ema_state_for_new_channel_count() {
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        let _ = detect_onsets(&[0.1, 0.1], &[], &mut ema, &mut band_ema, 1.0, false);
        assert_eq!(ema.len(), 2);
        // Zmiana urządzenia -> inna liczba kanałów; nie powinno panikować.
        let _ = detect_onsets(&[0.0; 6], &[], &mut ema, &mut band_ema, 1.0, false);
        assert_eq!(ema.len(), 6);
    }

    #[test]
    fn detect_onsets_sensitivity_lets_a_too_quiet_jump_cross_the_threshold() {
        // Regresja: przy typowych, niegłośnych nagraniach (RMS ~0.01-0.03,
        // patrz log diagnostyczny z prawdziwego audio) domyślna czułość
        // (1.0) w ogóle nie widziała skoku głośności — dopiero podniesienie
        // suwaka "Czułość radaru" (ten sam `sensitivity` co
        // `RadarConfig::radar_sensitivity`) miało pozwolić dotąd
        // niewystarczającym skokom przebić próg.
        let jump = [0.012, 0.012]; // wyraźnie poniżej ONSET_MIN_ENERGY (0.02)
        let mut ema_low = vec![0.0, 0.0];
        let mut band_ema_low = Vec::new();
        assert!(
            detect_onsets(&jump, &[], &mut ema_low, &mut band_ema_low, 1.0, false).is_empty(),
            "przy domyślnej czułości ten poziom nie powinien jeszcze przebić progu"
        );

        let mut ema_high = vec![0.0, 0.0];
        let mut band_ema_high = Vec::new();
        let onsets = detect_onsets(&jump, &[], &mut ema_high, &mut band_ema_high, 3.0, false);
        assert_eq!(
            onsets.len(),
            2,
            "podniesiona czułość powinna pozwolić temu samemu skokowi przebić próg"
        );
    }

    #[test]
    fn band_excursion_triggers_onset_even_when_broadband_barely_moves() {
        // Kanał zdominowany przez stały, głośny bas (pasmo A) — energia
        // szerokopasmowa (channel_rms) w ogóle się nie zmienia między
        // "rozgrzaniem" a ramką testową, więc szerokopasmowy próg onsetu
        // NIGDY by tu nie zadziałał. Ale ciche pasmo B (np. wysoki alarm)
        // nagle podskakuje daleko ponad SWOJE WŁASNE, dotąd znikome tło —
        // to powinno samo z siebie wygenerować onset.
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        let rms = [0.3, 0.3];
        let steady_bands = vec![vec![0.3, 0.001], vec![0.3, 0.001]];
        for _ in 0..50 {
            detect_onsets(&rms, &steady_bands, &mut ema, &mut band_ema, 1.0, false);
        }
        // Potwierdzenie, że po rozgrzaniu ten sam, stały poziom NIE onsetuje
        // (ani szerokopasmowo, ani per-pasmowo) — punkt odniesienia.
        assert!(
            detect_onsets(&rms, &steady_bands, &mut ema, &mut band_ema, 1.0, false).is_empty(),
            "stały poziom (również per pasmo) nie powinien generować onsetów"
        );

        // Pasmo B kanału 0 skacze z 0.001 do 0.02 — RMS szerokopasmowy kanału
        // 0 zostaje BEZ ZMIAN (0.3), więc to wyłącznie pasmowy wyskok.
        let jump_bands = vec![vec![0.3, 0.02], vec![0.3, 0.001]];
        let onsets = detect_onsets(&rms, &jump_bands, &mut ema, &mut band_ema, 1.0, false);
        assert!(
            onsets.iter().any(|o| o.channel_index == 0),
            "skok w jednym cichym paśmie powinien wygenerować onset, mimo że RMS szerokopasmowy się nie zmienił"
        );
        assert!(
            onsets.iter().all(|o| o.channel_index != 1),
            "kanał 1 (bez zmiany w żadnym paśmie) nie powinien onsetować"
        );
    }

    #[test]
    fn empty_band_energies_behaves_exactly_like_before_band_detection_existed() {
        // Pusta `channel_band_energies` (np. wywołujący nie ma jeszcze danych
        // per-pasmowych) musi dawać DOKŁADNIE to samo zachowanie co przed
        // dodaniem sprawdzania per-pasmowego — czysto szerokopasmowe.
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        let onsets = detect_onsets(&[0.012, 0.012], &[], &mut ema, &mut band_ema, 1.0, false);
        assert!(onsets.is_empty(), "poniżej ONSET_MIN_ENERGY nadal nie powinno onsetować");
    }

    // ---------- kontrast międzykanałowy (contrast_enabled) ----------

    #[test]
    fn contrast_lets_a_moderate_single_channel_rise_onset_against_a_stable_quiet_crowd() {
        // 7.1 (8 kanałów), wszystkie 7 kierunkowych "rozgrzane" na stałym,
        // cichym poziomie 0.1 — dwa NIEZALEŻNE stany (z kontrastem i bez),
        // każdy rozgrzewany konsekwentnie z tym samym trybem, żeby historyczna
        // średnia krocząca była uczciwie porównywalna.
        let quiet = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1]; // LFE (idx 3) nie ma kierunku

        let mut ema_without = vec![0.0; 8];
        let mut band_ema_without = Vec::new();
        let mut ema_with = vec![0.0; 8];
        let mut band_ema_with = Vec::new();
        for _ in 0..80 {
            detect_onsets(&quiet, &[], &mut ema_without, &mut band_ema_without, 1.0, false);
            detect_onsets(&quiet, &[], &mut ema_with, &mut band_ema_with, 1.0, true);
        }

        // FrontLeft (idx 0) rośnie do 0.22 — to tylko 2.2x własnej historii
        // (0.1), czyli PONIŻEJ szerokopasmowego TRANSIENT_RATIO=2.5x, więc bez
        // kontrastu ten wzrost NIE powinien onsetować. Pozostałe kanały
        // zostają bez zmian (0.1) — to jest wyraźnie JEDNOSTRONNE zdarzenie,
        // nie jednolity wzrost tła.
        let mut bump = quiet;
        bump[0] = 0.22;

        let onsets_without = detect_onsets(&bump, &[], &mut ema_without, &mut band_ema_without, 1.0, false);
        assert!(
            onsets_without.iter().all(|o| o.channel_index != 0),
            "bez kontrastu wzrost 2.2x własnej historii nie powinien jeszcze przebić progu {TRANSIENT_RATIO}x"
        );

        let onsets_with = detect_onsets(&bump, &[], &mut ema_with, &mut band_ema_with, 1.0, true);
        assert!(
            onsets_with.iter().any(|o| o.channel_index == 0),
            "z kontrastem to samo zdarzenie powinno przebić próg, bo sąsiedzi zostali bez zmian"
        );
    }

    #[test]
    fn contrast_does_not_change_outcome_when_all_directional_channels_rise_together() {
        // Kontrast NIE powinien pogarszać wykrywania jednolitego zdarzenia —
        // gdy WSZYSTKIE kanały kierunkowe rosną RÓWNO (naprawdę niekierunkowy
        // dźwięk), matematycznie skaluje w dół i historię, i bieżący odczyt o
        // ten sam czynnik, więc wynik progu mnożnikowego jest identyczny.
        let uniform_loud = [0.4; 8];
        let mut ema_a = vec![0.0; 8];
        let mut band_ema_a = Vec::new();
        let onsets_without = detect_onsets(&uniform_loud, &[], &mut ema_a, &mut band_ema_a, 1.0, false);

        let mut ema_b = vec![0.0; 8];
        let mut band_ema_b = Vec::new();
        let onsets_with = detect_onsets(&uniform_loud, &[], &mut ema_b, &mut band_ema_b, 1.0, true);

        let mut channels_without: Vec<usize> = onsets_without.iter().map(|o| o.channel_index).collect();
        let mut channels_with: Vec<usize> = onsets_with.iter().map(|o| o.channel_index).collect();
        channels_without.sort_unstable();
        channels_with.sort_unstable();
        assert_eq!(
            channels_without, channels_with,
            "jednolity wzrost na wszystkich kanałach powinien dać te same onsety z kontrastem i bez"
        );
    }

    #[test]
    fn contrast_is_a_no_op_for_plain_stereo() {
        // 2.0 ma tylko 2 kanały kierunkowe — poniżej MIN_CHANNELS_FOR_CONTRAST,
        // więc `contrast_enabled=true` nie powinno nic zmienić względem
        // `false` (patrz doc-komentarz `MIN_CHANNELS_FOR_CONTRAST`: przy
        // stereo L≈R to normalny zapis wyśrodkowanego dźwięku, nie otoczenie).
        let jump = [0.012, 0.012];
        let mut ema_without = vec![0.0, 0.0];
        let mut band_ema_without = Vec::new();
        let without = detect_onsets(&jump, &[], &mut ema_without, &mut band_ema_without, 3.0, false);

        let mut ema_with = vec![0.0, 0.0];
        let mut band_ema_with = Vec::new();
        let with = detect_onsets(&jump, &[], &mut ema_with, &mut band_ema_with, 3.0, true);

        assert_eq!(without.len(), with.len(), "stereo powinno ignorować contrast_enabled");
        assert_eq!(ema_without, ema_with, "EMA stereo nie powinno się różnić z/bez kontrastu");
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

    #[test]
    fn onset_angle_deg_stereo_follows_left_right_balance() {
        // Zwykłe stereo (2.0): kąt zależy od tego, który kanał przeważa, a nie
        // jest przyklejony do głośnika. Pełna przewaga L -> FL (-45°), pełna
        // przewaga P -> FR (+45°), równy poziom -> środek (0°) dla obu kanałów.
        assert_close(onset_angle_deg(&[0.3, 0.0], 0), -45.0, "tylko lewy -> FL");
        assert_close(onset_angle_deg(&[0.0, 0.3], 1), 45.0, "tylko prawy -> FR");
        assert_close(onset_angle_deg(&[0.2, 0.2], 0), 0.0, "równy poziom, kanał L -> środek");
        assert_close(onset_angle_deg(&[0.2, 0.2], 1), 0.0, "równy poziom, kanał R -> środek");
    }

    #[test]
    fn onset_angle_deg_stereo_moves_with_partial_balance() {
        // Przewaga 3:1 w stronę lewej: pan = (0.1 - 0.3) / 0.4 = -0.5 -> -22.5°.
        assert_close(onset_angle_deg(&[0.3, 0.1], 0), -22.5, "przewaga L 3:1");
    }

    #[test]
    fn stereo_onset_angle_points_at_the_side_that_rose_not_the_steady_background() {
        // Prawy kanał to stałe, głośne tło (np. muzyka); lewy nagle skacze.
        // Kąt ma wskazywać LEWĄ stronę, mimo że surowo prawy kanał przeważa.
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        for _ in 0..50 {
            detect_channel_activity(&[0.1, 0.4], &[], &mut ema, &mut band_ema, 1.0, false);
        }
        let activity = detect_channel_activity(&[0.3, 0.4], &[], &mut ema, &mut band_ema, 1.0, false);
        let left = activity.iter().find(|a| a.channel_index == 0).unwrap();
        assert!(left.is_onset, "skok lewego kanału powinien być onsetem");
        assert!(
            left.angle_deg < -40.0,
            "krok z lewej ma wskazywać lewo, a nie tło z prawej: {}",
            left.angle_deg
        );
    }

    #[test]
    fn stereo_onset_angle_is_centre_when_both_sides_rise_equally() {
        let mut ema = vec![0.0, 0.0];
        let mut band_ema = Vec::new();
        for _ in 0..50 {
            detect_channel_activity(&[0.1, 0.1], &[], &mut ema, &mut band_ema, 1.0, false);
        }
        let activity = detect_channel_activity(&[0.4, 0.4], &[], &mut ema, &mut band_ema, 1.0, false);
        assert_eq!(activity.len(), 2);
        for a in &activity {
            assert!(a.is_onset);
            assert_close(a.angle_deg, 0.0, "równy wzrost obu stron = źródło na środku");
        }
    }
}
