//! Log radaru przy odsłuchu pliku: jeden wiersz na zdarzenie warstwy. Kolumny są takie, jakie pokazuje
//! wybrany tryb wizualizacji (patrz `ui::radar_egui::view_snapshot`), wartości są liczbowe, a każdy wiersz
//! ma też opis dźwięku słowami i opis tego, co radar pokazał w tej chwili. Plik:
//! `config/odsluch/radar_log_<nazwa>_<nr>.csv`, z blokiem komentarzy `#` na początku, który opisuje kody
//! i każdą kolumnę.
//!
//! Zapis idzie z wątku UI, bo zdarzeń jest kilka na sekundę. Nie dotyka callbacku audio.

use crate::sound::devices::{QueuedLayerHit, RadarConfig, VisualizationMode};
use crate::sound::direction::{channel_layout, ChannelRole};
use crate::sound::layers::{Class, LayerHit};
use crate::ui::radar_egui::{self, layer_code, RadarState, ViewSnapshot};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const FILE_PREFIX: &str = "radar_log_";
const FILE_SUFFIX: &str = ".csv";
/// Podfolder w katalogu configu, w którym leżą logi z odsłuchu.
const LOG_SUBDIR: &str = "odsluch";
/// Poziom podawany dla ciszy w kanale (dBFS), taki sam jak w analizie warstw.
const SILENCE_DBFS: f32 = -180.0;

/// Kolumny zawsze obecne, na początku wiersza.
const BASE_COLUMNS: [&str; 3] = ["czas_s", "nr", "tryb"];
/// Kolumny zdarzenia, tylko dla trybów z warstwami (3+3 i 2+2). Poziomy L i R są z ramki szczytu zdarzenia.
const LAYER_COLUMNS: [&str; 8] = [
    "klasa",
    "poziom_dbfs",
    "jasnosc_db",
    "odstep_s",
    "warstwa",
    "pan",
    "kanal_L_dbfs",
    "kanal_R_dbfs",
];
/// Kolumny tekstowe, zawsze na końcu wiersza.
const TEXT_COLUMNS: [&str; 2] = ["dzwiek_tekst", "widok_tekst"];

/// Katalog logów radaru: `config/odsluch`.
pub fn log_dir() -> PathBuf {
    crate::config_store::config_dir().join(LOG_SUBDIR)
}

/// Nazwa pliku audio bez rozszerzenia, bezpieczna w nazwie pliku logu.
pub fn file_stem_for_log(audio: &Path) -> String {
    let stem = audio.file_stem().and_then(|s| s.to_str()).unwrap_or("nagranie");
    let clean: String = stem
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    if clean.is_empty() {
        "nagranie".to_string()
    } else {
        clean
    }
}

/// Następny wolny plik `radar_log_<nazwa>_<nr>.csv` w katalogu. Numer rośnie osobno dla każdego nagrania.
pub fn next_log_path(dir: &Path, stem: &str) -> PathBuf {
    let prefix = format!("{FILE_PREFIX}{stem}_");
    let mut max_n = 0u32;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let number = name
                .to_str()
                .and_then(|s| s.strip_prefix(&prefix))
                .and_then(|s| s.strip_suffix(FILE_SUFFIX))
                .and_then(|s| s.parse::<u32>().ok());
            if let Some(n) = number {
                max_n = max_n.max(n);
            }
        }
    }
    dir.join(format!("{prefix}{:03}{FILE_SUFFIX}", max_n + 1))
}

/// Krótka nazwa trybu, do nagłówka pliku.
pub fn mode_key(mode: VisualizationMode) -> &'static str {
    match mode {
        VisualizationMode::OnlyRadar => "tylko_radar",
        VisualizationMode::OnlyBars => "tylko_slupki_LR",
        VisualizationMode::Both => "radar_i_slupki",
        VisualizationMode::LayerBars => "warstwy_3p3",
        VisualizationMode::LayerBars22 => "warstwy_2p2",
        VisualizationMode::None => "nic",
    }
}

/// Kod trybu w kolumnie `tryb`: 0 nic, 1 tylko radar, 2 tylko słupki L/R, 3 radar i słupki,
/// 4 warstwy 3+3, 5 warstwy 2+2.
pub fn mode_code(mode: VisualizationMode) -> u8 {
    match mode {
        VisualizationMode::None => 0,
        VisualizationMode::OnlyRadar => 1,
        VisualizationMode::OnlyBars => 2,
        VisualizationMode::Both => 3,
        VisualizationMode::LayerBars => 4,
        VisualizationMode::LayerBars22 => 5,
    }
}

/// Kod klasy w kolumnie `klasa`: 0 brak, 1 A, 2 B, 3 C, 4 D.
fn class_code(class: Option<Class>) -> u8 {
    match class {
        None => 0,
        Some(Class::A) => 1,
        Some(Class::B) => 2,
        Some(Class::C) => 3,
        Some(Class::D) => 4,
    }
}

/// Krótka nazwa roli kanału do nazwy kolumny: FL, FR, C, LFE, RL, RR, SL, SR.
fn role_short(role: ChannelRole) -> &'static str {
    match role {
        ChannelRole::FrontLeft => "FL",
        ChannelRole::FrontRight => "FR",
        ChannelRole::Center => "C",
        ChannelRole::Lfe => "LFE",
        ChannelRole::RearLeft => "RL",
        ChannelRole::RearRight => "RR",
        ChannelRole::SideLeft => "SL",
        ChannelRole::SideRight => "SR",
    }
}

/// Kolumny poziomów kanałów w kolejności układu, np. `kanal_FL_dbfs`.
pub fn channel_names(channels: usize) -> Vec<String> {
    channel_layout(channels)
        .iter()
        .map(|role| format!("kanal_{}_dbfs", role_short(*role)))
        .collect()
}

/// Opis kolumny, który idzie do nagłówka pliku.
pub fn describe(column: &str) -> String {
    if let Some(name) = column.strip_prefix("kanal_").and_then(|s| s.strip_suffix("_dbfs")) {
        return match name {
            "L" | "R" => "poziom kanału w dBFS w chwili szczytu zdarzenia, z tej samej ramki".to_string(),
            role => format!("poziom kanału {role} w dBFS, z chwili odczytu klatki"),
        };
    }
    let text = match column {
        "czas_s" => "sekundy od startu odtwarzania pliku, w chwili zdarzenia warstwy",
        "nr" => "numer zdarzenia w tym logu, od 1",
        "tryb" => "kod trybu wyświetlania (legenda w nagłówku)",
        "klasa" => "klasa zdarzenia: 0 brak, 1 A (głośny), 2 B (średni), 3 C (cichy, ciemny), 4 D (cichy, jasny)",
        "poziom_dbfs" => "szczyt zdarzenia w dBFS, z obu kanałów",
        "jasnosc_db" => "jasność zdarzenia w dB; pusta, gdy zdarzenie jest ciche",
        "odstep_s" => "odstęp od poprzedniego zdarzenia w sekundach; pusty dla pierwszego",
        "warstwa" => "warstwa zdarzenia: 1 strzały, 2 kroki, 3 reszta",
        "pan" => "balans L/R: -1 pełne lewo, 0 środek, +1 pełne prawo",
        "radar_wskazniki" => "liczba widocznych wskaźników kierunku",
        "radar_kierunek_deg" => "kąt najnowszego wskaźnika kierunku w stopniach; pusty, gdy brak",
        "radar_intensywnosc_proc" => "siła najnowszego wskaźnika kierunku, 0-100 %",
        "radar_kolor_R" => "składowa czerwona koloru najnowszego wskaźnika, 0-255",
        "radar_kolor_G" => "składowa zielona koloru najnowszego wskaźnika, 0-255",
        "radar_kolor_B" => "składowa niebieska koloru najnowszego wskaźnika, 0-255",
        "bars_L_proc" => "wysokość lewego słupka L/R, 0-100 %",
        "bars_R_proc" => "wysokość prawego słupka L/R, 0-100 %",
        "bars_obrys" => "strona z czerwonym obrysem przewagi: 0 brak, 1 L, 2 R",
        "moc_kroki_L_proc" => "moc łuku kroków po lewej, 0-100 %; 0 gdy łuk schowany",
        "moc_kroki_R_proc" => "moc łuku kroków po prawej, 0-100 %; 0 gdy łuk schowany",
        "moc_strzaly_L_proc" => "moc łuku strzałów i wybuchów po lewej, 0-100 %; 0 gdy schowany",
        "moc_strzaly_R_proc" => "moc łuku strzałów i wybuchów po prawej, 0-100 %; 0 gdy schowany",
        "moc_wszystko_L_proc" => "moc zewnętrznego łuku (wszystko) po lewej, 0-100 %; 0 gdy schowany",
        "moc_wszystko_R_proc" => "moc zewnętrznego łuku (wszystko) po prawej, 0-100 %; 0 gdy schowany",
        "podswietlona" => "warstwa podświetlonej pary w tej chwili: 1 strzały, 2 kroki, 3 reszta",
        "obwodka" => "strona z czerwoną obwódką na podświetlonej parze: 0 brak, 1 L, 2 R",
        "moc_wewnetrzny_L_proc" => "moc wewnętrznego łuku (strzały i kroki) po lewej, 0-100 %",
        "moc_wewnetrzny_R_proc" => "moc wewnętrznego łuku (strzały i kroki) po prawej, 0-100 %",
        "zrodlo_wewnetrzny_L" => "warstwa dająca wewnętrzny łuk po lewej: 1 strzały, 2 kroki",
        "zrodlo_wewnetrzny_R" => "warstwa dająca wewnętrzny łuk po prawej: 1 strzały, 2 kroki",
        "moc_zewnetrzny_L_proc" => "moc zewnętrznego łuku (wszystko) po lewej, 0-100 %",
        "moc_zewnetrzny_R_proc" => "moc zewnętrznego łuku (wszystko) po prawej, 0-100 %",
        "zrodlo_zewnetrzny_L" => "warstwa dająca zewnętrzny łuk po lewej: 1 strzały, 2 kroki, 3 reszta",
        "zrodlo_zewnetrzny_R" => "warstwa dająca zewnętrzny łuk po prawej: 1 strzały, 2 kroki, 3 reszta",
        "poswiata_strona" => "strona, po której świeci poświata 2+2: 0 brak, 1 L, 2 R",
        "poswiata_proc" => "siła poświaty 2+2 w procentach; przy mocy poświaty powyżej 1 może przekroczyć 100",
        "przod_proc" => "przód na łukach PRZÓD/TYŁ, 0-100 %; 0 gdy widok przestrzenny jest schowany",
        "tyl_proc" => "tył na łukach PRZÓD/TYŁ, 0-100 %; 0 gdy widok przestrzenny jest schowany",
        "dzwiek_tekst" => "opis dźwięku słowami: rodzaj, klasa, poziom, jasność, balans i odstęp",
        "widok_tekst" => "opis słowami tego, co wizualizacja pokazała w tej chwili",
        _ => "",
    };
    text.to_string()
}

/// Opis balansu słowami: środek, albo strona i procent wychylenia.
fn pan_words(pan: f32) -> String {
    let pct = (pan.abs() * 100.0).round() as i32;
    if pan.abs() < 0.1 {
        "środek".to_string()
    } else if pan < 0.0 {
        format!("z lewej {pct}%")
    } else {
        format!("z prawej {pct}%")
    }
}

/// Opis dźwięku słowami, z tego samego zdarzenia co kolumny liczbowe.
pub fn sound_text(hit: &LayerHit) -> String {
    let kind = match (hit.layer, hit.class) {
        (crate::sound::layers::Layer::Shots, Some(Class::D)) => "granat (krótki odstęp po strzale)",
        (crate::sound::layers::Layer::Shots, _) => "strzał lub wybuch",
        (crate::sound::layers::Layer::Steps, _) => "krok",
        (crate::sound::layers::Layer::Rest, _) => "dźwięk z tła",
    };
    let class = match hit.class {
        Some(Class::A) => "klasa A (głośny)",
        Some(Class::B) => "klasa B (średni)",
        Some(Class::C) => "klasa C (cichy, ciemny)",
        Some(Class::D) => "klasa D (cichy, jasny)",
        None => "bez klasy",
    };
    let brightness = hit
        .brightness_db
        .map(|b| format!("jasność {b:.1} dB"))
        .unwrap_or_else(|| "bez jasności".to_string());
    let gap = hit
        .gap_s
        .map(|g| format!("odstęp {g:.2} s"))
        .unwrap_or_else(|| "pierwsze zdarzenie".to_string());
    format!(
        "{kind}, {class}, {:.1} dBFS, {brightness}, {}, {gap}",
        hit.peak_dbfs,
        pan_words(hit.pan)
    )
}

/// Kolumny liczbowe zdarzenia dla trybów z warstwami, w kolejności `LAYER_COLUMNS`.
fn layer_fields(hit: &LayerHit) -> Vec<String> {
    vec![
        class_code(hit.class).to_string(),
        format!("{:.1}", hit.peak_dbfs),
        hit.brightness_db.map(|b| format!("{b:.1}")).unwrap_or_default(),
        hit.gap_s.map(|g| format!("{g:.3}")).unwrap_or_default(),
        layer_code(hit.layer).to_string(),
        format!("{:+.2}", hit.pan),
        format!("{:.1}", hit.peak_left_dbfs),
        format!("{:.1}", hit.peak_right_dbfs),
    ]
}

/// Poziom dBFS z wartości RMS kanału (liniowej, 0..1). Cisza daje `SILENCE_DBFS`.
fn db_of_rms(rms: f32) -> f32 {
    if rms > 1e-9 {
        20.0 * rms.log10()
    } else {
        SILENCE_DBFS
    }
}

/// Poziomy kanałów z tej klatki, w kolejności układu. Brakujące kanały to cisza.
fn channel_fields(channels: usize, rms: &[f32]) -> Vec<String> {
    (0..channel_layout(channels).len())
        .map(|i| format!("{:.1}", db_of_rms(rms.get(i).copied().unwrap_or(0.0))))
        .collect()
}

/// Pole CSV: w cudzysłowie, gdy zawiera przecinek, cudzysłów albo koniec wiersza.
fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn csv_line<'a>(fields: impl IntoIterator<Item = &'a str>) -> String {
    let mut line = fields.into_iter().map(csv_field).collect::<Vec<_>>().join(",");
    line.push('\n');
    line
}

/// Czy tryb rysuje poziomy kanałów z klatki (radar albo słupki, ale bez warstw). Warstwy mają poziomy
/// L i R z ramki zdarzenia.
fn has_channel_block(cfg: &RadarConfig) -> bool {
    !cfg.visualization_mode.shows_layers()
        && (cfg.visualization_mode.shows_radar() || cfg.visualization_mode.shows_bars())
}

/// Kolumny pliku dla wybranego trybu i widoku przestrzennego, w kolejności zapisu. Tylko to, co
/// wybrany tryb faktycznie pokazuje.
pub fn columns_for(cfg: &RadarConfig, channels: usize) -> Vec<String> {
    let mut columns: Vec<String> = BASE_COLUMNS.iter().map(|c| c.to_string()).collect();
    if cfg.visualization_mode.shows_layers() {
        columns.extend(LAYER_COLUMNS.iter().map(|c| c.to_string()));
    } else if has_channel_block(cfg) {
        columns.extend(channel_names(channels));
    }
    columns.extend(
        radar_egui::view_snapshot(&RadarState::new(), cfg, 0.0)
            .names
            .into_iter()
            .map(String::from),
    );
    columns.extend(TEXT_COLUMNS.iter().map(|c| c.to_string()));
    columns
}

fn header_block(audio_name: &str, cfg: &RadarConfig, started_ms: u128, channels: usize, columns: &[String]) -> String {
    let mut out = String::new();
    let mut line = |text: String| {
        out.push_str("# ");
        out.push_str(&text);
        out.push('\n');
    };
    line("Hear Radar: log radaru".to_string());
    line(format!("plik audio: {audio_name}"));
    line(format!("start (unix ms): {started_ms}"));
    line(format!("tryb wyświetlania: {} ({})", mode_key(cfg.visualization_mode), cfg.visualization_mode.label()));
    line(format!("układ głośników: {}, kanałów w pliku: {channels}", cfg.speaker_setup.label()));
    line(format!("czułość radaru: {:.2}, czas zaniku: {:.2} s", cfg.radar_sensitivity, cfg.decay_secs));
    line(format!(
        "wzmocnienia: kroki {:.2}, strzały {:.2}, zewnętrzny łuk {:.2}; wewnętrzny promień {:.2}, odstęp {:.2}",
        cfg.layer_steps_gain, cfg.layer_shots_gain, cfg.layer_rest_gain, cfg.layer_arc_inner, cfg.layer_arc_spacing
    ));
    line(format!("moc poświaty 2+2: {:.2}", cfg.layer_side_glow));
    line("jeden wiersz na zdarzenie warstwy; czas_s liczony od startu odtwarzania".to_string());
    line("kody: tryb 0 nic, 1 tylko radar, 2 tylko słupki L/R, 3 radar i słupki, 4 warstwy 3+3, 5 warstwy 2+2".to_string());
    line("kody: klasa 0 brak, 1 A, 2 B, 3 C, 4 D; warstwa i źródło 1 strzały, 2 kroki, 3 reszta".to_string());
    line("kody: strona 0 brak, 1 L (lewa), 2 R (prawa); moce 0-100 %, 0 gdy łuk schowany".to_string());
    line("kolumny:".to_string());
    for column in columns {
        line(format!("{column}: {}", describe(column)));
    }
    out
}

/// Otwarty log radaru dla jednego odtwarzania pliku.
pub struct RadarLogSession {
    writer: BufWriter<std::fs::File>,
    path: PathBuf,
    /// Czas strumienia analizy w chwili startu odtwarzania. Od niego liczy się `czas_s`.
    start_stream_s: f64,
    mode: VisualizationMode,
    spatial: bool,
    channels: usize,
    layer_columns: bool,
    channel_columns: bool,
    rows: u64,
}

impl RadarLogSession {
    /// Tworzy nowy plik logu w `dir` i zapisuje blok opisu oraz nagłówek kolumn. `channels` to liczba
    /// kanałów pliku audio, od niej zależą nazwy kolumn poziomów.
    pub fn open(dir: &Path, audio: &Path, cfg: &RadarConfig, start_stream_s: f64, channels: usize) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = next_log_path(dir, &file_stem_for_log(audio));
        let file = OpenOptions::new().create_new(true).write(true).open(&path)?;
        let columns = columns_for(cfg, channels);
        let started_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let audio_name = audio.file_name().and_then(|s| s.to_str()).unwrap_or("nieznany");
        let mut writer = BufWriter::new(file);
        writer.write_all(header_block(audio_name, cfg, started_ms, channels, &columns).as_bytes())?;
        writer.write_all(csv_line(columns.iter().map(String::as_str)).as_bytes())?;
        writer.flush()?;
        Ok(Self {
            writer,
            path,
            start_stream_s,
            mode: cfg.visualization_mode,
            spatial: cfg.spatial_audio_view,
            channels,
            layer_columns: cfg.visualization_mode.shows_layers(),
            channel_columns: has_channel_block(cfg),
            rows: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Czy kolumny logu nadal pasują do ustawień (tryb i widok przestrzenny). Inaczej log trzeba zamknąć.
    pub fn matches(&self, cfg: &RadarConfig) -> bool {
        cfg.visualization_mode == self.mode && cfg.spatial_audio_view == self.spatial
    }

    /// Zapisuje zdarzenia warstw, które zaszły po starcie logu. `view` to to, co widać w tej samej klatce,
    /// a `channel_rms` to poziomy kanałów z tej klatki (dla trybów bez warstw). Zwraca liczbę wierszy.
    pub fn log_hits(&mut self, hits: &[QueuedLayerHit], view: &ViewSnapshot, channel_rms: &[f32]) -> io::Result<usize> {
        let mut written = 0;
        for queued in hits {
            let hit = &queued.hit;
            let time_s = hit.onset_s - self.start_stream_s;
            if time_s < 0.0 {
                continue;
            }
            self.rows += 1;
            let mut fields: Vec<String> = vec![
                format!("{time_s:.3}"),
                self.rows.to_string(),
                mode_code(self.mode).to_string(),
            ];
            if self.layer_columns {
                fields.extend(layer_fields(hit));
            } else if self.channel_columns {
                fields.extend(channel_fields(self.channels, channel_rms));
            }
            fields.extend(view.values.iter().cloned());
            fields.push(sound_text(hit));
            fields.push(view.text.clone());
            self.writer.write_all(csv_line(fields.iter().map(String::as_str)).as_bytes())?;
            written += 1;
        }
        if written > 0 {
            self.writer.flush()?;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sound::devices::{RadarConfig, VisualizationMode as M};
    use crate::sound::layers::Layer;
    use crate::ui::radar_egui::view_snapshot;
    use std::time::Instant;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hearradar_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn cfg_for(mode: M) -> RadarConfig {
        RadarConfig {
            visualization_mode: mode,
            ..RadarConfig::default()
        }
    }

    fn has(columns: &[String], name: &str) -> bool {
        columns.iter().any(|c| c == name)
    }

    fn shot(onset_s: f64, peak_dbfs: f32, pan: f32) -> QueuedLayerHit {
        QueuedLayerHit {
            captured_at: Instant::now(),
            hit: LayerHit {
                layer: Layer::Shots,
                class: Some(Class::A),
                pan,
                peak_dbfs,
                peak_left_dbfs: peak_dbfs,
                peak_right_dbfs: peak_dbfs,
                brightness_db: Some(-3.0),
                gap_s: None,
                onset_s,
            },
        }
    }

    #[test]
    fn log_files_are_numbered_per_recording() {
        let dir = temp_dir("numbering");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(next_log_path(&dir, "nagranie").file_name().unwrap(), "radar_log_nagranie_001.csv");
        std::fs::write(dir.join("radar_log_nagranie_001.csv"), "").unwrap();
        std::fs::write(dir.join("radar_log_nagranie_007.csv"), "").unwrap();
        std::fs::write(dir.join("radar_log_inne_009.csv"), "").unwrap();
        assert_eq!(next_log_path(&dir, "nagranie").file_name().unwrap(), "radar_log_nagranie_008.csv");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn logs_go_to_the_odsluch_subfolder_of_config() {
        let dir = log_dir();
        assert!(dir.ends_with(Path::new("config").join("odsluch")), "{}", dir.display());
    }

    #[test]
    fn audio_name_is_safe_for_a_file_name() {
        assert_eq!(file_stem_for_log(Path::new("C:/x/moj plik (1).wav")), "moj_plik__1_");
        assert_eq!(file_stem_for_log(Path::new("ścieżka.wav")), "ścieżka");
    }

    #[test]
    fn every_column_of_every_mode_is_described() {
        for &mode in VisualizationMode::ALL {
            for spatial in [false, true] {
                let cfg = RadarConfig {
                    visualization_mode: mode,
                    spatial_audio_view: spatial,
                    ..RadarConfig::default()
                };
                for column in columns_for(&cfg, 6) {
                    assert!(!describe(&column).is_empty(), "brak opisu kolumny {column} w trybie {}", mode_key(mode));
                }
            }
        }
    }

    #[test]
    fn layer_columns_follow_the_three_by_three_view() {
        let columns = columns_for(&cfg_for(M::LayerBars), 2);
        assert!(has(&columns, "moc_kroki_L_proc"));
        assert!(has(&columns, "moc_wszystko_R_proc"));
        assert!(has(&columns, "podswietlona"));
        assert!(has(&columns, "kanal_L_dbfs"), "warstwy mają poziomy L i R zdarzenia");
        assert!(!has(&columns, "moc_wewnetrzny_L_proc"), "3+3 nie pokazuje łuków 2+2");
        assert!(!has(&columns, "kanal_FL_dbfs"), "poziomy kanałów są z ramki zdarzenia");
    }

    #[test]
    fn two_by_two_columns_follow_the_two_by_two_view() {
        let columns = columns_for(&cfg_for(M::LayerBars22), 2);
        assert!(has(&columns, "moc_wewnetrzny_L_proc"));
        assert!(has(&columns, "poswiata_proc"));
        assert!(!has(&columns, "moc_kroki_L_proc"), "2+2 nie ma osobnego łuku kroków");
    }

    #[test]
    fn radar_columns_list_the_channels_of_the_layout() {
        let columns = columns_for(&cfg_for(M::OnlyRadar), 6);
        assert!(has(&columns, "radar_kolor_R"));
        assert!(has(&columns, "kanal_FL_dbfs") && has(&columns, "kanal_LFE_dbfs"));
        assert!(!has(&columns, "klasa"), "radar nie pokazuje klas");
        assert!(!has(&columns, "bars_L_proc"), "radar bez słupków");
        let bars = columns_for(&cfg_for(M::OnlyBars), 2);
        assert!(has(&bars, "bars_L_proc") && has(&bars, "kanal_FL_dbfs"));
        assert!(!has(&bars, "radar_kolor_R"));
    }

    #[test]
    fn nothing_mode_writes_no_view_or_channel_columns() {
        let columns = columns_for(&cfg_for(M::None), 2);
        assert_eq!(columns, ["czas_s", "nr", "tryb", "dzwiek_tekst", "widok_tekst"]);
    }

    #[test]
    fn sound_text_names_the_grenade_and_the_side() {
        let grenade = LayerHit {
            layer: Layer::Shots,
            class: Some(Class::D),
            pan: -0.6,
            peak_dbfs: -38.2,
            peak_left_dbfs: -38.2,
            peak_right_dbfs: -38.2,
            brightness_db: Some(-7.6),
            gap_s: Some(0.2),
            onset_s: 0.0,
        };
        let text = sound_text(&grenade);
        assert!(text.starts_with("granat"), "{text}");
        assert!(text.contains("z lewej 60%"), "{text}");
        assert!(text.contains("odstęp 0.20 s"), "{text}");
        let centred = LayerHit { pan: 0.02, gap_s: None, ..grenade };
        assert!(sound_text(&centred).contains("środek"));
    }

    #[test]
    fn csv_quotes_fields_with_commas() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("x\"y"), "\"x\"\"y\"");
        assert_eq!(csv_field("plain"), "plain");
    }

    #[test]
    fn a_played_burst_becomes_one_numeric_row_with_its_time_and_both_texts() {
        use crate::sound::devices::{self, SharedAudioAnalysis};
        let rate = 48_000.0f32;
        // Sekunda stereo: głośny ton 1 kHz w 0,2 s, długości 0,1 s.
        let mut samples = vec![0.0f32; 2 * rate as usize];
        for i in 0..4_800 {
            let s = 0.2 * (std::f32::consts::TAU * 1_000.0 * i as f32 / rate).sin();
            samples[2 * (9_600 + i)] = s;
            samples[2 * (9_600 + i) + 1] = s;
        }
        let analysis = SharedAudioAnalysis::new();
        let cfg = cfg_for(M::LayerBars);
        let dir = temp_dir("burst");
        let mut log = RadarLogSession::open(&dir, Path::new("burst.wav"), &cfg, 0.0, 2).unwrap();
        let mut state = RadarState::new();
        let mut now = 0.0;
        for chunk in samples.chunks(480 * 2) {
            devices::update_analysis(chunk, 2, rate, &analysis, None);
            now += 0.01;
            let hits = analysis.lock().unwrap().drain_layer_hits();
            state.record_layer_hits(now, &hits);
            let view = view_snapshot(&state, &cfg, now);
            log.log_hits(&hits, &view, &[]).unwrap();
        }
        let path = log.path().to_path_buf();
        assert_eq!(log.rows(), 1, "jedno uderzenie, jeden wiersz");
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let row = text.lines().last().unwrap();
        let fields: Vec<&str> = row.split(',').collect();
        assert!(row.starts_with("0.2"), "czas od startu odtwarzania: {row}");
        assert_eq!(fields[2], "4", "tryb 3+3 to kod 4: {row}");
        assert_eq!(fields[3], "1", "klasa A to kod 1: {row}");
        assert_eq!(fields[7], "1", "strzał to warstwa 1: {row}");
        assert!(row.contains("strzał lub wybuch"), "opis dźwięku: {row}");
        assert!(row.contains("3+3: kroki L"), "opis widoku: {row}");
    }

    #[test]
    fn session_writes_header_and_one_row_per_hit_from_its_start() {
        let dir = temp_dir("session");
        let cfg = cfg_for(M::LayerBars);
        let mut log = RadarLogSession::open(&dir, Path::new("test.wav"), &cfg, 1.0, 2).unwrap();
        let state = RadarState::new();
        let view = view_snapshot(&state, &cfg, 0.0);
        // Zdarzenie przed startem logu jest pomijane, drugie liczy się od startu.
        let written = log
            .log_hits(&[shot(0.5, -14.0, -1.0), shot(1.25, -14.0, 0.0)], &view, &[])
            .unwrap();
        assert_eq!(written, 1);
        assert_eq!(log.rows(), 1);
        let path = log.path().to_path_buf();
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(text.starts_with("# Hear Radar: log radaru"));
        assert!(text.contains("tryb wyświetlania: warstwy_3p3"));
        assert!(text.contains("# kody: tryb 0 nic"), "legenda kodów w nagłówku");
        let header_line = text.lines().find(|l| l.starts_with("czas_s,")).expect("nagłówek kolumn");
        assert!(header_line.starts_with("czas_s,nr,tryb,klasa,"));
        assert!(header_line.ends_with(",dzwiek_tekst,widok_tekst"));
        let row = text.lines().last().unwrap();
        assert!(row.starts_with("0.250,1,4,1,"), "{row}");
        assert!(row.contains("strzał lub wybuch"), "{row}");
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields[9], "-14.0", "kanał L z ramki szczytu: {row}");
        assert_eq!(fields[10], "-14.0", "kanał R z ramki szczytu: {row}");
    }

    #[test]
    fn radar_rows_carry_the_channel_levels_of_the_frame() {
        let dir = temp_dir("radar_channels");
        let cfg = cfg_for(M::OnlyRadar);
        let mut log = RadarLogSession::open(&dir, Path::new("radar.wav"), &cfg, 0.0, 2).unwrap();
        let view = view_snapshot(&RadarState::new(), &cfg, 0.0);
        // Poziomy RMS 0,1 i 0,01 to −20 dB i −40 dB.
        log.log_hits(&[shot(0.5, -14.0, -1.0)], &view, &[0.1, 0.01]).unwrap();
        let path = log.path().to_path_buf();
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let row = text.lines().last().unwrap();
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields[2], "1", "tylko radar to kod 1: {row}");
        assert_eq!(fields[3], "-20.0", "kanał FL: {row}");
        assert_eq!(fields[4], "-40.0", "kanał FR: {row}");
    }
}
