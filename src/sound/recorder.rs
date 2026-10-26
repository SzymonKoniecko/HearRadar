//! Nagrywanie sesji do treningu: `radar --recordSample <nazwa sesji>` (dźwięk)
//! i `radar --recordVideo <nazwa sesji>` (dźwięk + wideo z `video.rs`).
//!
//! Zapisuje dwa pierwsze kanały urządzenia jako WAV 16-bit (indeks 0 i 1; patrz pomiar
//! S0 w `pomiary.txt`). Obok leżą `session.json` z metadanymi i `markers.csv`
//! ze znacznikami czasu. Każda sesja ma własny katalog, a istniejący nie jest nadpisywany.
//!
//! Wzorzec jak w `sample_logger`: callback audio tylko kopiuje dwa kanały do kolejki,
//! a zapis na dysk robi wątek roboczy. Kopia to jedna alokacja na ramkę audio. Zniknie,
//! gdy analiza przejdzie do wątku roboczego (A4/A6).
//!
//! Nagłówek WAV jest poprawiany co sekundę, więc po przerwaniu procesu plik da się
//! odtworzyć do ostatniej sekundy. Status w `session.json` zostaje wtedy „recording”.

use anyhow::{Context, Result, anyhow, bail};
use cpal::StreamInstant;
use cpal::traits::DeviceTrait;
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::sound::channel_test::find_output_device;
use crate::sound::devices::{SharedAudioAnalysis, fetch_audio_from_device};
use crate::sound::video::{self, VIDEO_FILE, VideoCapture, VideoOptions};

/// Nagrywane są pierwsze dwa kanały: L i P. Urządzenie mono nagrywa jeden.
const MAX_RECORDED_CHANNELS: usize = 2;
const WAV_HEADER_LEN: usize = 44;
const PATCH_INTERVAL: Duration = Duration::from_secs(1);
const AUDIO_FILE: &str = "audio.wav";
const SESSION_FILE: &str = "session.json";
const MARKERS_FILE: &str = "markers.csv";

/// Metadane podawane z linii poleceń. Brakujące pola zapisują się jako `null`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SessionMeta {
    pub game: Option<String>,
    pub scene: Option<String>,
    pub mode: Option<String>,
    pub output: Option<String>,
    pub spatial_sound: Option<String>,
    pub game_volume: Option<String>,
    pub windows_volume: Option<String>,
}

/// Zawartość `session.json`. Zapisywana na starcie i ponownie po zakończeniu.
#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub name: String,
    pub status: String,
    pub started_unix_ms: u128,
    pub source_device: String,
    pub source_channels: u16,
    pub recorded_channels: u16,
    pub channel_map: Vec<String>,
    pub sample_rate: u32,
    pub bit_depth: u16,
    pub audio_file: String,
    pub frames: u64,
    pub duration_s: f64,
    /// Plik wideo w katalogu sesji, jeśli nagrywano wideo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_file: Option<String>,
    /// O ile sekund wideo wystartowało po dźwięku (ujemne: wcześniej).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_offset_s: Option<f64>,
    #[serde(flatten)]
    pub meta: SessionMeta,
}

#[derive(Debug)]
struct ActiveSession {
    dir: PathBuf,
    info: SessionInfo,
}

/// Ile brakującej ciszy wolno przeoczyć bez dopełniania (0,1 s przy 48 kHz). Mniejsze luki to drżenie.
const PAD_THRESHOLD_FRAMES: u64 = 4_800;
/// Ciszę dopisujemy porcjami tej długości, żeby długa przerwa nie tworzyła ogromnej tablicy.
const SILENCE_CHUNK_FRAMES: u64 = 48_000;
/// Najmniejsza luka dopełniana od razu (1 ms przy 48 kHz). Krótsze kumulujemy, aż dojdą do tego progu.
const MIN_GAP_FRAMES: u64 = 48;

/// Oś czasu z WASAPI: koniec ostatniej porcji i luka, która jeszcze nie jest dopełniona ciszą.
#[derive(Debug, Default)]
struct CaptureClock {
    end: Option<StreamInstant>,
    owed_frames: u64,
}

#[derive(Debug)]
pub struct Recorder {
    /// Czy trwa nagrywanie. Czytane z wątku audio na każdą ramkę, więc musi być tanie.
    enabled: AtomicBool,
    /// Klatki przyjęte od startu sesji. Liczone w callbacku, więc dają czas znaczników.
    frames: AtomicU64,
    sample_rate: AtomicU32,
    source_channels: AtomicU16,
    /// Chwila przyjęcia pierwszej ramki audio, czyli początku `audio.wav`.
    first_sample_at: Mutex<Option<Instant>>,
    /// Oś czasu z czasów przechwycenia WASAPI, z której liczymy luki w dźwięku.
    capture: Mutex<CaptureClock>,
    active: Mutex<Option<ActiveSession>>,
    tx: Mutex<Sender<Command>>,
}

enum Command {
    Start(Box<WavWriter>),
    Frames(Vec<f32>),
    /// Kończy zapis i odsyła liczbę zapisanych klatek.
    Stop(SyncSender<u64>),
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            frames: AtomicU64::new(0),
            sample_rate: AtomicU32::new(0),
            source_channels: AtomicU16::new(0),
            first_sample_at: Mutex::new(None),
            capture: Mutex::new(CaptureClock::default()),
            active: Mutex::new(None),
            tx: Mutex::new(spawn_writer_thread()),
        }
    }

    pub fn is_recording(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Rozpoczyna sesję w `base_dir/name`. Zwraca katalog sesji.
    pub fn start(
        &self,
        base_dir: &Path,
        name: &str,
        source_device: &str,
        source_channels: u16,
        sample_rate: u32,
        meta: SessionMeta,
    ) -> Result<PathBuf> {
        if self.is_recording() {
            bail!("nagrywanie już trwa");
        }
        validate_name(name)?;
        let dir = base_dir.join(name);
        if dir.exists() {
            bail!("sesja '{name}' już istnieje ({}), wybierz inną nazwę", dir.display());
        }
        fs::create_dir_all(&dir).with_context(|| format!("utworzenie {}", dir.display()))?;

        let recorded = source_channels.min(MAX_RECORDED_CHANNELS as u16);
        let started_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let info = SessionInfo {
            name: name.to_string(),
            status: "recording".to_string(),
            started_unix_ms,
            source_device: source_device.to_string(),
            source_channels,
            recorded_channels: recorded,
            channel_map: channel_map(recorded),
            sample_rate,
            bit_depth: 16,
            audio_file: AUDIO_FILE.to_string(),
            frames: 0,
            duration_s: 0.0,
            video_file: None,
            video_offset_s: None,
            meta,
        };
        write_session(&dir, &info)?;
        fs::write(dir.join(MARKERS_FILE), "t_s,frame,label\n")
            .with_context(|| format!("zapis {MARKERS_FILE}"))?;

        let mut file = BufWriter::new(
            File::create(dir.join(AUDIO_FILE)).with_context(|| format!("utworzenie {AUDIO_FILE}"))?,
        );
        file.write_all(&wav_header(recorded, sample_rate, 0))?;
        let writer = WavWriter {
            file,
            channels: recorded,
            sample_rate,
            data_bytes: 0,
            last_patch: Instant::now(),
            failed: false,
        };

        self.frames.store(0, Ordering::Relaxed);
        if let Ok(mut first) = self.first_sample_at.lock() {
            *first = None;
        }
        if let Ok(mut clock) = self.capture.lock() {
            *clock = CaptureClock::default();
        }
        self.sample_rate.store(sample_rate, Ordering::Relaxed);
        self.source_channels.store(source_channels, Ordering::Relaxed);
        *self.active.lock().map_err(|_| anyhow!("blokada sesji"))? = Some(ActiveSession {
            dir: dir.clone(),
            info,
        });
        self.send(Command::Start(Box::new(writer)))?;
        // Start musi dotrzeć do wątku zapisu przed pierwszą ramką, dlatego flaga idzie na końcu.
        self.enabled.store(true, Ordering::Release);
        Ok(dir)
    }

    /// Kończy sesję, domyka WAV i zapisuje `session.json`. Zwraca liczbę klatek.
    pub fn stop(&self) -> Result<u64> {
        self.enabled.store(false, Ordering::Release);
        let mut session = self
            .active
            .lock()
            .map_err(|_| anyhow!("blokada sesji"))?
            .take()
            .ok_or_else(|| anyhow!("nagrywanie nie trwa"))?;

        // Dopełnienie do końca sesji, żeby audio miało tyle co wideo, nawet gdy na końcu była cisza.
        let missing = self.missing_frames_at(Instant::now(), 0, 1);
        self.send_silence(missing, session.info.recorded_channels as usize)?;

        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        self.send(Command::Stop(ack_tx))?;
        let frames = ack_rx.recv().context("wątek zapisu nie odpowiedział")?;

        session.info.status = "done".to_string();
        session.info.frames = frames;
        session.info.duration_s = frames as f64 / session.info.sample_rate.max(1) as f64;
        write_session(&session.dir, &session.info)?;
        Ok(frames)
    }

    /// Chwila przyjęcia pierwszej ramki audio, czyli sample 0 w `audio.wav`.
    pub fn first_sample_at(&self) -> Option<Instant> {
        self.first_sample_at.lock().ok().and_then(|first| *first)
    }

    /// Ile ramek ciszy trzeba dopisać, żeby audio dotrzymało kroku zegarowi od pierwszej ramki.
    /// `incoming` to ramki, które właśnie przychodzą i są już wliczone w czas `now`. `threshold`
    /// to najmniejsza luka, którą w ogóle dopełniamy.
    fn missing_frames_at(&self, now: Instant, incoming: u64, threshold: u64) -> u64 {
        let Some(first) = self.first_sample_at() else {
            return 0;
        };
        let rate = self.sample_rate.load(Ordering::Relaxed) as f64;
        let expected = (now.saturating_duration_since(first).as_secs_f64() * rate) as u64;
        let have = self.frames.load(Ordering::Relaxed) + incoming;
        let missing = expected.saturating_sub(have);
        if missing >= threshold { missing } else { 0 }
    }

    /// Wpisuje do zapisu `frames` ramek ciszy (zera), porcjami.
    fn send_silence(&self, frames: u64, recorded: usize) -> Result<()> {
        let mut left = frames;
        while left > 0 {
            let part = left.min(SILENCE_CHUNK_FRAMES);
            left -= part;
            self.frames.fetch_add(part, Ordering::Relaxed);
            self.send(Command::Frames(vec![0.0; part as usize * recorded]))?;
        }
        Ok(())
    }

    /// Zapisuje w `session.json`, że sesja ma wideo i jak jest przesunięte względem dźwięku.
    pub fn set_video(&self, file: &str, offset_s: f64) -> Result<()> {
        let mut guard = self.active.lock().map_err(|_| anyhow!("blokada sesji"))?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("nagrywanie nie trwa"))?;
        session.info.video_file = Some(file.to_string());
        session.info.video_offset_s = Some(offset_s);
        write_session(&session.dir, &session.info)
    }

    /// Dopisuje znacznik w bieżącej chwili nagrania. Zwraca czas w sekundach.
    pub fn add_marker(&self, label: &str) -> Option<f64> {
        if !self.is_recording() {
            return None;
        }
        let frame = self.frames.load(Ordering::Relaxed);
        let sample_rate = self.sample_rate.load(Ordering::Relaxed).max(1);
        let t = frame as f64 / sample_rate as f64;
        let dir = self.active.lock().ok()?.as_ref()?.dir.clone();
        let clean = label.replace(',', ";").replace(['\n', '\r'], " ");
        let mut file = OpenOptions::new().append(true).open(dir.join(MARKERS_FILE)).ok()?;
        writeln!(file, "{t:.3},{frame},{clean}").ok()?;
        Some(t)
    }

    /// Wołane z callbacku audio przy każdej porcji. Bez nagrywania kończy od razu. `captured` to czas
    /// przechwycenia pierwszej ramki porcji z WASAPI. `None` (odtwarzanie pliku) oznacza brak czasu,
    /// wtedy luki liczymy zegarem.
    pub fn push(&self, samples: &[f32], channels: usize, captured: Option<StreamInstant>) {
        if !self.is_recording() || channels == 0 {
            return;
        }
        if channels as u16 != self.source_channels.load(Ordering::Relaxed) {
            return;
        }
        let recorded = channels.min(MAX_RECORDED_CHANNELS);
        let frames = samples.len() / channels;
        // Loopback nie dostarcza próbek, gdy nic nie gra. Lukę dopełniamy ciszą w miejscu, w którym
        // wystąpiła. Liczymy ją z czasu przechwycenia, a nie z zegara w callbacku, bo spóźniony
        // callback nie jest przerwą w dźwięku.
        let missing = match captured {
            Some(at) => self.gap_before(at, frames),
            None => self.missing_frames_at(Instant::now(), frames as u64, PAD_THRESHOLD_FRAMES),
        };
        if self.send_silence(missing, recorded).is_err() {
            return;
        }
        let mut chunk = Vec::with_capacity(frames * recorded);
        for frame in samples.chunks_exact(channels) {
            chunk.extend_from_slice(&frame[..recorded]);
        }
        let before = self.frames.fetch_add(frames as u64, Ordering::Relaxed);
        if before == 0 {
            if let Ok(mut first) = self.first_sample_at.lock() {
                *first = Some(Instant::now());
            }
        }
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(Command::Frames(chunk));
        }
    }

    /// Ile ramek ciszy dopisać przed porcją, która zaczyna się w `at` (czas przechwycenia z WASAPI).
    /// Luka liczy się od końca poprzedniej porcji. Luki krótsze niż [`MIN_GAP_FRAMES`] kumulują się,
    /// aż przekroczą ten próg, i dopiero wtedy są dopełniane.
    fn gap_before(&self, at: StreamInstant, frames: usize) -> u64 {
        let rate = self.sample_rate.load(Ordering::Relaxed) as f64;
        if rate <= 0.0 {
            return 0;
        }
        let Ok(mut clock) = self.capture.lock() else {
            return 0;
        };
        if let Some(gap) = clock.end.and_then(|end| at.duration_since(&end)) {
            clock.owed_frames += (gap.as_secs_f64() * rate).round() as u64;
        }
        clock.end = at.add(Duration::from_secs_f64(frames as f64 / rate)).or(Some(at));
        if clock.owed_frames >= MIN_GAP_FRAMES {
            std::mem::take(&mut clock.owed_frames)
        } else {
            0
        }
    }

    fn send(&self, command: Command) -> Result<()> {
        self.tx
            .lock()
            .map_err(|_| anyhow!("blokada kanału zapisu"))?
            .send(command)
            .map_err(|_| anyhow!("wątek zapisu nie działa"))
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Katalog na sesje: `recordings/` w korzeniu repozytorium (poza gitem).
pub fn recordings_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("recordings")
}

/// Uruchamia sesję z linii poleceń. Bez `duration` nagrywa do `q`. Z `duration`
/// kończy się po tym czasie, a `q` przerywa wcześniej. Każda inna linia to znacznik.
/// Z `video` nagrywa też obraz przez ffmpeg (patrz `sound::video`).
pub fn run(
    session_name: &str,
    device_name: &str,
    meta: SessionMeta,
    duration: Option<Duration>,
    video: Option<VideoOptions>,
) -> Result<()> {
    if let Some(options) = &video {
        video::check_ffmpeg(&options.ffmpeg)?;
    }
    let device = find_output_device(device_name)?;
    let source = device.name()?;
    let config = device.default_output_config()?;
    let channels = config.channels();
    let sample_rate = config.sample_rate().0;

    let analysis = SharedAudioAnalysis::new();
    let mut meta = meta;
    if meta.mode.is_none() {
        meta.mode = Some("stereo".to_string());
    }
    let audio_started = Instant::now();
    let dir = analysis
        .recorder()
        .start(&recordings_dir(), session_name, &source, channels, sample_rate, meta)?;

    let _loopback = match fetch_audio_from_device(&source, analysis.clone()) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = analysis.recorder().stop();
            return Err(err);
        }
    };

    let capture = match &video {
        Some(options) => {
            // Przesunięcie = pierwsza klatka wideo względem startu dźwięku.
            let started = VideoCapture::start(options, &dir.join(VIDEO_FILE)).and_then(|capture| {
                match capture.wait_first_frame() {
                    Ok(first_frame) => Ok((capture, first_frame)),
                    Err(err) => {
                        capture.abort();
                        Err(err)
                    }
                }
            });
            match started {
                Ok((capture, first_frame)) => {
                    // Od pierwszej ramki audio, a nie od startu sesji: loopback startuje z opóźnieniem.
                    let audio_origin = analysis.recorder().first_sample_at().unwrap_or(audio_started);
                    let offset = first_frame.saturating_duration_since(audio_origin).as_secs_f64();
                    if let Err(err) = analysis.recorder().set_video(VIDEO_FILE, offset) {
                        capture.abort();
                        let _ = analysis.recorder().stop();
                        return Err(err);
                    }
                    Some(capture)
                }
                Err(err) => {
                    let _ = analysis.recorder().stop();
                    let _ = fs::remove_dir_all(&dir);
                    return Err(err);
                }
            }
        }
        None => None,
    };

    println!("Nagrywanie: {}", dir.display());
    println!(
        "Źródło: {source}, {channels} kan., {sample_rate} Hz, zapis: {} kan.",
        channels.min(MAX_RECORDED_CHANNELS as u16)
    );
    match duration {
        Some(limit) => println!(
            "Nagrywanie przez {} s. Etykieta + Enter dodaje znacznik, q + Enter kończy wcześniej.",
            limit.as_secs()
        ),
        None => println!("Wpisz etykietę i Enter, żeby dodać znacznik. Wpisz q i Enter, żeby zakończyć."),
    }

    // Czytanie konsoli w osobnym wątku, żeby pętla mogła też pilnować czasu.
    let (line_tx, line_rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });
    control_loop(&line_rx, duration, |label| match analysis.recorder().add_marker(label) {
        Some(t) => println!("znacznik t={t:.2} s: {label}"),
        None => println!("znacznik pominięty (nagrywanie nie trwa)"),
    });

    // Najpierw wideo (ffmpeg kończy się na `q`), potem dźwięk.
    let video_result = capture.map(VideoCapture::stop).transpose();
    let frames = analysis.recorder().stop()?;
    video_result?;
    println!(
        "Zapisano {frames} klatek ({:.1} s) w {}",
        frames as f64 / sample_rate.max(1) as f64,
        dir.display()
    );
    if video.is_some() {
        println!("Wideo: {}", dir.join(VIDEO_FILE).display());
    }
    Ok(())
}

/// Pętla sterowania nagraniem. Czeka na linie z konsoli albo na koniec czasu.
/// Kończy się, gdy wpisano `q`, minął `duration`, albo wejście się zamknęło
/// (bez limitu czasu). Każda inna niepusta linia trafia do `on_label`.
fn control_loop(lines: &Receiver<String>, duration: Option<Duration>, mut on_label: impl FnMut(&str)) {
    const POLL: Duration = Duration::from_millis(200);
    let started = Instant::now();
    loop {
        let wait = match duration {
            Some(limit) => {
                let left = limit.saturating_sub(started.elapsed());
                if left.is_zero() {
                    return;
                }
                left.min(POLL)
            }
            None => POLL,
        };
        match lines.recv_timeout(wait) {
            Ok(line) => {
                let label = line.trim();
                if label == "q" {
                    return;
                }
                if !label.is_empty() {
                    on_label(label);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                // Wejście zamknięte (np. koniec stdin). Z limitem czasu czekamy do końca.
                if let Some(limit) = duration {
                    std::thread::sleep(limit.saturating_sub(started.elapsed()));
                }
                return;
            }
        }
    }
}

/// Opis kanałów w pliku, żeby z samego `session.json` było wiadomo, co jest czym.
/// Indeks 0 zakładamy jako lewy: to do potwierdzenia testem kierunku (S3).
fn channel_map(recorded: u16) -> Vec<String> {
    match recorded {
        1 => vec!["0 = mono".to_string()],
        _ => vec!["0 = lewy (L)".to_string(), "1 = prawy (P)".to_string()],
    }
}

fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !ok {
        bail!("nazwa sesji może zawierać tylko litery ASCII, cyfry, '-' i '_'");
    }
    Ok(())
}

fn write_session(dir: &Path, info: &SessionInfo) -> Result<()> {
    let json = serde_json::to_string_pretty(info).context("serializacja session.json")?;
    fs::write(dir.join(SESSION_FILE), json).context("zapis session.json")
}

/// Nagłówek WAV PCM 16-bit. Rozmiary są wpisywane osobno, bo znane są dopiero na końcu.
fn wav_header(channels: u16, sample_rate: u32, data_bytes: u64) -> [u8; WAV_HEADER_LEN] {
    let block_align = channels * 2;
    let byte_rate = sample_rate * block_align as u32;
    let data_size = u32::try_from(data_bytes).unwrap_or(u32::MAX);
    let riff_size = u32::try_from(36 + data_bytes).unwrap_or(u32::MAX);

    let mut h = [0u8; WAV_HEADER_LEN];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&riff_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}

struct WavWriter {
    file: BufWriter<File>,
    channels: u16,
    sample_rate: u32,
    data_bytes: u64,
    last_patch: Instant,
    failed: bool,
}

impl WavWriter {
    fn write_samples(&mut self, samples: &[f32]) {
        if self.failed {
            return;
        }
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(err) = self.file.write_all(&bytes) {
            eprintln!("[recorder] błąd zapisu: {err}");
            self.failed = true;
            return;
        }
        self.data_bytes += bytes.len() as u64;
        if self.last_patch.elapsed() >= PATCH_INTERVAL {
            if let Err(err) = self.patch_header() {
                eprintln!("[recorder] błąd nagłówka WAV: {err}");
                self.failed = true;
            }
            self.last_patch = Instant::now();
        }
    }

    fn patch_header(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file
            .write_all(&wav_header(self.channels, self.sample_rate, self.data_bytes))?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    /// Domyka plik i zwraca liczbę zapisanych klatek.
    fn finish(mut self) -> u64 {
        if let Err(err) = self.patch_header() {
            eprintln!("[recorder] błąd nagłówka WAV: {err}");
        }
        let _ = self.file.flush();
        self.data_bytes / (self.channels as u64 * 2).max(1)
    }
}

fn spawn_writer_thread() -> Sender<Command> {
    let (tx, rx) = mpsc::channel::<Command>();
    std::thread::spawn(move || {
        let mut writer: Option<WavWriter> = None;
        for command in rx {
            match command {
                Command::Start(w) => writer = Some(*w),
                Command::Frames(samples) => {
                    if let Some(w) = writer.as_mut() {
                        w.write_samples(&samples);
                    }
                }
                Command::Stop(ack) => {
                    let frames = writer.take().map(WavWriter::finish).unwrap_or(0);
                    let _ = ack.send(frames);
                }
            }
        }
    });
    tx
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// Katalog tymczasowy dla jednego testu. Nigdy nie wskazuje na `recordings/` w repo.
    fn temp_base(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hearradar_rec_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Ramki przeplecione: kanał `k` ma stałą wartość `values[k]`.
    fn interleaved(values: &[f32], frames: usize) -> Vec<f32> {
        (0..frames).flat_map(|_| values.iter().copied()).collect()
    }

    fn read_samples(path: &Path) -> Vec<i16> {
        let bytes = fs::read(path).unwrap();
        bytes[WAV_HEADER_LEN..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    #[test]
    fn records_only_the_first_two_channels_of_an_eight_channel_source() {
        let base = temp_base("two_of_eight");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "CABLE Input", 8, SR, SessionMeta::default())
            .unwrap();
        let frame = [0.5, -0.25, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        rec.push(&interleaved(&frame, 100), 8, None);
        let frames = rec.stop().unwrap();
        assert_eq!(frames, 100);

        let samples = read_samples(&dir.join(AUDIO_FILE));
        assert_eq!(samples.len(), 200, "100 klatek × 2 kanały");
        assert!((samples[0] - 16_383).abs() <= 1, "L: {}", samples[0]);
        assert!((samples[1] - -8_191).abs() <= 1, "P: {}", samples[1]);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn header_sizes_match_the_file_length() {
        let base = temp_base("header");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "CABLE Input", 2, SR, SessionMeta::default())
            .unwrap();
        rec.push(&interleaved(&[0.1, 0.2], 480), 2, None);
        rec.stop().unwrap();

        let bytes = fs::read(dir.join(AUDIO_FILE)).unwrap();
        assert_eq!(bytes.len(), WAV_HEADER_LEN + 480 * 2 * 2);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let riff = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let data = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(riff as usize, bytes.len() - 8);
        assert_eq!(data as usize, bytes.len() - WAV_HEADER_LEN);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2, "kanałów");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), SR, "Hz");
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16, "bitów");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn mono_source_records_one_channel() {
        let base = temp_base("mono");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "Mono device", 1, SR, SessionMeta::default())
            .unwrap();
        rec.push(&interleaved(&[0.5], 50), 1, None);
        assert_eq!(rec.stop().unwrap(), 50);
        let bytes = fs::read(dir.join(AUDIO_FILE)).unwrap();
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(bytes.len(), WAV_HEADER_LEN + 50 * 2);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn markers_carry_the_frame_position_and_time() {
        let base = temp_base("markers");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "CABLE Input", 8, SR, SessionMeta::default())
            .unwrap();
        rec.push(&interleaved(&[0.0; 8], 4_800), 8, None);
        let t = rec.add_marker("strzał, z lewej").unwrap();
        assert!((t - 0.1).abs() < 1e-9, "t={t}");
        rec.stop().unwrap();

        let markers = fs::read_to_string(dir.join(MARKERS_FILE)).unwrap();
        assert!(markers.starts_with("t_s,frame,label\n"));
        assert!(markers.contains("0.100,4800,strzał; z lewej"), "{markers}");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn session_json_records_metadata_and_final_counts() {
        let base = temp_base("json");
        let rec = Recorder::new();
        let meta = SessionMeta {
            game: Some("Gra X".to_string()),
            scene: Some("Mapa 1".to_string()),
            game_volume: Some("70%".to_string()),
            ..SessionMeta::default()
        };
        let dir = rec.start(&base, "sesja", "CABLE Input", 8, SR, meta).unwrap();
        rec.push(&interleaved(&[0.0; 8], 960), 8, None);
        rec.stop().unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(SESSION_FILE)).unwrap()).unwrap();
        assert_eq!(json["status"], "done");
        assert_eq!(json["frames"], 960);
        assert_eq!(json["recorded_channels"], 2);
        assert_eq!(json["source_channels"], 8);
        assert_eq!(json["sample_rate"], SR);
        assert_eq!(json["game"], "Gra X");
        assert_eq!(json["scene"], "Mapa 1");
        assert_eq!(json["game_volume"], "70%");
        assert!(json["windows_volume"].is_null());
        assert!((json["duration_s"].as_f64().unwrap() - 0.02).abs() < 1e-9);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn nothing_is_recorded_before_start() {
        let rec = Recorder::new();
        rec.push(&interleaved(&[0.3; 8], 100), 8, None);
        assert!(!rec.is_recording());
        assert!(rec.add_marker("nic").is_none());
        assert!(rec.stop().is_err());
    }

    #[test]
    fn refuses_an_existing_session_and_unsafe_names() {
        let base = temp_base("refuse");
        let rec = Recorder::new();
        rec.start(&base, "sesja", "CABLE Input", 2, SR, SessionMeta::default()).unwrap();
        assert!(
            rec.start(&base, "inna", "CABLE Input", 2, SR, SessionMeta::default()).is_err(),
            "drugi start w trakcie nagrywania"
        );
        rec.stop().unwrap();
        assert!(
            rec.start(&base, "sesja", "CABLE Input", 2, SR, SessionMeta::default()).is_err(),
            "nazwa już istnieje"
        );
        for bad in ["", "../x", "a b", "zażółć"] {
            assert!(
                rec.start(&base, bad, "CABLE Input", 2, SR, SessionMeta::default()).is_err(),
                "nazwa '{bad}' powinna być odrzucona"
            );
        }
        let _ = fs::remove_dir_all(&base);
    }
}

#[cfg(test)]
mod control_tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn stops_after_the_duration_when_no_input_arrives() {
        let (_tx, rx) = mpsc::channel::<String>();
        let started = Instant::now();
        control_loop(&rx, Some(Duration::from_millis(60)), |_| panic!("bez wejścia nie ma znaczników"));
        assert!(started.elapsed() >= Duration::from_millis(60));
    }

    #[test]
    fn collects_labels_and_stops_on_q() {
        let (tx, rx) = mpsc::channel::<String>();
        for line in ["strzał", "  ", "", "krok", "q", "po q nie czytam"] {
            tx.send(line.to_string()).unwrap();
        }
        let mut labels = Vec::new();
        control_loop(&rx, None, |l| labels.push(l.to_string()));
        assert_eq!(labels, vec!["strzał", "krok"]);
    }

    #[test]
    fn q_stops_before_the_duration_ends() {
        let (tx, rx) = mpsc::channel::<String>();
        tx.send("q".to_string()).unwrap();
        let started = Instant::now();
        control_loop(&rx, Some(Duration::from_secs(60)), |_| {});
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn without_duration_closed_input_ends_immediately() {
        let (tx, rx) = mpsc::channel::<String>();
        drop(tx);
        control_loop(&rx, None, |_| {});
    }

    #[test]
    fn closed_input_still_waits_out_the_duration() {
        let (tx, rx) = mpsc::channel::<String>();
        drop(tx);
        let started = Instant::now();
        control_loop(&rx, Some(Duration::from_millis(60)), |_| {});
        assert!(started.elapsed() >= Duration::from_millis(60));
    }
}

#[cfg(test)]
mod video_fields_tests {
    use super::*;

    fn temp_base(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hearradar_rec_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn read_json(dir: &Path) -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(dir.join(SESSION_FILE)).unwrap()).unwrap()
    }

    #[test]
    fn set_video_is_written_to_session_json_and_kept_after_stop() {
        let base = temp_base("video_fields");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "CABLE Input", 8, 48_000, SessionMeta::default())
            .unwrap();
        rec.set_video(VIDEO_FILE, 0.25).unwrap();

        let during = read_json(&dir);
        assert_eq!(during["video_file"], "video.mp4");
        assert!((during["video_offset_s"].as_f64().unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(during["status"], "recording");

        rec.stop().unwrap();
        let after = read_json(&dir);
        assert_eq!(after["status"], "done");
        assert_eq!(after["video_file"], "video.mp4");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn sessions_without_video_have_no_video_fields() {
        let base = temp_base("no_video_fields");
        let rec = Recorder::new();
        let dir = rec
            .start(&base, "sesja", "CABLE Input", 8, 48_000, SessionMeta::default())
            .unwrap();
        rec.stop().unwrap();
        let json = read_json(&dir);
        assert!(json.get("video_file").is_none());
        assert!(json.get("video_offset_s").is_none());
        let _ = fs::remove_dir_all(&base);
    }
}

#[cfg(test)]
mod padding_tests {
    use super::*;

    /// Luka w loopbacku (brak próbek, gdy nic nie gra) jest dopełniana do czasu zegara.
    #[test]
    fn gaps_in_the_loopback_are_padded_to_the_clock() {
        let rec = Recorder::new();
        rec.sample_rate.store(48_000, Ordering::Relaxed);
        let now = Instant::now();
        *rec.first_sample_at.lock().unwrap() = Some(now - Duration::from_secs(1));
        let missing = rec.missing_frames_at(now, 0, 1);
        assert!((47_000..=48_000).contains(&missing), "brakujące ramki: {missing}");
    }

    /// Drobne opóźnienie poniżej progu nie jest dopełniane.
    #[test]
    fn small_jitter_is_not_padded() {
        let rec = Recorder::new();
        rec.sample_rate.store(48_000, Ordering::Relaxed);
        let now = Instant::now();
        *rec.first_sample_at.lock().unwrap() = Some(now - Duration::from_millis(20));
        assert_eq!(rec.missing_frames_at(now, 0, PAD_THRESHOLD_FRAMES), 0);
    }

    fn recorder_at_48k() -> Recorder {
        let rec = Recorder::new();
        rec.sample_rate.store(48_000, Ordering::Relaxed);
        rec
    }

    /// Spóźniony callback nie tworzy luki: porcje przychodzą później, ale ich czas przechwycenia
    /// idzie dalej bez przerw, więc nic nie jest dopełniane.
    #[test]
    fn late_callbacks_with_contiguous_capture_times_add_no_silence() {
        let rec = recorder_at_48k();
        for k in 0..10u32 {
            let at = StreamInstant::new(0, k * 10_000_000);
            assert_eq!(rec.gap_before(at, 480), 0, "porcja {k}");
        }
    }

    /// Prawdziwa cisza, w której nie było pakietów, wychodzi jako luka i jest dopełniona w swoim miejscu.
    #[test]
    fn a_real_silence_is_padded_where_it_happened() {
        let rec = recorder_at_48k();
        assert_eq!(rec.gap_before(StreamInstant::new(0, 0), 480), 0);
        // Pierwsza porcja kończy się po 10 ms, następna zaczyna się po 500 ms: luka 490 ms.
        assert_eq!(rec.gap_before(StreamInstant::new(0, 500_000_000), 480), 23_520);
    }

    /// Luki poniżej progu kumulują się, aż dojdą do 1 ms, i wtedy są dopełniane.
    #[test]
    fn sub_millisecond_gaps_accumulate_before_padding() {
        let rec = recorder_at_48k();
        assert_eq!(rec.gap_before(StreamInstant::new(0, 0), 480), 0);
        // Luka 0,5 ms = 24 ramki: jeszcze poniżej progu.
        assert_eq!(rec.gap_before(StreamInstant::new(0, 10_500_000), 480), 0);
        // Druga luka 0,5 ms: razem 48 ramek, dopełniamy.
        assert_eq!(rec.gap_before(StreamInstant::new(0, 21_000_000), 480), 48);
    }
}
