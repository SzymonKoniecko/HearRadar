//! Wideo do sesji `radar --recordVideo`. Obraz nagrywa zewnętrzny `ffmpeg`
//! (przechwytywanie przez gdigrab), a dźwięk idzie tą samą ścieżką co `--recordSample`.
//!
//! Dwa strumienie zapisują się osobno: `audio.wav` i `video.mp4`. Przesunięcie zapisujemy
//! w `session.json` jako `video_offset_s`: czas od startu dźwięku do pierwszej klatki wideo.
//! Pierwszą klatkę wykrywamy po raporcie `-progress` ffmpeg, z dokładnością ok. 0,1 s.
//!
//! Ograniczenia gdigrab: przechwytuje tylko to, co widać w oknie pulpitu. Gra w trybie
//! pełnoekranowym wyłącznym może dać czarny obraz. Przerwanie przez Ctrl+C może
//! uszkodzić MP4, dlatego kończymy nagranie przez `q` (patrz `VideoCapture::stop`).

use anyhow::{Context, Result, anyhow, bail};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// Nazwa pliku wideo w katalogu sesji.
pub const VIDEO_FILE: &str = "video.mp4";
/// Jak długo czekamy na pierwszą klatkę, zanim uznamy, że ffmpeg nie nagrywa.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);
/// Jak często ffmpeg raportuje postęp. Tyle wynosi dokładność pomiaru pierwszej klatki.
const PROGRESS_PERIOD_SECS: &str = "0.1";

/// Co nagrywamy: cały pulpit albo okno o podanym tytule.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoSource {
    Desktop,
    Window(String),
}

#[derive(Debug, Clone)]
pub struct VideoOptions {
    /// Ścieżka do ffmpeg albo nazwa z PATH.
    pub ffmpeg: String,
    pub source: VideoSource,
    pub fps: u32,
}

/// Sprawdza, czy ffmpeg da się uruchomić. Wołane przed startem sesji, żeby nie
/// zostawiać pustego katalogu, gdy narzędzia brakuje.
pub fn check_ffmpeg(ffmpeg: &str) -> Result<()> {
    let status = Command::new(ffmpeg)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| {
            format!("nie znaleziono '{ffmpeg}'. Zainstaluj ffmpeg albo podaj ścieżkę przez --ffmpeg")
        })?;
    if !status.success() {
        bail!("'{ffmpeg} -version' zakończył się błędem: {status}");
    }
    Ok(())
}

/// Argumenty ffmpeg: przechwytywanie gdigrab, kodowanie H.264 do MP4, raport postępu na stdout.
pub fn ffmpeg_args(output: &Path, source: &VideoSource, fps: u32) -> Vec<String> {
    let input = match source {
        VideoSource::Desktop => "desktop".to_string(),
        VideoSource::Window(title) => format!("title={title}"),
    };
    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-f".into(),
        "gdigrab".into(),
        "-framerate".into(),
        fps.to_string(),
        "-i".into(),
        input,
        "-c:v".into(),
        "libx264".into(),
        "-preset".into(),
        "veryfast".into(),
        "-crf".into(),
        "23".into(),
        "-pix_fmt".into(),
        "yuv420p".into(),
        // Bez tego x264 trzyma klatki w buforze wyprzedzającym (kilka sekund przy 22 fps),
        // więc pierwsza zakodowana klatka pojawia się w raporcie za późno.
        "-tune".into(),
        "zerolatency".into(),
        "-progress".into(),
        "pipe:1".into(),
        "-stats_period".into(),
        PROGRESS_PERIOD_SECS.into(),
    ];
    args.push(output.display().to_string());
    args
}

/// Liczba klatek z linii raportu `-progress` (`frame=12`). Inne linie dają `None`.
pub fn parse_frame_count(line: &str) -> Option<u64> {
    line.strip_prefix("frame=")?.trim().parse().ok()
}

/// Działające nagrywanie wideo. Kończy się przez `stop` albo przerywa przez `abort`.
pub struct VideoCapture {
    child: Child,
    /// Dostaje jedną wiadomość: chwilę, w której zauważono pierwszą klatkę.
    first_frame: Receiver<Instant>,
}

impl VideoCapture {
    pub fn start(options: &VideoOptions, output: &Path) -> Result<Self> {
        let mut child = Command::new(&options.ffmpeg)
            .args(ffmpeg_args(output, &options.source, options.fps))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("uruchomienie '{}' nie powiodło się", options.ffmpeg))?;
        let stdout = child.stdout.take().context("brak wyjścia ffmpeg")?;

        // Czytamy raport aż do końca strumienia, żeby ffmpeg nie zablokował się na pełnym potoku.
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reported = false;
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if !reported && parse_frame_count(&line).is_some_and(|n| n > 0) {
                    reported = true;
                    let _ = tx.send(Instant::now());
                }
            }
        });
        Ok(Self { child, first_frame: rx })
    }

    /// Czeka na pierwszą klatkę i zwraca chwilę, w której ją zauważono.
    /// Błąd, gdy ffmpeg zakończy się przed klatką albo nie zacznie w czasie `FIRST_FRAME_TIMEOUT`.
    pub fn wait_first_frame(&self) -> Result<Instant> {
        self.first_frame
            .recv_timeout(FIRST_FRAME_TIMEOUT)
            .map_err(|err| match err {
                RecvTimeoutError::Timeout => anyhow!(
                    "ffmpeg nie zaczął nagrywać w ciągu {} s", FIRST_FRAME_TIMEOUT.as_secs()
                ),
                RecvTimeoutError::Disconnected => {
                    anyhow!("ffmpeg zakończył się przed pierwszą klatką; sprawdź --window")
                }
            })
    }

    /// Kończy nagrywanie przez `q` na stdin, żeby ffmpeg domknął plik MP4.
    pub fn stop(mut self) -> Result<()> {
        if let Some(mut stdin) = self.child.stdin.take() {
            let _ = stdin.write_all(b"q\n");
        }
        let status = self.child.wait()?;
        if !status.success() {
            bail!("ffmpeg zakończył się błędem: {status}");
        }
        Ok(())
    }

    /// Przerywa ffmpeg bez domykania pliku. Używane, gdy sesja się nie udała.
    pub fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_capture_uses_gdigrab_and_writes_to_the_output_file() {
        let args = ffmpeg_args(Path::new("out.mp4"), &VideoSource::Desktop, 30);
        assert!(args.windows(2).any(|w| w == ["-f", "gdigrab"]));
        assert!(args.windows(2).any(|w| w == ["-i", "desktop"]));
        assert!(args.windows(2).any(|w| w == ["-framerate", "30"]));
        assert_eq!(args.last().map(String::as_str), Some("out.mp4"));
    }

    #[test]
    fn window_capture_passes_the_title_as_one_argument() {
        let source = VideoSource::Window("Gra X".to_string());
        let args = ffmpeg_args(Path::new("out.mp4"), &source, 25);
        assert!(args.windows(2).any(|w| w == ["-i", "title=Gra X"]));
        assert!(args.windows(2).any(|w| w == ["-framerate", "25"]));
    }

    #[test]
    fn encoder_runs_without_lookahead_so_the_first_frame_is_reported_early() {
        let args = ffmpeg_args(Path::new("out.mp4"), &VideoSource::Desktop, 30);
        assert!(args.windows(2).any(|w| w == ["-tune", "zerolatency"]));
    }

    #[test]
    fn progress_report_goes_to_stdout_with_a_short_period() {
        let args = ffmpeg_args(Path::new("out.mp4"), &VideoSource::Desktop, 30);
        assert!(args.windows(2).any(|w| w == ["-progress", "pipe:1"]));
        assert!(args.windows(2).any(|w| w == ["-stats_period", "0.1"]));
    }

    #[test]
    fn frame_count_is_read_only_from_frame_lines() {
        assert_eq!(parse_frame_count("frame=0"), Some(0));
        assert_eq!(parse_frame_count("frame=42"), Some(42));
        assert_eq!(parse_frame_count("frame= 7"), Some(7));
        assert_eq!(parse_frame_count("fps=0.00"), None);
        assert_eq!(parse_frame_count("progress=continue"), None);
    }

    #[test]
    fn missing_ffmpeg_is_reported_with_a_hint() {
        let err = check_ffmpeg("radar-nie-ma-takiego-programu-xyz").unwrap_err();
        assert!(err.to_string().contains("--ffmpeg"), "{err}");
    }
}
