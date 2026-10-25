//! Test kolejności kanałów: `radar --channel-test <nazwa urządzenia>`.
//!
//! Po kolei dla każdego indeksu kanału urządzenia gra sinus, a na pozostałych
//! kanałach ciszę. Równolegle nasłuchuje pętli zwrotnej tego samego urządzenia
//! (ten sam mechanizm co `--listen-cable`) i sprawdza, który indeks kanału
//! faktycznie dostał sygnał. Potwierdza to, że indeksy zgadzają się z tym, co
//! wysyła aplikacja.
//!
//! Nazwy głośników przy kanałach to oczekiwana kolejność wg maski kanałów
//! WASAPI. Tego test nie sprawdza automatycznie: trzeba to usłyszeć na
//! prawdziwych głośnikach albo ustalić w grze.
//!
//! Tryb `--monitor` to podgląd na żywo: pokazuje RMS każdego kanału pętli
//! zwrotnej, więc widać, na których indeksach gra wysyła dźwięk w danym trybie.

use anyhow::{Context, Result, anyhow, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::f32::consts::TAU;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::sound::devices::{SharedAudioAnalysis, fetch_audio_from_device};

const TEST_FREQ_HZ: f32 = 1000.0;
const TEST_AMPLITUDE: f32 = 0.3;
/// Jak długo gra każdy kanał.
const STEP: Duration = Duration::from_millis(1000);
/// Pauza po zakończeniu kanału, żeby jego ogon nie trafił do następnego pomiaru.
const PAUSE: Duration = Duration::from_millis(400);
/// Opóźnienie pętli zwrotnej: pierwsze ms po włączeniu kanału pomijamy.
const SETTLE: Duration = Duration::from_millis(150);
const READ_INTERVAL: Duration = Duration::from_millis(20);
/// Poniżej tego średniego RMS uznajemy, że na pętli zwrotnej nic nie słychać.
const SIGNAL_FLOOR: f32 = 0.01;
/// Wartość „cisza na wszystkich kanałach”.
const SILENT: usize = usize::MAX;
/// Odstęp między kolejnymi wierszami podglądu.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(250);
/// Od tego średniego RMS kanał uznajemy w podglądzie za aktywny.
const ACTIVE_FLOOR: f32 = 0.002;

/// Uruchamia test na pierwszym urządzeniu wyjściowym, którego nazwa zawiera
/// `device_name` (bez rozróżniania wielkości liter). Liczba kanałów to ta, którą
/// urządzenie ma ustawioną w Windows.
pub fn run(device_name: &str) -> Result<()> {
    let device = find_output_device(device_name)?;
    let name = device.name()?;
    let supported = device.default_output_config()?;
    let config: StreamConfig = supported.clone().into();
    let channels = config.channels as usize;
    let speakers = speaker_names(channels);

    println!("Urządzenie: {name}");
    println!("Kanałów: {channels}, {} Hz, {}", config.sample_rate.0, supported.sample_format());
    println!("Oczekiwana kolejność wg maski WASAPI: {}", speakers.join(", "));

    let active = Arc::new(AtomicUsize::new(SILENT));
    let _output = build_output(&device, &supported, active.clone()).context("budowa strumienia wyjściowego")?;
    _output.play().context("start strumienia wyjściowego")?;

    let analysis = SharedAudioAnalysis::new();
    let _loopback = match fetch_audio_from_device(&name, analysis.clone()) {
        Ok(stream) => Some(stream),
        Err(err) => {
            println!("Pętla zwrotna niedostępna ({err}), sprawdzasz tylko słuchem.");
            None
        }
    };

    let mut matched = 0;
    for (index, speaker) in speakers.iter().enumerate() {
        active.store(index, Ordering::Relaxed);
        std::thread::sleep(SETTLE);
        let levels = sample_levels(&analysis, STEP);
        active.store(SILENT, Ordering::Relaxed);
        std::thread::sleep(PAUSE);

        let result = verdict(index, &levels);
        if result.starts_with("pętla zwrotna: kanał") {
            matched += 1;
        }
        println!("[{index}] {speaker:>3}  → {result}");
    }

    println!();
    println!("Zgodnych z pętlą zwrotną: {matched}/{channels}");
    println!("Nazwy głośników to oczekiwana kolejność. Potwierdź ją słuchem na prawdziwych głośnikach.");
    Ok(())
}

/// Podgląd na żywo: co `PREVIEW_INTERVAL` wypisuje średni RMS każdego kanału
/// pętli zwrotnej i listę kanałów, na których coś słychać. Służy do sprawdzenia,
/// na których indeksach gra wysyła dźwięk w danym trybie (stereo, 4.1, 5.1, 7.1).
pub fn monitor(device_name: &str, duration: Duration) -> Result<()> {
    let device = find_output_device(device_name)?;
    let name = device.name()?;
    let channels = device.default_output_config()?.channels();
    let analysis = SharedAudioAnalysis::new();
    let _loopback = fetch_audio_from_device(&name, analysis.clone())?;

    println!("Podgląd: {name}, kanałów: {channels}. Graj i patrz, które indeksy się zapalają.");
    let start = Instant::now();
    while start.elapsed() < duration {
        let levels = sample_levels(&analysis, PREVIEW_INTERVAL);
        let active: Vec<usize> = levels
            .iter()
            .enumerate()
            .filter(|(_, level)| **level >= ACTIVE_FLOOR)
            .map(|(i, _)| i)
            .collect();
        let row: Vec<String> = levels.iter().map(|l| format!("{l:.3}")).collect();
        println!(
            "t={:5.1}s  [{}]  aktywne: {active:?}",
            start.elapsed().as_secs_f32(),
            row.join(" ")
        );
    }
    Ok(())
}

/// Pierwsze urządzenie wyjściowe, którego nazwa zawiera `device_name` (bez
/// rozróżniania wielkości liter).
pub(crate) fn find_output_device(device_name: &str) -> Result<cpal::Device> {
    let needle = device_name.to_lowercase();
    cpal::default_host()
        .output_devices()?
        .find(|d| d.name().map(|n| n.to_lowercase().contains(&needle)).unwrap_or(false))
        .ok_or_else(|| anyhow!("nie znaleziono urządzenia wyjściowego zawierającego '{device_name}'"))
}

/// Kolejność głośników wg maski kanałów WASAPI (kolejność bitów: FL, FR, FC, LFE,
/// BL, BR, SL, SR). Dla liczby kanałów bez znanej maski zwraca `?`.
pub fn speaker_names(channels: usize) -> Vec<&'static str> {
    match channels {
        1 => vec!["FC"],
        2 => vec!["FL", "FR"],
        4 => vec!["FL", "FR", "BL", "BR"],
        5 => vec!["FL", "FR", "LFE", "BL", "BR"],
        6 => vec!["FL", "FR", "FC", "LFE", "BL", "BR"],
        8 => vec!["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
        n => vec!["?"; n],
    }
}

/// Opis wyniku dla jednego kanału na podstawie średnich poziomów pętli zwrotnej.
fn verdict(index: usize, levels: &[f32]) -> String {
    if levels.iter().all(|&l| l < SIGNAL_FLOOR) {
        return "pętla zwrotna: brak sygnału".to_string();
    }
    let loudest = levels
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if loudest == index {
        format!("pętla zwrotna: kanał {index} ✓")
    } else {
        format!("ROZJAZD: pętla zwrotna pokazuje kanał {loudest}")
    }
}

/// Średni RMS każdego kanału z pętli zwrotnej w czasie `duration`.
fn sample_levels(analysis: &SharedAudioAnalysis, duration: Duration) -> Vec<f32> {
    // Wyzeruj poprzednie wartości: gdy pętla zwrotna milczy, nie chcemy starego poziomu.
    if let Ok(mut guard) = analysis.lock() {
        guard.channel_rms.iter_mut().for_each(|v| *v = 0.0);
    }
    let mut sums: Vec<f32> = Vec::new();
    let mut reads = 0usize;
    let start = Instant::now();
    while start.elapsed() < duration {
        std::thread::sleep(READ_INTERVAL);
        let Ok(guard) = analysis.lock() else { continue };
        if guard.channel_rms.is_empty() {
            continue;
        }
        if sums.len() != guard.channel_rms.len() {
            sums = vec![0.0; guard.channel_rms.len()];
        }
        for (sum, &rms) in sums.iter_mut().zip(&guard.channel_rms) {
            *sum += rms;
        }
        reads += 1;
    }
    if reads == 0 {
        return Vec::new();
    }
    sums.iter().map(|s| s / reads as f32).collect()
}

fn build_output(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    active: Arc<AtomicUsize>,
) -> Result<cpal::Stream> {
    let config: StreamConfig = supported.clone().into();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_sine_stream::<f32>(device, &config, active)?,
        SampleFormat::I16 => build_sine_stream::<i16>(device, &config, active)?,
        fmt => bail!("nieobsługiwany format próbek: {fmt}"),
    };
    Ok(stream)
}

fn build_sine_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    active: Arc<AtomicUsize>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let step = TAU * TEST_FREQ_HZ / config.sample_rate.0 as f32;
    let mut phase = 0.0f32;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let on = active.load(Ordering::Relaxed);
            for frame in data.chunks_mut(channels) {
                let value = phase.sin() * TEST_AMPLITUDE;
                phase = (phase + step) % TAU;
                for (ch, sample) in frame.iter_mut().enumerate() {
                    *sample = T::from_sample(if ch == on { value } else { 0.0 });
                }
            }
        },
        report_error,
        None,
    )?;
    Ok(stream)
}

fn report_error(err: cpal::StreamError) {
    eprintln!("błąd strumienia testowego: {err}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_names_follow_wasapi_bit_order_for_7_1() {
        assert_eq!(
            speaker_names(8),
            vec!["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"]
        );
    }

    #[test]
    fn lfe_comes_before_rear_speakers_in_4_1() {
        assert_eq!(speaker_names(5), vec!["FL", "FR", "LFE", "BL", "BR"]);
    }

    #[test]
    fn speaker_names_length_matches_channel_count() {
        for n in 1..=8 {
            assert_eq!(speaker_names(n).len(), n);
        }
    }

    #[test]
    fn verdict_accepts_loudest_channel_at_its_own_index() {
        assert_eq!(verdict(2, &[0.0, 0.0, 0.2, 0.0]), "pętla zwrotna: kanał 2 ✓");
    }

    #[test]
    fn verdict_flags_a_mismatch_with_the_loudest_channel() {
        assert_eq!(verdict(1, &[0.0, 0.0, 0.2, 0.0]), "ROZJAZD: pętla zwrotna pokazuje kanał 2");
    }

    #[test]
    fn verdict_reports_silence_on_loopback() {
        assert_eq!(verdict(0, &[0.0, 0.001]), "pętla zwrotna: brak sygnału");
    }
}
