//! Rozpoznawanie kroków w strumieniu nagłych skoków energii (onsetów).
//!
//! Krok to uderzenie, które przychodzi w rytmie chodu: od poprzedniego
//! uderzenia mija od [`STEP_MIN_INTERVAL_SECS`] do [`STEP_MAX_INTERVAL_SECS`]
//! (bieg to ~3–4 kroki/s, powolne skradanie się ~1 krok/s). Pojedynczy strzał
//! czy wybuch zwykle nie ma takiego odstępu do poprzedniego zdarzenia, więc
//! nie zostanie uznany za krok. Szybsze serie (np. ogień z karabinu) odpadają,
//! bo odstępy między nimi są krótsze niż [`STEP_MIN_INTERVAL_SECS`].
//!
//! Moduł nie zna UI ani strumienia audio — operuje tylko na czasach uderzeń,
//! więc rytm da się przetestować syntetycznie (jak `sound::direction`).

/// Najkrótszy odstęp między kolejnymi krokami (szybki bieg).
pub const STEP_MIN_INTERVAL_SECS: f64 = 0.20;

/// Najdłuższy odstęp, który jeszcze jest krokiem (powolne skradanie się).
pub const STEP_MAX_INTERVAL_SECS: f64 = 1.00;

/// Odstęp, w którym kolejny skok uznajemy za ogon tego samego uderzenia, a nie
/// nowe zdarzenie. Onset potrafi być widoczny w kilku kolejnych ramkach UI —
/// bez tego jeden krok liczyłby się jako kilka uderzeń naraz.
pub const HIT_REFRACTORY_SECS: f64 = 0.12;

/// Śledzi rytm uderzeń (grup onsetów z jednej ramki) i mówi, które z nich
/// pasują do rytmu kroków.
#[derive(Debug, Default, Clone, Copy)]
pub struct StepTracker {
    /// Czas ostatniego uderzenia, które nie było ogonem (patrz [`HIT_REFRACTORY_SECS`]).
    last_hit_at: Option<f64>,
}

impl StepTracker {
    /// Zgłasza uderzenie w chwili `now` (sekundy, monotoniczne). Zwraca `true`,
    /// gdy odstęp od poprzedniego uderzenia mieści się w zakresie kroków.
    /// Pierwsze uderzenie zawsze zwraca `false` — bez poprzedniego nie da się
    /// ocenić rytmu. Ogon uderzenia (w [`HIT_REFRACTORY_SECS`]) jest ignorowany
    /// i nie przesuwa punktu odniesienia.
    pub fn observe_hit(&mut self, now: f64) -> bool {
        let gap = match self.last_hit_at {
            None => {
                self.last_hit_at = Some(now);
                return false;
            }
            Some(prev) => now - prev,
        };
        if gap < HIT_REFRACTORY_SECS {
            return false;
        }
        self.last_hit_at = Some(now);
        (STEP_MIN_INTERVAL_SECS..=STEP_MAX_INTERVAL_SECS).contains(&gap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_hit_is_never_a_step() {
        let mut tracker = StepTracker::default();
        assert!(!tracker.observe_hit(10.0));
    }

    #[test]
    fn steady_walking_rhythm_is_recognised_as_steps() {
        let mut tracker = StepTracker::default();
        assert!(!tracker.observe_hit(0.0));
        for k in 1..=8 {
            assert!(tracker.observe_hit(k as f64 * 0.5), "krok {k} w rytmie 0,5 s");
        }
    }

    #[test]
    fn running_rhythm_at_the_fast_end_still_counts() {
        let mut tracker = StepTracker::default();
        tracker.observe_hit(0.0);
        assert!(tracker.observe_hit(0.25));
        assert!(tracker.observe_hit(0.50));
    }

    #[test]
    fn single_shot_after_long_silence_is_not_a_step() {
        let mut tracker = StepTracker::default();
        tracker.observe_hit(0.0);
        assert!(!tracker.observe_hit(5.0), "odstęp 5 s to nie rytm kroków");
    }

    #[test]
    fn rapid_fire_faster_than_a_step_is_not_a_step() {
        let mut tracker = StepTracker::default();
        tracker.observe_hit(0.0);
        // 0,15 s między strzałami: powyżej ogona, ale poniżej najkrótszego kroku.
        assert!(!tracker.observe_hit(0.15));
        assert!(!tracker.observe_hit(0.30));
    }

    #[test]
    fn refractory_tail_does_not_break_the_rhythm() {
        let mut tracker = StepTracker::default();
        tracker.observe_hit(0.0);
        // Ogon tego samego uderzenia — ignorowany, punkt odniesienia zostaje na 0.0.
        assert!(!tracker.observe_hit(0.05));
        assert!(tracker.observe_hit(0.5), "odstęp liczony od prawdziwego uderzenia");
    }
}
