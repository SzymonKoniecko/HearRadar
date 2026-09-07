//! Gotowe profile kolorów dopasowane pod gry typu competetive (FPS/BR), żeby
//! szybciej odróżnić typowe "sytuacje" w miksie po samym kolorze kropki —
//! kroki, strzały, wybuchy, komunikację głosową itd.
//!
//! To wciąż tylko podział wg energii w pasmach częstotliwości (patrz
//! [`crate::sound::spectrum`]) — żaden profil nie "rozpoznaje" faktycznie, że
//! coś jest strzałem czy krokiem, tylko dobiera zakresy Hz tak, żeby typowe
//! dźwięki tych kategorii w danej grze zwykle w nie trafiały. Użytkownik może
//! traktować je jako punkt startowy i dalej edytować reguły ręcznie.

use crate::sound::spectrum::ColorRule;

/// Gotowy zestaw reguł kolorowania z nazwą i krótkim opisem do wyboru w UI.
pub struct ColorProfile {
    pub name: &'static str,
    pub description: &'static str,
    pub build: fn() -> Vec<ColorRule>,
}

/// Zwraca wszystkie wbudowane profile, w kolejności do wyświetlenia w UI.
pub fn built_in_profiles() -> &'static [ColorProfile] {
    &[
        ColorProfile {
            name: "Domyślny (3 pasma)",
            description: "Ogólny podział bas/średnie/wysokie — bezpieczny punkt startowy dla dowolnej gry.",
            build: default_profile,
        },
        ColorProfile {
            name: "FPS taktyczny (CS/Valorant-style)",
            description: "5 pasm rozróżniających kroki, komunikację głosową i strzały — pod gry z bliskim dystansem i szybkim tempem.",
            build: tactical_fps_profile,
        },
        ColorProfile {
            name: "Battle Royale (daleki dystans)",
            description: "Rozbite niskie częstotliwości (pojazdy/wybuchy vs kroki) — pod gry, gdzie źródła dźwięku bywają daleko i słabo słyszalne.",
            build: battle_royale_profile,
        },
        ColorProfile {
            name: "Wysoki kontrast (2 kolory)",
            description: "Tylko 'cicho/otoczenie' vs 'głośno/zagrożenie' — maksymalnie czytelne przy słabym słuchu lub dużym zmęczeniu wzroku.",
            build: high_contrast_profile,
        },
    ]
}

fn default_profile() -> Vec<ColorRule> {
    crate::sound::spectrum::default_color_rules()
}

/// FPS taktyczny: kroki i strzały to najważniejsze "sytuacje" do odróżnienia,
/// więc mają węższe, dedykowane pasma zamiast szerokiego "średnie/wysokie".
fn tactical_fps_profile() -> Vec<ColorRule> {
    vec![
        ColorRule::new(0.0, 150.0, [200, 60, 220]).with_label("Wybuchy / uderzenia"),
        ColorRule::new(150.0, 1000.0, [80, 220, 90]).with_label("Kroki / otoczenie"),
        ColorRule::new(1000.0, 3000.0, [0, 200, 220]).with_label("Mowa / komunikaty"),
        ColorRule::new(3000.0, 8000.0, [255, 120, 40]).with_label("Strzały / ostre dźwięki"),
        ColorRule::new(8000.0, 20_000.0, [255, 230, 60]).with_label("Przeładowanie / wysokie"),
    ]
}

/// Battle Royale: dystans do źródła dźwięku bywa duży, więc niskie
/// częstotliwości (pojazdy, wybuchy, kroki na różnych powierzchniach) są
/// rozbite drobniej niż w typowym profilu 3-pasmowym.
fn battle_royale_profile() -> Vec<ColorRule> {
    vec![
        ColorRule::new(0.0, 90.0, [180, 40, 200]).with_label("Pojazdy / eksplozje"),
        ColorRule::new(90.0, 300.0, [255, 80, 80]).with_label("Kroki bliskie"),
        ColorRule::new(300.0, 1200.0, [80, 220, 90]).with_label("Kroki dalekie / otoczenie"),
        ColorRule::new(1200.0, 5000.0, [255, 150, 30]).with_label("Strzały / echo"),
        ColorRule::new(5000.0, 20_000.0, [255, 230, 60]).with_label("Wysokie / cel"),
    ]
}

/// Wysoki kontrast: tylko dwie kategorie, maksymalnie czytelne bez potrzeby
/// rozróżniania odcieni — "spokojnie" vs "coś się dzieje".
fn high_contrast_profile() -> Vec<ColorRule> {
    vec![
        ColorRule::new(0.0, 1000.0, [90, 200, 255]).with_label("Otoczenie / ciche"),
        ColorRule::new(1000.0, 20_000.0, [255, 70, 70]).with_label("Zagrożenie / głośne"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_profile_has_at_least_one_rule() {
        for profile in built_in_profiles() {
            let rules = (profile.build)();
            assert!(!rules.is_empty(), "profil '{}' nie ma żadnych reguł", profile.name);
        }
    }

    #[test]
    fn every_profile_covers_the_full_audible_range_without_gaps() {
        for profile in built_in_profiles() {
            let mut rules = (profile.build)();
            rules.sort_by(|a, b| a.freq_min.partial_cmp(&b.freq_min).unwrap());
            assert_eq!(rules.first().unwrap().freq_min, 0.0, "profil '{}' nie zaczyna się od 0 Hz", profile.name);
            for pair in rules.windows(2) {
                assert_eq!(
                    pair[0].freq_max, pair[1].freq_min,
                    "profil '{}' ma lukę/nakładkę między regułami", profile.name
                );
            }
        }
    }

    #[test]
    fn profile_names_are_unique() {
        let profiles = built_in_profiles();
        for i in 0..profiles.len() {
            for j in (i + 1)..profiles.len() {
                assert_ne!(profiles[i].name, profiles[j].name);
            }
        }
    }
}
