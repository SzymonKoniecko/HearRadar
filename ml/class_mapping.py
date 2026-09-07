"""Mapowanie 521 klas AudioSet (używanych przez YAMNet) na małą liczbę
kategorii "sytuacji" pod gry competetive. To celowo proste dopasowanie po
słowach kluczowych w nazwie klasy AudioSet, nie osobny model — YAMNet i tak
już zwraca prawdopodobieństwo dla każdej z 521 klas, a to tylko grupuje je
w coś, co da się pokazać jako 8 kolorów zamiast 521.

Uwaga: to punkt startowy, nie precyzyjna taksonomia. Klasy AudioSet są
ogólne (np. "Explosion" obejmuje fajerwerki i wybuchy bomb jednakowo) —
w razie potrzeby dopisz więcej słów kluczowych albo przenieś pojedyncze
klasy między kategoriami ręcznie w `CATEGORY_KEYWORDS` poniżej.
"""

from __future__ import annotations

# Kolejność ma znaczenie: pierwsza pasująca kategoria wygrywa, więc bardziej
# specyficzne/ważne dla competetive gier kategorie są wyżej (np. "Strzały"
# przed ogólnym "Mechaniczne").
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Strzały/wybuchy": [
        "gunshot", "gunfire", "machine gun", "artillery", "explosion",
        "fireworks", "cap gun", "boom", "eruption",
    ],
    "Kroki": [
        "walk", "footsteps", "run",
    ],
    "Mowa": [
        "speech", "conversation", "narration", "monologue", "shout", "yell",
        "screaming", "babbling", "child speech", "male speech", "female speech",
        "whispering",
    ],
    "Pojazdy": [
        "vehicle", "car", "truck", "motor vehicle", "engine", "aircraft",
        "helicopter", "train", "boat", "motorcycle", "traffic",
    ],
    "Mechaniczne/przeładowanie": [
        "mechanisms", "clatter", "click", "clock", "typing", "keys jangling",
        "zipper", "cash register", "ratchet",
    ],
    "Muzyka/otoczenie": [
        "music", "ambient", "wind", "rain", "thunder", "nature", "outside",
        "environmental noise", "white noise",
    ],
    "Cisza": [
        "silence",
    ],
}

DEFAULT_CATEGORY = "Inne"


def categorize(display_name: str) -> str:
    """Zwraca kategorię dla nazwy klasy AudioSet (np. "Gunshot, gunfire")."""
    name = display_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return category
    return DEFAULT_CATEGORY


def all_categories() -> list[str]:
    """Wszystkie możliwe wartości zwracane przez `categorize`, w stałej
    kolejności — używane przez Rust do zbudowania domyślnej palety kolorów."""
    return list(CATEGORY_KEYWORDS.keys()) + [DEFAULT_CATEGORY]
