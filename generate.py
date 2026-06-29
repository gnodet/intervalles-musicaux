#!/usr/bin/env python3
"""
Generateur de fichiers MP3 pour l'apprentissage des intervalles musicaux.

Chaque fichier contient les premieres notes d'une melodie connue,
servant de reference mnemotechnique pour reconnaitre un intervalle.

Dependances :
    pip install numpy
    ffmpeg (pour la conversion WAV -> MP3)

Usage :
    python generate.py [--output-dir audio/]
"""

import argparse
import json
import os
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 44100

# --- Note helpers ---

NOTES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def note(name):
    """Convert note name to MIDI number. E.g. 'C4' -> 60, 'A3' -> 57."""
    pitch = name[:-1]
    octave = int(name[-1])
    return 12 * (octave + 1) + NOTES[pitch]


def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


# --- Audio synthesis ---

def generate_tone(freq, duration, volume=0.6):
    """Synthesize a piano-like tone (harmonics + decay)."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    w = (
        volume * 0.60 * np.sin(2 * np.pi * freq * t)
        + volume * 0.25 * np.sin(2 * np.pi * 2 * freq * t)
        + volume * 0.10 * np.sin(2 * np.pi * 3 * freq * t)
        + volume * 0.05 * np.sin(2 * np.pi * 4 * freq * t)
    )
    # Fade in / out (30 ms)
    fade = int(SAMPLE_RATE * 0.03)
    if 0 < fade < len(w) // 2:
        w[:fade] *= np.linspace(0, 1, fade)
        w[-fade:] *= np.linspace(1, 0, fade)
    # Exponential decay (slower for longer notes)
    decay_rate = max(1.5, 3.0 - duration)
    w *= np.exp(-t * decay_rate)
    return w


def render_melody(notes_list, tempo_bpm=120):
    """Render a sequence of (midi_note|None, beats) into audio samples."""
    beat_dur = 60.0 / tempo_bpm
    parts = []
    for midi_note, beats in notes_list:
        dur = beats * beat_dur
        target_len = int(SAMPLE_RATE * dur)
        if midi_note is None:
            parts.append(np.zeros(target_len))
        else:
            tone = generate_tone(midi_to_freq(midi_note), dur)
            if len(tone) < target_len:
                tone = np.concatenate([tone, np.zeros(target_len - len(tone))])
            else:
                tone = tone[:target_len]
            parts.append(tone)
    audio = np.concatenate(parts)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def save_mp3(audio, path):
    """Write audio to WAV then convert to MP3 via ffmpeg."""
    wav_path = path.replace(".mp3", ".wav")
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
         "-b:a", "128k", "-ar", "44100", path],
        capture_output=True, check=True,
    )
    os.remove(wav_path)


# --- Melody definitions ---
# Each entry: (midi_note, duration_in_beats)
# None = silence/rest

N = note

MELODIES = [
    # ================================================================
    #                    ASCENDING INTERVALS
    # ================================================================

    # 1. Seconde mineure (1 demi-ton) - Jaws (Les Dents de la mer)
    # E-F repeated, accelerating - the menacing shark motif
    {
        "id": "01", "interval": "Seconde mineure", "semitones": 1,
        "direction": "asc", "song": "Les Dents de la mer (Jaws)",
        "notes_str": "E2\u2192F2 (accelerando)",
        "tempo": 120,
        "notes": [
            (N("E2"), 1), (N("F2"), 1),
            (N("E2"), 0.75), (N("F2"), 0.75),
            (N("E2"), 0.5), (N("F2"), 0.5),
            (N("E2"), 0.35), (N("F2"), 0.35),
            (N("E2"), 0.25), (N("F2"), 0.25),
            (N("E2"), 0.2), (N("F2"), 0.2),
        ],
    },

    # 2. Seconde majeure (2 demi-tons) - Frere Jacques
    # C-D-E-C, C-D-E-C (do-re-mi-do)
    {
        "id": "02", "interval": "Seconde majeure", "semitones": 2,
        "direction": "asc", "song": "Fr\u00e8re Jacques",
        "notes_str": "Do\u2192R\u00e9\u2192Mi\u2192Do",
        "tempo": 120,
        "notes": [
            (N("C4"), 1), (N("D4"), 1), (N("E4"), 1), (N("C4"), 1),
            (N("C4"), 1), (N("D4"), 1), (N("E4"), 1), (N("C4"), 1),
        ],
    },

    # 3. Tierce mineure (3 demi-tons) - Greensleeves
    # A3 (pickup) -> C4 (minor 3rd up), then D4, E4, F4, E4, D4, B3...
    # 3/4 time, tempo ~140
    {
        "id": "03", "interval": "Tierce mineure", "semitones": 3,
        "direction": "asc", "song": "Greensleeves",
        "notes_str": "La\u2192Do (A3\u2192C4)",
        "tempo": 140,
        "notes": [
            (N("A3"), 1),               # pickup: A3
            (N("C4"), 2), (N("D4"), 1), # bar 1
            (N("E4"), 1.5), (N("F4"), 0.5), (N("E4"), 1),  # bar 2
            (N("D4"), 2), (N("B3"), 1), # bar 3
            (N("G3"), 1.5), (N("A3"), 0.5), (N("B3"), 1),  # bar 4
            (N("C4"), 2), (N("A3"), 1), # bar 5
            (N("A3"), 1.5), (N("G#3"), 0.5), (N("A3"), 1), # bar 6
            (N("B3"), 2), (N("G#3"), 1),# bar 7
            (N("E3"), 3),               # bar 8
        ],
    },

    # 4. Tierce majeure (4 demi-tons) - Oh When the Saints
    # C-E-F-G (do-mi-fa-sol), held, then repeat
    {
        "id": "04", "interval": "Tierce majeure", "semitones": 4,
        "direction": "asc", "song": "Oh When the Saints",
        "notes_str": "Do\u2192Mi\u2192Fa\u2192Sol",
        "tempo": 112,
        "notes": [
            (N("C4"), 1), (N("E4"), 1), (N("F4"), 1), (N("G4"), 3),
            (N("C4"), 1), (N("E4"), 1), (N("F4"), 1), (N("G4"), 3),
        ],
    },

    # 5. Quarte juste (5 demi-tons) - La Marseillaise
    # Guillaume: "do-do-do-fa" (3 do puis le saut)
    # In C: C4, C4, C4, F4 then continuation
    # Rythme martial 4/4: noire, noire, blanche, blanche, blanche, blanche
    # "Al-lons en-FANTS de la pa-tri-i-e"
    {
        "id": "05", "interval": "Quarte juste", "semitones": 5,
        "direction": "asc", "song": "La Marseillaise",
        "notes_str": "Do\u2192Do\u2192Do\u2192Fa (Al-lons en-fants)",
        "tempo": 100,
        "notes": [
            # Al-lons en-FANTS de la pa-TRI-e (en sol majeur)
            # re(16) re(8.) re(16) sol(4) sol(4) la(4) la(4) re'(2)
            (N("D4"), 0.25),   # Al-    (double croche)
            (N("D4"), 0.75),   # -lons  (croche pointee)
            (N("D4"), 0.25),   # en-    (double croche)
            (N("G4"), 1),      # -FANTS (noire, quarte juste!)
            (N("G4"), 1),      # de     (noire)
            (N("A4"), 1),      # la     (noire)
            (N("A4"), 1),      # pa-    (noire)
            (N("D5"), 2),      # -TRI-e (blanche)
        ],
    },

    # 6. Triton (6 demi-tons) - The Simpsons
    # Vocal "The Simp-sons!": C5 (pickup), C5 -> F#4 (descending tritone)
    # But for ascending tritone mnemonic: use the instrumental intro
    # C-E-F#-A ascending line, tempo ~160
    # Actually the choral part is more recognizable:
    # "The" C5 (8th), "Simp-" C5 (quarter), "-sons!" F#4 (half)
    # This is a DESCENDING tritone C5->F#4
    # For ascending: the opening instrumental C4-E4-F#4
    {
        "id": "06", "interval": "Triton", "semitones": 6,
        "direction": "asc", "song": "Les Simpson (The Simpsons)",
        "notes_str": "Do\u2192Mi\u2192Fa# (intro instrumental)",
        "tempo": 160,
        "notes": [
            # Intro: C - F# - G (the tritone is C to F#)
            (N("C4"), 1.5),   # C
            (N("F#4"), 1.5),  # F# (tritone!)
            (N("G4"), 2),     # G
        ],
    },

    # 7. Quinte juste (7 demi-tons) - Star Wars (theme principal)
    # Pickup triplet G-G-G then C (4th up), then G (5th up from C)
    # The key interval: C4 -> G4
    {
        "id": "07", "interval": "Quinte juste", "semitones": 7,
        "direction": "asc", "song": "Star Wars (th\u00e8me principal)",
        "notes_str": "Sol\u2192Sol\u2192Sol\u2192Do\u2192Sol",
        "tempo": 108,
        "notes": [
            (N("G3"), 0.33), (N("G3"), 0.33), (N("G3"), 0.34),  # triplet pickup
            (N("C4"), 2),     # C (4th up)
            (N("G4"), 2),     # G (5th up from C - the key interval!)
            (N("F4"), 0.33), (N("E4"), 0.33), (N("D4"), 0.34),
            (N("C5"), 2),     # high C
        ],
    },

    # 8. Sixte mineure (8 demi-tons) - Le Parrain (The Godfather)
    # Opening theme: A3 -> F4 (minor 6th up)
    # Haunting waltz feel, 3/4
    {
        "id": "08", "interval": "Sixte mineure", "semitones": 8,
        "direction": "asc", "song": "Le Parrain (The Godfather)",
        "notes_str": "La\u2192Fa (A3\u2192F4)",
        "tempo": 100,
        "notes": [
            (N("A3"), 2),     # A3 (start)
            (N("F4"), 1),     # F4 (minor 6th up!)
            (N("E4"), 2),     # E4
            (N("F4"), 1),     # F4
            (N("A4"), 2),     # A4
            (N("F4"), 1),     # F4
            (N("E4"), 1.5),   # E4
            (N("D4"), 0.5),   # D4
            (N("C4"), 2),     # C4
        ],
    },

    # 9. Sixte majeure (9 demi-tons) - My Bonnie Lies Over the Ocean
    # D4 (pickup "My") -> B4 ("Bon-nie") = major 6th up
    # 3/4 waltz
    {
        "id": "09", "interval": "Sixte majeure", "semitones": 9,
        "direction": "asc", "song": "My Bonnie Lies Over the Ocean",
        "notes_str": "R\u00e9\u2192Si (D4\u2192B4) \u00ab My Bon- \u00bb",
        "tempo": 112,
        "notes": [
            (N("D4"), 1),     # "My" (pickup)
            (N("B4"), 3),     # "Bon-" (dotted half, full bar)
            (N("A4"), 1),     # "-nie"
            (N("G4"), 1),     # "lies"
            (N("A4"), 1),     # "o-"
            (N("G4"), 1),     # "-ver"
            (N("E4"), 1),     # "the"
            (N("G4"), 2),     # "o-"
            (N("E4"), 1),     # "-cean"
        ],
    },

    # 10. Septieme mineure (10 demi-tons) - Star Trek (theme)
    # Bb3 -> Ab4 (minor 7th up)
    {
        "id": "10", "interval": "Septi\u00e8me mineure", "semitones": 10,
        "direction": "asc", "song": "Star Trek (th\u00e8me)",
        "notes_str": "Sib\u2192Lab (Bb3\u2192Ab4)",
        "tempo": 80,
        "notes": [
            (N("Bb3"), 3),    # Bb3 (long, dramatic)
            (N("Ab4"), 4),    # Ab4 (minor 7th up, held)
            (N("A4"), 1),     # A4
            (N("Bb4"), 3),    # Bb4
        ],
    },

    # 11. Septieme majeure (11 demi-tons) - Take On Me (a-ha)
    # Synth riff at 169 BPM, all 8th notes:
    # F#5 F#5 D5 B4 r B4 r E5 | r E5 r G#5 G#5 A5 B5 A5
    {
        "id": "11", "interval": "Septi\u00e8me majeure", "semitones": 11,
        "direction": "asc", "song": "Take On Me (a-ha)",
        "notes_str": "F#5\u2192F#5\u2192D5\u2192B4\u2192E5\u2192G#5\u2192A5\u2192B5",
        "tempo": 169,
        "notes": [
            # Bar 1
            (N("F#5"), 0.5), (N("F#5"), 0.5), (N("D5"), 0.5), (N("B4"), 0.5),
            (None, 0.5), (N("B4"), 0.5),
            (None, 0.5), (N("E5"), 0.5),
            # Bar 2
            (None, 0.5), (N("E5"), 0.5),
            (None, 0.5), (N("G#5"), 0.5),
            (N("G#5"), 0.5), (N("A5"), 0.5), (N("B5"), 0.5), (N("A5"), 0.5),
        ],
    },

    # 12. Octave (12 demi-tons) - Somewhere Over the Rainbow
    # Ab3 (half) -> Ab4 (half) then G4, Eb4, F4, G4, F4
    # Key of Ab major
    {
        "id": "12", "interval": "Octave", "semitones": 12,
        "direction": "asc", "song": "Somewhere Over the Rainbow",
        "notes_str": "Lab\u2192Lab (Ab3\u2192Ab4) \u00ab Some-where \u00bb",
        "tempo": 88,
        "notes": [
            (N("Ab3"), 2),    # "Some-" (half note)
            (N("Ab4"), 2),    # "-where" (half note, octave!)
            (N("G4"), 1),     # "o-"
            (N("Eb4"), 0.5),  # "-ver"
            (N("F4"), 0.5),   # "the"
            (N("G4"), 1),     # "rain-"
            (N("F4"), 1),     # "-bow"
        ],
    },

    # ================================================================
    #                   DESCENDING INTERVALS
    # ================================================================

    # 13. Seconde mineure desc (1 demi-ton) - Fur Elise
    # E5-D#5-E5-D#5-E5-B4-D5-C5-A4
    {
        "id": "13", "interval": "Seconde mineure", "semitones": 1,
        "direction": "desc", "song": "La Lettre \u00e0 \u00c9lise (F\u00fcr Elise)",
        "notes_str": "Mi\u2192R\u00e9#\u2192Mi\u2192R\u00e9#\u2192Mi\u2192Si\u2192R\u00e9\u2192Do\u2192La",
        "tempo": 140,
        "notes": [
            (N("E5"), 0.5), (N("D#5"), 0.5), (N("E5"), 0.5),
            (N("D#5"), 0.5), (N("E5"), 0.5),
            (N("B4"), 0.5), (N("D5"), 0.5), (N("C5"), 0.5),
            (N("A4"), 1.5),
        ],
    },

    # 14. Seconde majeure desc (2 demi-tons) - Yesterday (Beatles)
    # G4 -> F4 on "Yes-ter-" then continuing down
    # Key of F major, moderate tempo
    {
        "id": "14", "interval": "Seconde majeure", "semitones": 2,
        "direction": "desc", "song": "Yesterday (Beatles)",
        "notes_str": "Sol\u2192Fa (G4\u2192F4) \u00ab Yes-ter-day \u00bb",
        "tempo": 96,
        "notes": [
            (N("G4"), 1),     # "Yes-"
            (N("F4"), 0.5),   # "-ter-"
            (N("E4"), 0.5),   # "-day"
            (N("F4"), 1),     # "all"
            (N("A4"), 0.5),   # "my"
            (N("C5"), 1),     # "trou-"
            (N("Bb4"), 0.5),  # "-bles"
            (N("A4"), 0.5),   # "seemed"
            (N("G4"), 0.5),   # "so"
            (N("F4"), 1),     # "far"
            (N("G4"), 0.5),   # "a-"
            (N("A4"), 2),     # "-way"
        ],
    },

    # 15. Tierce mineure desc (3 demi-tons) - Hey Jude (Beatles)
    # C5 -> A4 on "Hey Jude" (descending minor 3rd)
    # Key of F major, 4/4
    {
        "id": "15", "interval": "Tierce mineure", "semitones": 3,
        "direction": "desc", "song": "Hey Jude (Beatles)",
        "notes_str": "Do\u2192La (C5\u2192A4) \u00ab Hey Jude \u00bb",
        "tempo": 76,
        "notes": [
            (N("C5"), 2),     # "Hey" (half note)
            (N("A4"), 2),     # "Jude" (half note, minor 3rd down!)
            (N("A4"), 0.5),   # "don't"
            (N("Bb4"), 0.5),  # "make"
            (N("F4"), 0.5),   # "it"
            (N("A4"), 1.5),   # "bad"
        ],
    },

    # 16. Tierce majeure desc (4 demi-tons) - Coucou !
    # The universal cuckoo call: G4 -> E4
    {
        "id": "16", "interval": "Tierce majeure", "semitones": 4,
        "direction": "desc", "song": "Coucou !",
        "notes_str": "Sol\u2192Mi (G4\u2192E4)",
        "tempo": 100,
        "notes": [
            (N("G4"), 1),     # "Cou-"
            (N("E4"), 1.5),   # "-cou!"
            (None, 0.5),
            (N("G4"), 1),     # "Cou-"
            (N("E4"), 1.5),   # "-cou!"
            (None, 0.5),
            (N("G4"), 1),     # "Cou-"
            (N("E4"), 2),     # "-cou!"
        ],
    },

    # 17. Quarte juste desc (5 demi-tons) - Carillon de Westminster (Big Ben)
    # The famous Westminster Chimes first phrase
    # E4-B3 is the descending perfect 4th
    # Full first phrase: E4, G#4, F#4, B3 (variation I)
    # Actually the most recognizable pattern:
    # G#4, F#4, E4, B3 (descending)
    {
        "id": "17", "interval": "Quarte juste", "semitones": 5,
        "direction": "desc", "song": "Carillon de Westminster (Big Ben)",
        "notes_str": "Mi\u2192Si (E4\u2192B3)",
        "tempo": 60,
        "notes": [
            # Westminster Chimes - first sequence
            (N("E4"), 2), (N("C4"), 2), (N("D4"), 2), (N("G3"), 3),
            (None, 1),
            # Second sequence
            (N("D4"), 2), (N("E4"), 2), (N("C4"), 2), (N("G3"), 3),
        ],
    },

    # 18. Triton desc (6 demi-tons) - Danse Macabre (Saint-Saens)
    # Violin scordatura: A5 -> Eb5 (descending tritone)
    # The famous opening solo violin double-stop
    {
        "id": "18", "interval": "Triton", "semitones": 6,
        "direction": "desc", "song": "Danse Macabre (Saint-Sa\u00ebns)",
        "notes_str": "La\u2192Mib (A5\u2192Eb5)",
        "tempo": 76,
        "notes": [
            # The violin plays the tritone repeatedly
            (N("A5"), 1.5), (N("Eb5"), 1.5),
            (None, 0.5),
            (N("A5"), 1.5), (N("Eb5"), 1.5),
            (None, 0.5),
            (N("A5"), 1), (N("Eb5"), 1),
            (N("A5"), 1), (N("Eb5"), 1),
            (N("A5"), 0.75), (N("Eb5"), 0.75),
            (N("A5"), 0.75), (N("Eb5"), 0.75),
        ],
    },

    # 19. Quinte juste desc (7 demi-tons) - Feelings (Morris Albert)
    # E4 -> A3 on "Feel-ings" (descending perfect 5th)
    {
        "id": "19", "interval": "Quinte juste", "semitones": 7,
        "direction": "desc", "song": "Feelings (Morris Albert)",
        "notes_str": "Mi\u2192La (E4\u2192A3) \u00ab Feel-ings \u00bb",
        "tempo": 80,
        "notes": [
            (N("E4"), 2),     # "Feel-"
            (N("A3"), 2),     # "-ings" (5th down!)
            (None, 0.5),
            (N("B3"), 1),     # "no-"
            (N("C4"), 0.5),   # "-thing"
            (N("D4"), 0.5),   # "more"
            (N("E4"), 0.5),   # "than"
            (N("C4"), 2),     # "feel-"
            (N("A3"), 2),     # "-ings"
        ],
    },

    # 20. Sixte mineure desc (8 demi-tons) - Love Story (Francis Lai)
    # Using the theme: the descending leap in the melody
    # Actually let's use "Un homme et une femme" or keep Le Parrain reversed
    # Better: the NBC chime pattern or...
    # Let's use "Go Down Moses" - "Let my people go"
    # E4 -> G#3 on "Go down" - that's a minor 6th down
    # Actually simpler: Beethoven's 5th, 3rd movement theme
    # Eb5 -> G4 is a minor 6th down
    # Most recognizable: "Where Do I Begin" (Love Story)
    # A4 -> C#4 ? No...
    # Let's keep it simple with La 5e de Beethoven opening:
    # G-G-G-Eb (desc minor 3rd), but later theme has 6ths
    # Best bet: Theme from "Doctor Zhivago" / "Lara's Theme"
    # Or: simply use "The Entertainer" (Scott Joplin) opening
    # D5 -> F#4 ? Not quite...
    # Going with: "Lullaby" (Brahms) - Bonsoir bel ange
    # Or the Chopin Marche Funebre opening: Bb3 -> D3 (minor 6th down)
    # Actually that's well known! The funeral march.
    {
        "id": "20", "interval": "Sixte mineure", "semitones": 8,
        "direction": "desc", "song": "Marche fun\u00e8bre (Chopin)",
        "notes_str": "Sib\u2192R\u00e9 (Bb3\u2192D3)",
        "tempo": 52,
        "notes": [
            # Funeral march - recognizable opening
            (N("Bb3"), 1.5),  # dotted quarter
            (N("Bb3"), 0.5),  # eighth
            (N("Bb3"), 2),    # half
            (N("Bb3"), 1.5),
            (N("Bb3"), 0.5),
            (N("Bb3"), 1),
            (N("D3"), 1),     # minor 6th down
            (N("Eb3"), 1),
            (N("F3"), 1),
            (N("Gb3"), 2),
        ],
    },

    # 21. Sixte majeure desc (9 demi-tons) - Nobody Knows
    # The spiritual "Nobody Knows the Trouble I've Seen"
    # G4 -> Bb3 on "No-body" (desc major 6th)
    # Well known in choral tradition
    {
        "id": "21", "interval": "Sixte majeure", "semitones": 9,
        "direction": "desc", "song": "Nobody Knows the Trouble I've Seen",
        "notes_str": "Sol\u2192Sib (G4\u2192Bb3) \u00ab No-body \u00bb",
        "tempo": 72,
        "notes": [
            (N("G4"), 2),     # "No-"
            (N("Bb3"), 1),    # "-bo-" (major 6th down!)
            (N("Bb3"), 1),    # "-dy"
            (N("Bb3"), 1),    # "knows"
            (N("C4"), 0.5),   # "the"
            (N("D4"), 1.5),   # "trou-"
            (N("Eb4"), 1),    # "-ble"
            (N("F4"), 1),     # "I've"
            (N("G4"), 2),     # "seen"
        ],
    },

    # 22. Septieme mineure desc (10 demi-tons)
    # Using the opening of "Watermelon Man" (Herbie Hancock)
    # Or better for French audience: the interval in the bass of many pieces
    # Actually: "An American in Paris" first blues note
    # Let's use something more universally known:
    # The first two notes of the riff from "Smoke on the Water" are not a 7th
    # Let's try: descending m7 = inversion of M2 up
    # "Emmanuelle" theme? Not clear enough opening
    # Going with Watermelon Man: riff starts C5 -> D4 (minor 7th down)
    {
        "id": "22", "interval": "Septi\u00e8me mineure", "semitones": 10,
        "direction": "desc", "song": "Watermelon Man (Herbie Hancock)",
        "notes_str": "Do\u2192R\u00e9 (C5\u2192D4)",
        "tempo": 112,
        "notes": [
            # Funky bass riff
            (N("C5"), 0.5),
            (N("D4"), 0.5),   # minor 7th down!
            (N("F4"), 0.5),
            (N("G4"), 0.5),
            (N("Ab4"), 1),
            (None, 0.5),
            (N("C5"), 0.5),
            (N("D4"), 0.5),
            (N("F4"), 0.5),
            (N("G4"), 0.5),
            (N("Ab4"), 1),
        ],
    },

    # 23. Septieme majeure desc (11 demi-tons)
    # "I Love You" (Cole Porter) - standard mnemonic
    # C5 -> C#4/Db4 on "I Love"
    # Known through jazz standards in France
    {
        "id": "23", "interval": "Septi\u00e8me majeure", "semitones": 11,
        "direction": "desc", "song": "I Love You (Cole Porter)",
        "notes_str": "Do\u2192R\u00e9b (C5\u2192Db4) \u00ab I Love \u00bb",
        "tempo": 120,
        "notes": [
            (N("C5"), 1.5),   # "I"
            (N("Db4"), 2),    # "love" (major 7th down!)
            (N("C4"), 0.5),   # "you"
            (None, 0.5),
            (N("F4"), 1),
            (N("E4"), 1),
            (N("D4"), 1),
            (N("C4"), 2),
        ],
    },

    # 24. Octave desc (12 demi-tons) - Mon Dieu (Edith Piaf)
    # The dramatic "Mon Dieu!" at the climax
    # Bb4 -> Bb3 (descending octave)
    {
        "id": "24", "interval": "Octave", "semitones": 12,
        "direction": "desc", "song": "Mon Dieu (\u00c9dith Piaf)",
        "notes_str": "Sib\u2192Sib (Bb4\u2192Bb3) \u00ab Mon Dieu \u00bb",
        "tempo": 72,
        "notes": [
            (N("Bb4"), 2),    # "Mon"
            (N("Bb3"), 3),    # "Dieu!" (octave down, dramatic!)
            (None, 1),
            (N("C4"), 1),     # continuation
            (N("D4"), 1),
            (N("Eb4"), 1),
            (N("F4"), 2),
        ],
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="audio",
                        help="Output directory for MP3 files")
    parser.add_argument("--manifest", default=None,
                        help="Write JSON manifest for the HTML page")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = []
    for m in MELODIES:
        direction = "asc" if m["direction"] == "asc" else "desc"
        symbol = "\u2191" if direction == "asc" else "\u2193"
        filename = f"{m['id']}_{direction}_{m['interval'].replace(' ', '_').replace('\u00e8', 'e')}.mp3"
        filepath = os.path.join(args.output_dir, filename)

        audio = render_melody(m["notes"], m["tempo"])
        save_mp3(audio, filepath)

        manifest.append({
            "id": m["id"],
            "interval": m["interval"],
            "semitones": m["semitones"],
            "direction": m["direction"],
            "direction_symbol": symbol,
            "song": m["song"],
            "notes": m["notes_str"],
            "file": filename,
        })
        print(f"  {symbol} {m['interval']:20s} -- {m['song']}")

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n  Manifest written to {args.manifest}")

    print(f"\n  {len(manifest)} fichiers generes dans {args.output_dir}/")


if __name__ == "__main__":
    main()
