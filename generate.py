#!/usr/bin/env python3
"""
Générateur de fichiers MP3 pour l'apprentissage des intervalles musicaux.

Chaque fichier contient les premières notes d'une mélodie connue,
servant de référence mnémotechnique pour reconnaître un intervalle.

Dépendances :
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
    # Exponential decay
    w *= np.exp(-t * 2.0)
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

N = note

MELODIES = [
    # === ASCENDING ===
    {
        "id": "01", "interval": "Seconde mineure", "semitones": 1,
        "direction": "asc", "song": "Les Dents de la mer (Jaws)",
        "notes_str": "E2-F2 (repete, accelerant)",
        "tempo": 120,
        "notes": [
            (N("E2"), 0.8), (N("F2"), 0.8),
            (N("E2"), 0.6), (N("F2"), 0.6),
            (N("E2"), 0.4), (N("F2"), 0.4),
            (N("E2"), 0.3), (N("F2"), 0.3),
            (N("E2"), 0.2), (N("F2"), 0.2),
            (N("E2"), 0.15), (N("F2"), 0.15),
        ],
    },
    {
        "id": "02", "interval": "Seconde majeure", "semitones": 2,
        "direction": "asc", "song": "Frere Jacques",
        "notes_str": "C4-D4-E4-C4",
        "tempo": 120,
        "notes": [
            (N("C4"), 1), (N("D4"), 1), (N("E4"), 1), (N("C4"), 1),
            (N("C4"), 1), (N("D4"), 1), (N("E4"), 1), (N("C4"), 1),
        ],
    },
    {
        "id": "03", "interval": "Tierce mineure", "semitones": 3,
        "direction": "asc", "song": "Greensleeves",
        "notes_str": "A3-C4-D4-E4-F4-E4-D4-B3...",
        "tempo": 140,
        "notes": [
            (N("A3"), 1),
            (N("C4"), 2), (N("D4"), 1),
            (N("E4"), 1.5), (N("F4"), 0.5), (N("E4"), 1),
            (N("D4"), 2), (N("B3"), 1),
            (N("G3"), 1.5), (N("A3"), 0.5), (N("B3"), 1),
            (N("C4"), 2), (N("A3"), 1),
            (N("A3"), 1.5), (N("G#3"), 0.5), (N("A3"), 1),
            (N("B3"), 2), (N("G#3"), 1),
            (N("E3"), 3),
        ],
    },
    {
        "id": "04", "interval": "Tierce majeure", "semitones": 4,
        "direction": "asc", "song": "Oh When the Saints",
        "notes_str": "C4-E4-F4-G4",
        "tempo": 120,
        "notes": [
            (N("C4"), 1), (N("E4"), 1), (N("F4"), 1), (N("G4"), 2),
            (None, 0.5),
            (N("C4"), 1), (N("E4"), 1), (N("F4"), 1), (N("G4"), 2),
        ],
    },
    {
        "id": "05", "interval": "Quarte juste", "semitones": 5,
        "direction": "asc", "song": "La Marseillaise",
        "notes_str": "G3-G3-G3-C4 (Al-lons en-fants)",
        "tempo": 100,
        "notes": [
            (N("G3"), 0.5), (N("G3"), 0.5),
            (N("G3"), 1), (N("G3"), 1),
            (N("C4"), 1), (N("C4"), 1),
            (N("D4"), 1.5), (N("E4"), 0.5), (N("C4"), 2),
        ],
    },
    {
        "id": "06", "interval": "Triton", "semitones": 6,
        "direction": "asc", "song": "The Simpsons (theme)",
        "notes_str": "C4-E4-F#4-A4-Ab4",
        "tempo": 120,
        "notes": [
            (N("C4"), 1), (N("E4"), 1), (N("F#4"), 1),
            (N("A4"), 0.5), (N("Ab4"), 2),
        ],
    },
    {
        "id": "07", "interval": "Quinte juste", "semitones": 7,
        "direction": "asc", "song": "Star Wars (theme principal)",
        "notes_str": "G3-G3-G3-C4-G4",
        "tempo": 100,
        "notes": [
            (N("G3"), 0.5), (N("G3"), 0.5), (N("G3"), 0.5),
            (N("C4"), 2), (N("G4"), 2),
            (N("F4"), 0.5), (N("E4"), 0.5), (N("D4"), 0.5),
            (N("C5"), 2),
        ],
    },
    {
        "id": "08", "interval": "Sixte mineure", "semitones": 8,
        "direction": "asc", "song": "Love Story (theme)",
        "notes_str": "E4-C5",
        "tempo": 100,
        "notes": [
            (N("E4"), 2), (N("C5"), 2),
            (N("B4"), 1), (N("C5"), 1),
            (N("C5"), 1), (N("B4"), 0.5), (N("A4"), 0.5), (N("B4"), 2),
        ],
    },
    {
        "id": "09", "interval": "Sixte majeure", "semitones": 9,
        "direction": "asc", "song": "My Bonnie Lies Over the Ocean",
        "notes_str": "G3-E4 (My Bon-nie)",
        "tempo": 100,
        "notes": [
            (N("G3"), 1), (N("E4"), 1.5), (N("D4"), 0.5), (N("C4"), 1),
            (N("D4"), 1), (N("E4"), 1), (N("D4"), 1), (N("C4"), 2),
        ],
    },
    {
        "id": "10", "interval": "Septieme mineure", "semitones": 10,
        "direction": "asc", "song": "Star Trek (theme)",
        "notes_str": "Bb3-Ab4",
        "tempo": 80,
        "notes": [
            (N("Bb3"), 2), (N("Ab4"), 3), (N("A4"), 1), (N("Bb4"), 2),
        ],
    },
    {
        "id": "11", "interval": "Septieme majeure", "semitones": 11,
        "direction": "asc", "song": "Take On Me (a-ha)",
        "notes_str": "F#4-F#4-D4-B3-B3-E4...",
        "tempo": 160,
        "notes": [
            (N("F#4"), 0.5), (N("F#4"), 0.5), (N("D4"), 0.5), (N("B3"), 0.5),
            (N("B3"), 0.5), (N("E4"), 0.5), (N("E4"), 0.5), (N("E4"), 0.5),
            (N("G#4"), 0.5), (N("G#4"), 0.5), (N("G#4"), 0.5), (N("F#4"), 0.5),
            (N("F#4"), 0.5), (N("F#4"), 0.5), (N("E4"), 0.5), (N("F#4"), 0.5),
        ],
    },
    {
        "id": "12", "interval": "Octave", "semitones": 12,
        "direction": "asc", "song": "Somewhere Over the Rainbow",
        "notes_str": "C4-C5 (Some-where)",
        "tempo": 90,
        "notes": [
            (N("C4"), 2), (N("C5"), 2),
            (N("B4"), 1.5), (N("C5"), 0.5),
            (N("B4"), 1), (N("G4"), 1), (N("A4"), 2),
        ],
    },
    # === DESCENDING ===
    {
        "id": "13", "interval": "Seconde mineure", "semitones": 1,
        "direction": "desc", "song": "La Lettre a Elise (Fur Elise)",
        "notes_str": "E5-D#5-E5-D#5-E5-B4-D5-C5-A4",
        "tempo": 140,
        "notes": [
            (N("E5"), 0.5), (N("D#5"), 0.5), (N("E5"), 0.5),
            (N("D#5"), 0.5), (N("E5"), 0.5),
            (N("B4"), 0.5), (N("D5"), 0.5), (N("C5"), 0.5),
            (N("A4"), 1.5),
        ],
    },
    {
        "id": "14", "interval": "Tierce mineure", "semitones": 3,
        "direction": "desc", "song": "Hey Jude (Beatles)",
        "notes_str": "A4-F#4 (Hey Jude)",
        "tempo": 100,
        "notes": [
            (N("A4"), 1.5), (N("F#4"), 1),
            (None, 0.5),
            (N("D4"), 1), (N("E4"), 0.5), (N("F#4"), 0.5), (N("A4"), 1),
        ],
    },
    {
        "id": "15", "interval": "Tierce majeure", "semitones": 4,
        "direction": "desc", "song": "Summertime (Gershwin)",
        "notes_str": "E5-C#5-B4 (Sum-mer-time)",
        "tempo": 80,
        "notes": [
            (N("E5"), 2), (N("C#5"), 1), (N("B4"), 2),
            (None, 0.5),
            (N("A4"), 1), (N("B4"), 0.5), (N("C#5"), 0.5), (N("E5"), 2),
        ],
    },
    {
        "id": "16", "interval": "Quarte juste", "semitones": 5,
        "direction": "desc", "song": "Une chanson douce (Henri Salvador)",
        "notes_str": "G4-G4-E4-D4 (U-ne chan-son)",
        "tempo": 100,
        "notes": [
            (N("G4"), 1), (N("G4"), 0.5), (N("E4"), 1), (N("D4"), 0.5),
            (N("G4"), 1), (N("E4"), 1),
        ],
    },
    {
        "id": "17", "interval": "Quinte juste", "semitones": 7,
        "direction": "desc", "song": "The Flintstones (theme)",
        "notes_str": "G4-C4 (Flint-stones!)",
        "tempo": 120,
        "notes": [
            (N("G4"), 1), (N("C4"), 1), (None, 0.3),
            (N("D4"), 0.5), (N("D4"), 0.5),
            (N("G4"), 1.5), (N("C4"), 1),
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
        filename = f"{m['id']}_{direction}_{m['interval'].replace(' ', '_')}.mp3"
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
