# 🎵 Intervalles Musicaux — Mnémotechniques

Outil d'apprentissage des intervalles musicaux à l'oreille, basé sur les premières notes de mélodies connues.

## Utilisation en ligne

👉 **[Ouvrir la page](https://gnodet.github.io/intervalles-musicaux/)**

## Générer les fichiers audio

```bash
pip install numpy
python generate.py --output-dir audio/ --manifest manifest.json
```

Requiert `ffmpeg` pour la conversion WAV → MP3.

## Structure

- `generate.py` — Script de génération des MP3 (synthèse sinusoïdale)
- `index.html` — Page web interactive avec lecteur audio
- `audio/` — Fichiers MP3 générés
- `manifest.json` — Métadonnées des intervalles (utilisé par le HTML)

## Intervalles couverts

### Ascendants ⬆️
| Intervalle | Demi-tons | Chanson |
|---|---|---|
| Seconde mineure | 1 | Jaws |
| Seconde majeure | 2 | Frère Jacques |
| Tierce mineure | 3 | Greensleeves |
| Tierce majeure | 4 | Oh When the Saints |
| Quarte juste | 5 | La Marseillaise |
| Triton | 6 | The Simpsons |
| Quinte juste | 7 | Star Wars |
| Sixte mineure | 8 | Love Story |
| Sixte majeure | 9 | My Bonnie |
| Septième mineure | 10 | Star Trek |
| Septième majeure | 11 | Take On Me |
| Octave | 12 | Over the Rainbow |

### Descendants ⬇️
| Intervalle | Demi-tons | Chanson |
|---|---|---|
| Seconde mineure | 1 | Für Elise |
| Tierce mineure | 3 | Hey Jude |
| Tierce majeure | 4 | Summertime |
| Quarte juste | 5 | Une chanson douce |
| Quinte juste | 7 | The Flintstones |
