# jocsfx - Atmos object extraction + FX / dialogue / both stems

Takes any **Dolby Digital Plus (E-AC-3) JOC** or **Dolby TrueHD Atmos** audio track, pulls out the Atmos
objects, and builds a clean stereo stem for any time range ("scene"/"shot"), in one of three modes:

| mode | result | files written per segment |
|---|---|---|
| `fx` | sound effects only (no dialogue, no music) | `<name>_fx_stereo.wav` |
| `dialogue` | dialogue only | `<name>_dialogue_stereo.wav` |
| `both` | dialogue + fx, music removed | `<name>_fx_dialogue_stereo.wav` + the separate `_fx_` and `_dialogue_` stems |

All outputs are 24-bit / 48 kHz stereo WAV, trimmed sample-accurately to the requested range, level-matched with one
common gain per segment. `<name>_decisions.json` records what was kept/dropped per object and why.
`--clean-adm` additionally writes the separated per-object ADM BWF (same ADM metadata) for DAW use.

## How it works

```
source.mkv/.mka/.ec3/.thd
  |  ffprobe: codec + Atmos profile
  |  ffmpeg cut of [start,end] (+0.1 s tail), stream copy
  |-- E-AC-3 JOC : Cavern  -> ADM BWF (16 channels: bed + objects, positions in axml)
  |-- TrueHD     : truehdd decode --presentation 3 -> .atmos -> Cavern -> ADM BWF
  v
objdrop.process()
  1. per object: Bandit-v2 separation (speech / music / sfx) of the single object
  2. keep/drop objects by how much wanted material they contain (centre always kept, LFE only for fx/both,
     moving objects kept in fx/both)
  3. pan the kept objects to stereo from their ADM positions (time-varying, constant power, rear -3 dB, LFE -6 dB)
  4. Bandit again per stereo channel, sum the wanted stems
  5. align to a sample-accurate 5.1 decode (xcorr) and trim to the range
  6. if the object route is far quieter than a plain 5.1 fold-down route (broken ADM), use the 5.1 route
```

Separating each object *alone* first is what makes it work: an object that is 90 % music is dropped entirely instead of
being smeared by the separator.

## Dependencies (none are bundled except the 25-line exporter)

1. **ffmpeg + ffprobe** on PATH.
2. **.NET 8 runtime** (`dotnet`) - runs the exporter on Windows / macOS (Apple Silicon + Intel) / Linux.
3. **Cavern** (https://github.com/VoidXH/Cavern) - needs `Cavern.dll`, `Cavern.Format.dll`, `netstandard.dll`
   dropped next to `tool/ec3_to_adm.exe`. `tool/ec3_to_adm.cs` is the exporter source (`AudioReader` -> ADM BWF) if you
   want to rebuild it. It reads E-AC-3 JOC and the `.atmos` sets written by truehdd.
4. **truehdd** (https://github.com/truehdd/truehdd) - only for TrueHD tracks. Pass `--truehdd /path/to/truehdd`.
5. **Bandit-v2**: clone https://github.com/kwatcharasupat/bandit-v2 and download `checkpoint-multi.ckpt` from
   https://zenodo.org/records/12701995. Pass `--bandit-repo` / `--bandit-ckpt` (or env `JOCSFX_BANDIT_REPO`,
   `JOCSFX_BANDIT_CKPT`). torchaudio / pytorch_lightning are NOT needed (shimmed).
6. Python 3.10+, `pip install -r requirements.txt` (numpy, soundfile, torch). CUDA is used when available; on a Mac use
   `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1` (not verified) or CPU.

> **Licences - check before shipping inside a commercial app.** Cavern's release readme restricts redistribution /
> use of its components in other software; check the licence of the exact Cavern build you use, the truehdd licence,
> and the Bandit-v2 code + weights terms (trained on the DnR v3 dataset). The simplest compliant design is to have the app
> download these on first run rather than bundle them.

## Usage

```bash
python -m jocsfx probe  movie.mkv
python -m jocsfx run movie.mkv --mode both --start 00:02:39.4 --end 00:02:41.3 --out out/
python -m jocsfx run movie.mkv --mode fx   --segments scenes.json --out out/
python -m jocsfx run movie.mkv --mode dialogue --chunk 120 --out out/        # whole film in 2 min pieces
```

`scenes.json`: `[{"name":"scene_001","start":158.4,"end":162.3}, ...]` (CSV `name,start,end` also works; seconds or `HH:MM:SS.fff`).
Other flags: `--track N` (audio track index), `--clean-adm`, `--fast` (skip the 5.1 fallback separation: fewer Bandit passes, so faster; the fallback then only triggers if the object route is essentially silent),
`--keep-work` (keep cut/ADM/ref files), `--device`.

Python API: `from jocsfx.cli import run_segments; from jocsfx.extract import Config` - `run_segments(src, segments, out_dir, mode, track, cfg)`.

## Notes for integrating into an app

* Call it as a subprocess per scene (it is stateless; each run loads the model once per invocation - batch scenes with
  `--segments`). Progress lines are printed per segment.
* Cost: Bandit runs several full passes per segment (every object, then L/R, then the 5.1 fallback). Use `--fast` for
  previews, and cache stems per (file, track, start, end, mode).
* Output length = `end - start` exactly (`frames = round((end-start)*48000)`).
* `fx_dialogue` is built from one object-drop pass, so `fx` + `dialogue` sum to it.

## What was verified

* The pipeline was run end to end on a DD+ Atmos (E-AC-3 JOC) 6-second segment in `both` mode: ADM export, object-drop
  decisions, three stems, cleaned ADM. Output lengths and levels were sane.
* **Not verified here:** the TrueHD route (truehdd was not installed). The calls follow truehdd's documented CLI
  (`truehdd decode --presentation 3 file.thd --output-path dir`). If truehdd needs the cut to start on a major-sync frame,
  cut on one or decode the whole track once with truehdd and point `ec3_to_adm` at the `.atmos` set.
* `dialogue` mode logic (judging moving objects by ratio, dropping LFE) is new relative to the original FX-only
  pipeline and has not been listened to.
