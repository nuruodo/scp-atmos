"""Probe a media file, cut a time range of its Atmos track and export the objects to an ADM BWF.

  E-AC-3 JOC  (DD+ Atmos)  : ffmpeg cut (-c copy)  ->  Cavern (ec3_to_adm)  ->  ADM BWF
  TrueHD Atmos             : ffmpeg cut (raw .thd) ->  truehdd decode  ->  .atmos  ->  Cavern (ec3_to_adm)  ->  ADM BWF
"""
import json, os, subprocess, glob
from dataclasses import dataclass

FS = 48000


@dataclass
class Config:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    dotnet: str = "dotnet"
    tool_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tool")  # ec3_to_adm.exe + Cavern*.dll
    truehdd: str = "truehdd"
    presentation: int = 3          # truehdd presentation index: 3 = full Atmos object presentation
    bandit_repo: str = ""
    bandit_ckpt: str = ""
    device: str = "auto"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout[-600:]}\n{r.stderr[-1200:]}")
    return r


def probe(cfg, path, track=0):
    """-> dict(kind='eac3'|'truehd'|'other', atmos=bool, codec, profile, channels, sample_rate, duration_s)"""
    r = run([cfg.ffprobe, "-v", "error", "-select_streams", f"a:{track}", "-show_entries",
             "stream=codec_name,profile,channels,sample_rate,duration:format=duration", "-of", "json", path])
    j = json.loads(r.stdout)
    s = j["streams"][0]
    codec, profile = s.get("codec_name", ""), s.get("profile", "") or ""
    kind = "eac3" if codec == "eac3" else "truehd" if codec == "truehd" else "other"
    dur = float(s.get("duration") or j.get("format", {}).get("duration") or 0)
    return dict(kind=kind, atmos="atmos" in profile.lower(), codec=codec, profile=profile,
                channels=s.get("channels"), sample_rate=s.get("sample_rate"), duration_s=dur)


def _cavern(cfg, src, dst):
    exe = os.path.join(cfg.tool_dir, "ec3_to_adm.exe")
    rc = os.path.join(cfg.tool_dir, "ec3_to_adm.runtimeconfig.json")
    r = run([cfg.dotnet, "exec", "--runtimeconfig", rc, exe, src, dst])
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def export_adm(cfg, src, track, start, end, workdir, name, kind):
    """-> (adm_wav_path, info_line). `start`/`end` in seconds of the source file; the cut carries 0.1 s extra tail."""
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, f"{name}_ADM.wav")
    dur = end - start + 0.1
    if kind == "eac3":
        cut = os.path.join(workdir, f"{name}_cut.mkv")
        run([cfg.ffmpeg, "-v", "error", "-y", "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", src,
             "-map", f"0:a:{track}", "-c", "copy", cut])
        info = _cavern(cfg, cut, out)
    elif kind == "truehd":
        cut = os.path.join(workdir, f"{name}_cut.thd")
        run([cfg.ffmpeg, "-v", "error", "-y", "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", src,
             "-map", f"0:a:{track}", "-c", "copy", "-f", "truehd", cut])
        dec = os.path.join(workdir, f"{name}_truehdd")
        os.makedirs(dec, exist_ok=True)
        run([cfg.truehdd, "decode", "--presentation", str(cfg.presentation), cut, "--output-path", dec])
        atmos = sorted(glob.glob(os.path.join(dec, "**", "*.atmos"), recursive=True))
        if not atmos:
            raise RuntimeError(f"truehdd produced no .atmos file in {dec} (is this an Atmos TrueHD track?)")
        info = _cavern(cfg, atmos[0], out)
    else:
        raise RuntimeError("only E-AC-3 (DD+) and TrueHD tracks are supported")
    if not os.path.exists(out):
        raise RuntimeError(f"ADM export failed for {name}")
    return out, info


def reference_51(cfg, src, track, start, end, dst):
    """Sample-accurate 5.1 decode of the same range (used to align the ADM export and as fallback)."""
    run([cfg.ffmpeg, "-v", "error", "-y", "-ss", f"{start:.6f}", "-t", f"{end - start:.6f}", "-i", src,
         "-map", f"0:a:{track}", "-ac", "6", "-ar", str(FS), "-c:a", "pcm_f32le", dst])
    return dst
