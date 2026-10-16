"""Command line / Python entry points.

  python -m scp_atmos probe  FILE [--track 0]
  python -m scp_atmos run    FILE --mode fx|dialogue|both --out OUT  (--start S --end E | --segments FILE | --chunk SECONDS)
                          [--track 0] [--clean-adm] [--keep-work] [--fast] [--device auto|cuda|cpu|mps]
                          --bandit-repo DIR --bandit-ckpt FILE      (or env SCP_ATMOS_BANDIT_REPO / SCP_ATMOS_BANDIT_CKPT)

Segments file: JSON  [{"name": "shot_001", "start": 158.4, "end": 162.3}, ...]   or CSV  name,start,end
Times are seconds or HH:MM:SS(.fff).
"""
import argparse, csv, json, os, shutil, struct, sys, time
import numpy as np
import soundfile as sf

from . import extract, objdrop
from .extract import Config, FS
from .bwf import read_bwf, pcm24, to_pcm24, write_rf64, n_channels


def parse_time(v):
    if isinstance(v, (int, float)):
        return float(v)
    p = str(v).split(":")
    t = 0.0
    for part in p:
        t = t * 60 + float(part)
    return t


def load_segments(path):
    if path.lower().endswith(".json"):
        raw = json.load(open(path, encoding="utf-8"))
    else:
        raw = [dict(name=r[0], start=r[1], end=r[2]) for r in csv.reader(open(path, encoding="utf-8")) if r and not r[0].startswith("#")]
    return [dict(name=s["name"], start=parse_time(s["start"]), end=parse_time(s["end"])) for s in raw]


def clean_adm(adm_path, out_path, model, mode):
    """Separate every non-silent object channel of the ADM BWF and write a new RF64 BWF with the original fmt/axml/chna."""
    want = objdrop.MODES[mode]
    d, ch = read_bwf(adm_path)
    n = n_channels(d, ch)
    x = pcm24(d, *ch[b"data"], n)
    y = np.zeros_like(x)
    from . import bandit_infer as B
    for c in range(n):
        if np.abs(x[:, c]).max() < 3e-5:
            continue
        st = B.separate_mono(model, np.ascontiguousarray(x[:, c]))
        y[:, c] = sum(st[s] for s in want)
    fb, fs = ch[b"fmt "]
    fmt = d[fb:fb + fs]
    extra = [(cid, d[ch[cid][0]:ch[cid][0] + ch[cid][1]]) for cid in (b"axml", b"chna", b"bext") if cid in ch]
    write_rf64(out_path, fmt, to_pcm24(y), extra)


def run_segments(src, segments, out_dir, mode="fx", track=0, cfg=None, clean=False, keep_work=False, fast=False, log=print):
    cfg = cfg or Config()
    info = extract.probe(cfg, src, track)
    log(f"[probe] {info['codec']} / {info['profile']} ch={info['channels']} dur={info['duration_s']:.1f}s")
    if info["kind"] == "other":
        raise SystemExit("track is not E-AC-3 or TrueHD")
    if not info["atmos"]:
        log("[warn] profile does not say Atmos - the export may contain a plain bed with no objects")
    from . import bandit_infer as B
    model = B.load_model(cfg.bandit_repo, cfg.bandit_ckpt, cfg.device)
    results = []
    t0 = time.time()
    for i, s in enumerate(segments, 1):
        name = s["name"]
        seg_dir = os.path.join(out_dir, name)
        os.makedirs(seg_dir, exist_ok=True)
        work = os.path.join(seg_dir, "_work")
        try:
            adm, line = extract.export_adm(cfg, src, track, s["start"], s["end"], work, name, info["kind"])
            ref = extract.reference_51(cfg, src, track, s["start"], s["end"], os.path.join(work, f"{name}_ref51.wav"))
            outs, dec = objdrop.process(adm, ref, model, mode, fast=fast, log=log)
            for key, a in outs.items():
                sf.write(os.path.join(seg_dir, f"{name}_{key}_stereo.wav"), a, FS, subtype="PCM_24")
            dec["export"] = line
            json.dump(dec, open(os.path.join(seg_dir, f"{name}_decisions.json"), "w"), indent=1)
            if clean:
                clean_adm(adm, os.path.join(seg_dir, f"{name}_ADM_{mode}.wav"), model, mode)
            log(f"[{i}/{len(segments)}] {name} {dec['source']} dropped={dec['dropped']} {time.time() - t0:.0f}s")
            results.append(dict(name=name, outputs=list(outs), source=dec["source"]))
        finally:
            if not keep_work:
                shutil.rmtree(work, ignore_errors=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scp-atmos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("file"); p.add_argument("--track", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("file"); r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=list(objdrop.MODES), default="fx")
    r.add_argument("--track", type=int, default=0)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--segments"); g.add_argument("--chunk", type=float); g.add_argument("--start")
    r.add_argument("--end")
    r.add_argument("--clean-adm", action="store_true", help="also write the separated per-object ADM BWF")
    r.add_argument("--keep-work", action="store_true"); r.add_argument("--fast", action="store_true")
    r.add_argument("--device", default="auto")
    r.add_argument("--presentation", type=int, default=3)
    r.add_argument("--bandit-repo", default=os.environ.get("SCP_ATMOS_BANDIT_REPO", ""))
    r.add_argument("--bandit-ckpt", default=os.environ.get("SCP_ATMOS_BANDIT_CKPT", ""))
    r.add_argument("--truehdd", default=os.environ.get("SCP_ATMOS_TRUEHDD", "truehdd"))
    r.add_argument("--tool-dir", default=os.environ.get("SCP_ATMOS_TOOL_DIR", Config.tool_dir))
    a = ap.parse_args(argv)
    cfg = Config()
    if a.cmd == "probe":
        print(json.dumps(extract.probe(cfg, a.file, a.track), indent=1))
        return 0
    cfg.bandit_repo, cfg.bandit_ckpt, cfg.device = a.bandit_repo, a.bandit_ckpt, a.device
    cfg.truehdd, cfg.presentation, cfg.tool_dir = a.truehdd, a.presentation, a.tool_dir
    if a.segments:
        segs = load_segments(a.segments)
    elif a.chunk:
        dur = extract.probe(cfg, a.file, a.track)["duration_s"]
        segs, t, k = [], 0.0, 1
        while t < dur:
            segs.append(dict(name=f"seg_{k:04d}", start=t, end=min(t + a.chunk, dur)))
            t += a.chunk; k += 1
    else:
        if a.end is None:
            ap.error("--start needs --end")
        segs = [dict(name="segment", start=parse_time(a.start), end=parse_time(a.end))]
    run_segments(a.file, segs, a.out, a.mode, a.track, cfg, a.clean_adm, a.keep_work, a.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
