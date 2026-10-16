"""Object-drop stem builder (stereo headphone mix) with three modes:

    fx        -> sound effects only      (Bandit 'sfx' stem; no dialogue, no music)
    dialogue  -> dialogue only           (Bandit 'speech' stem)
    both      -> dialogue + fx           (music removed). Also returns the separate fx and dialogue stems.

Per segment:
 1. Centre (screen channel) static objects are ALWAYS kept (cleaned alone). Front L/R are kept only if enough of the wanted
    material survives the separation. LFE is kept for fx / both and dropped for dialogue.
 2. Moving objects are kept in fx / both modes (they are effects); in dialogue mode they are judged like every other object.
 3. Every other object is separated alone: ratio = wanted energy / total energy. Objects dominated by unwanted material
    (ratio < DROP_BELOW) are deleted; objects with negligible energy are kept unanalysed.
 4. Remaining objects are panned to stereo from their ADM positions (+LFE at -6 dB), then separated again per channel.
 5. If the object route comes out much quieter than a plain 5.1 fold-down route (e.g. broken ADM), the 5.1 route is used.
"""
import os
import numpy as np
import soundfile as sf

from . import bandit_infer as B
from .bwf import read_bwf, pcm24, parse_adm, n_channels

FS = 48000
LFE_TO_STEREO = 0.5          # -6 dB
DROP_BELOW = 0.5
LR_MIN = 0.15
MIN_SHARE = 0.005
FALLBACK_RATIO = 0.35

MODES = {"fx": ("sfx",), "dialogue": ("speech",), "both": ("sfx", "speech")}


def stereo_gains(x, y):
    """constant-power stereo pan from X (-1 left .. +1 right); sources behind the listener -3 dB (headphone friendly)."""
    x = float(np.clip(x, -1, 1))
    th = (x + 1) * np.pi / 4
    gl, gr = np.cos(th), np.sin(th)
    rear = 0.7071 if y < -0.25 else 1.0
    if -0.25 <= y <= 0.25:
        rear = 1.0 - (0.25 - y) / 0.5 * (1 - 0.7071)
    return np.array([gl * rear, gr * rear])


def load_objects(path):
    d, ch = read_bwf(path)
    n = n_channels(d, ch)
    x = pcm24(d, *ch[b"data"], n)
    chfmt, tracks = parse_adm(d, ch)
    info = []
    for idx, acid in tracks:
        if acid not in chfmt or idx > n:
            continue
        typ, blocks = chfmt[acid]
        xs = [b[2] for b in blocks]; ys = [b[3] for b in blocks]; zs = [b[4] for b in blocks]
        move = (max(xs) - min(xs)) + (max(ys) - min(ys)) + (max(zs) - min(zs))
        info.append(dict(idx=idx, typ=typ, blocks=blocks, pos=(float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))),
                         dynamic=bool(len(blocks) > 2 or move > 0.05),
                         lfe=(typ == "DirectSpeakers" and "LFE" in (blocks[0][6] or "").upper())))
    return x, info


def is_centre(p):
    return p[1] >= 0.9 and abs(p[2]) <= 0.1 and abs(p[0]) < 0.1


def is_front_lr(p):
    return p[1] >= 0.9 and abs(p[2]) <= 0.1 and abs(p[0]) >= 0.9


def mix_stereo(x, objs, keep, override):
    N = x.shape[0]
    t = np.arange(N) / FS
    out = np.zeros((N, 2), np.float32)
    for o in objs:
        if o["idx"] not in keep:
            continue
        sig = override.get(o["idx"], x[:, o["idx"] - 1])
        if o["lfe"]:
            out += LFE_TO_STEREO * sig[:, None]
            continue
        bt, bg = [], []
        for (rt, du, X, Y, Z, il, _) in o["blocks"]:
            g = stereo_gains(X, Y)
            if bt:
                bt.append(rt); bg.append(bg[-1])
            bt.append(rt + il if bt else rt); bg.append(g)
        bt, bg = np.array(bt), np.array(bg)
        for k in (0, 1):
            out[:, k] += sig * np.interp(t, bt, bg[:, k]).astype(np.float32)
    return out


def estimate_offset(x, objs, ref):
    """samples by which the ADM export runs EARLY relative to the sample-accurate 5.1 decode (xcorr: centre object vs 5.1 C)"""
    c = next((o for o in objs if is_centre(o["pos"]) and not o["dynamic"]), None)
    if c is None or ref is None or ref.shape[1] < 3:
        return 0
    a, b = x[:, c["idx"] - 1], ref[:, 2]
    n = min(len(a), len(b), FS * 6)
    A = np.fft.rfft(a[:n] * np.hanning(n), 2 * n)
    Bf = np.fft.rfft(b[:n] * np.hanning(n), 2 * n)
    cc = np.fft.irfft(A * np.conj(Bf))
    vals = cc[0:4096]
    k = int(np.argmax(vals))
    return k if vals[k] > 0.2 * np.abs(cc).max() else 0


def _wanted(stems, want):
    return sum(stems[s] for s in want)


def _fold51(z, with_lfe):
    lfe = LFE_TO_STEREO * z[:, 3] if with_lfe else 0.0
    L = z[:, 0] + 0.7071 * z[:, 2] + 0.7071 * z[:, 4] + lfe
    R = z[:, 1] + 0.7071 * z[:, 2] + 0.7071 * z[:, 5] + lfe
    return np.stack([L, R], 1)


def process(adm_path, ref51_path, model, mode="fx", fast=False, log=print):
    """-> (outputs: {name: float32 (N,2)}, decisions: dict)"""
    want = MODES[mode]
    with_lfe = "sfx" in want
    x, objs = load_objects(adm_path)
    ref, frames = None, None
    if ref51_path and os.path.exists(ref51_path):
        ref, _ = sf.read(ref51_path, dtype="float32")
        frames = ref.shape[0]
    etot = float((x ** 2).sum()) + 1e-12
    cache = {}

    def stems_of(o):
        if o["idx"] not in cache:
            cache[o["idx"]] = B.separate_mono(model, np.ascontiguousarray(x[:, o["idx"] - 1]))
        return cache[o["idx"]]

    keep, rec, override = set(), [], {}
    for o in objs:
        e = float((x[:, o["idx"] - 1] ** 2).sum())
        share = e / etot
        ratio, why = None, ""
        if o["lfe"]:
            act, why = ("keep", "LFE") if with_lfe else ("DROP", "LFE (not part of this mode)")
        elif is_centre(o["pos"]) and not o["dynamic"]:
            st = stems_of(o)
            sig = _wanted(st, want)
            ratio = float((sig ** 2).sum()) / max(e, 1e-12)
            act, override[o["idx"]] = "keep", sig
            why = f"centre = screen channel (always kept; separated alone, wanted ratio {ratio:.2f})"
        elif is_front_lr(o["pos"]) and not o["dynamic"]:
            st = stems_of(o)
            sig = _wanted(st, want)
            ratio = float((sig ** 2).sum()) / max(e, 1e-12)
            if ratio >= LR_MIN:
                act, override[o["idx"]] = "keep", sig
                why = f"front L/R kept, cleaned (wanted ratio {ratio:.2f})"
            else:
                act, why = "DROP", f"front L/R dropped: almost no wanted material (ratio {ratio:.2f})"
        elif o["dynamic"] and with_lfe:
            act, why = "keep", "dynamic object"
        elif share < MIN_SHARE:
            act, why = "keep", "negligible energy"
        else:
            st = stems_of(o)
            ratio = float((_wanted(st, want) ** 2).sum()) / max(e, 1e-12)
            if ratio < DROP_BELOW:
                act, why = "DROP", f"unwanted material dominates (ratio {ratio:.2f})"
            else:
                act, why = "keep", f"wanted ratio {ratio:.2f}"
        if act == "keep":
            keep.add(o["idx"])
        rec.append(dict(ch=o["idx"], pos=[round(v, 2) for v in o["pos"]], share=round(share, 4),
                        ratio=None if ratio is None else round(ratio, 3), action=act, why=why))

    st_mix = mix_stereo(x, objs, keep, override)
    res = {s: np.zeros_like(st_mix) for s in B.STEMS}
    for k in (0, 1):
        r = B.separate_mono(model, np.ascontiguousarray(st_mix[:, k]))
        for s in B.STEMS:
            res[s][:, k] = r[s]

    off = estimate_offset(x, objs, ref)
    if frames is not None:
        def crop(a):
            a = a[off:off + frames]
            return np.pad(a, ((0, frames - a.shape[0]), (0, 0))) if a.shape[0] < frames else a
        res = {s: crop(a) for s, a in res.items()}
    main = _wanted(res, want)
    rms_o = float(np.sqrt((main ** 2).mean()))

    use_fb, rms_f = False, None
    if ref is not None:
        if fast:
            raw = _fold51(ref, with_lfe)
            rms_f = float(np.sqrt((raw ** 2).mean()))
            use_fb = rms_o < 1e-4 * max(rms_f, 1e-9)
        else:
            fb = {s: np.zeros((frames, 2), np.float32) for s in B.STEMS}
            per = [B.separate_mono(model, np.ascontiguousarray(ref[:, c])) for c in range(6)]
            for s in B.STEMS:
                z = np.stack([p[s] for p in per], 1)
                fb[s] = _fold51(z, with_lfe)
            fb_main = _wanted(fb, want)
            rms_f = float(np.sqrt((fb_main ** 2).mean()))
            use_fb = rms_o < FALLBACK_RATIO * rms_f
            if use_fb:
                res = fb
    chosen = res
    outs = {}
    if mode == "fx":
        outs["fx"] = chosen["sfx"]
    elif mode == "dialogue":
        outs["dialogue"] = chosen["speech"]
    else:
        outs["fx_dialogue"] = chosen["sfx"] + chosen["speech"]
        outs["fx"] = chosen["sfx"]
        outs["dialogue"] = chosen["speech"]
    # one common gain per segment so the three files stay level-matched
    peak = max(float(np.abs(a).max()) for a in outs.values())
    g = 0.98 / peak if peak > 0.98 else 1.0
    outs = {k: np.clip(a * g, -1, 1).astype(np.float32) for k, a in outs.items()}
    dec = dict(mode=mode, source="5.1 fallback" if use_fb else "object-drop", offset_samples=off,
               rms_objdrop=round(rms_o, 5), rms_fallback=None if rms_f is None else round(rms_f, 5),
               kept=sorted(keep), dropped=[r["ch"] for r in rec if r["action"] == "DROP"], objects=rec,
               peak_before_gain=round(peak, 3))
    return outs, dec
