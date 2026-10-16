"""Bandit-v2 (DnR v3, 'multi' checkpoint) inference: speech / music / sfx stems for MONO signals, overlap-add chunking.
Code:    https://github.com/kwatcharasupat/bandit-v2   (clone it)
Weights: https://zenodo.org/records/12701995           (checkpoint-multi.ckpt)
Uses tiny shims for torchaudio + pytorch_lightning so neither has to be installed."""
import sys, types, math
import numpy as np
import torch
from torch import nn

STEMS = ("speech", "music", "sfx")


def _install_shims():
    ta = types.ModuleType("torchaudio")
    tr = types.ModuleType("torchaudio.transforms")
    fn = types.ModuleType("torchaudio.functional")

    class Spectrogram(nn.Module):
        def __init__(self, n_fft, win_length, hop_length, pad_mode, pad, window_fn, wkwargs, power, normalized, center, onesided):
            super().__init__()
            assert power is None and pad == 0 and normalized
            self.n_fft, self.win, self.hop, self.center, self.pad_mode = n_fft, win_length, hop_length, center, pad_mode
            self.onesided = onesided
            self.register_buffer("window", window_fn(win_length, **(wkwargs or {})))

        def forward(self, x):
            shp = x.shape
            y = x.reshape(-1, shp[-1])
            s = torch.stft(y, self.n_fft, self.hop, self.win, self.window, center=self.center, pad_mode=self.pad_mode,
                           normalized=False, onesided=self.onesided, return_complex=True)
            s = s / self.window.pow(2.0).sum().sqrt()
            return s.reshape(shp[:-1] + s.shape[-2:])

    class InverseSpectrogram(nn.Module):
        def __init__(self, n_fft, win_length, hop_length, pad_mode, pad, window_fn, wkwargs, normalized, center, onesided):
            super().__init__()
            self.n_fft, self.win, self.hop, self.center, self.onesided = n_fft, win_length, hop_length, center, onesided
            self.register_buffer("window", window_fn(win_length, **(wkwargs or {})))

        def forward(self, s, length=None):
            shp = s.shape
            z = s.reshape(-1, shp[-2], shp[-1]) * self.window.pow(2.0).sum().sqrt()
            y = torch.istft(z, self.n_fft, self.hop, self.win, self.window, center=self.center, normalized=False,
                            onesided=self.onesided, length=length)
            return y.reshape(shp[:-2] + (y.shape[-1],))

    tr.Spectrogram, tr.InverseSpectrogram = Spectrogram, InverseSpectrogram
    ta.transforms, ta.functional = tr, fn
    sys.modules.update({"torchaudio": ta, "torchaudio.transforms": tr, "torchaudio.functional": fn})
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = nn.Module
    sys.modules["pytorch_lightning"] = pl


def pick_device(want="auto"):
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"   # 'mps' only with PYTORCH_ENABLE_MPS_FALLBACK=1: pass --device mps


def load_model(repo, ckpt, device="auto"):
    device = pick_device(device)
    _install_shims()
    sys.path.insert(0, repo)
    from src.models.bandit.bandit import Bandit
    m = Bandit(in_channels=1, stems=list(STEMS), band_type="musical", n_bands=64,
               normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=8, emb_dim=128,
               rnn_dim=256, bidirectional=True, rnn_type="GRU", mlp_dim=512, hidden_activation="Tanh",
               hidden_activation_kwargs=None, complex_mask=True, use_freq_weights=True, n_fft=2048, win_length=2048,
               hop_length=512, window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True,
               pad_mode="reflect", onesided=True, fs=48000)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, _ = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.endswith("window")]
    assert not real_missing, real_missing[:5]
    m = m.to(device).eval()
    m._device = device
    return m


@torch.inference_mode()
def separate_mono(model, x, fs=48000, chunk_s=8.0, hop_s=2.0, batch=4):
    """x: float32 np (N,)  ->  {'speech': (N,), 'music': (N,), 'sfx': (N,)}"""
    device = model._device
    C, Hp = int(chunk_s * fs), int(hop_s * fs)
    N = len(x)
    pad_front = C - Hp
    n_chunks = max(1, math.ceil((N + pad_front - C) / Hp) + 1)
    total = (n_chunks - 1) * Hp + C
    pad_back = total - N - pad_front
    xp = np.pad(x, (pad_front, max(pad_back, 0)), mode="reflect") if N > 1 else np.pad(x, (pad_front, max(pad_back, 0)))
    if len(xp) < total:
        xp = np.pad(xp, (0, total - len(xp)))
    win = torch.hann_window(C, device=device).clamp_min(1e-4)
    out = {s: torch.zeros(total, device=device) for s in STEMS}
    wsum = torch.zeros(total, device=device)
    xt = torch.from_numpy(xp.astype(np.float32)).to(device)
    for b0 in range(0, n_chunks, batch):
        idx = list(range(b0, min(n_chunks, b0 + batch)))
        seg = torch.stack([xt[i * Hp:i * Hp + C] for i in idx])[:, None, :]
        res = model({"mixture": {"audio": seg}})["estimates"]
        for k, i in enumerate(idx):
            for s in STEMS:
                out[s][i * Hp:i * Hp + C] += res[s]["audio"][k, 0] * win
            wsum[i * Hp:i * Hp + C] += win
    return {s: (out[s] / wsum)[pad_front:pad_front + N].cpu().numpy() for s in STEMS}
