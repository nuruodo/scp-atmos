"""Minimal BWF / RF64 + ADM (axml/chna) reader/writer for 24-bit multichannel object exports."""
import re, struct
import numpy as np
import xml.etree.ElementTree as ET


def read_bwf(path):
    """-> (bytes, {chunk_id: (body_offset, size)})   handles RF64 (ds64) data sizes."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] in (b"RIFF", b"RF64"), "not a RIFF/RF64 file"
    pos, ds64, chunks = 12, None, {}
    while pos + 8 <= len(d):
        cid = d[pos:pos + 4]
        sz = struct.unpack("<I", d[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"ds64":
            ds64 = struct.unpack("<QQQ", d[body:body + 24])
        if cid == b"data" and sz == 0xFFFFFFFF and ds64:
            sz = ds64[1]
        chunks[cid] = (body, sz)
        pos = body + sz + (sz & 1)
    return d, chunks


def n_channels(d, ch):
    fb, _ = ch[b"fmt "]
    return struct.unpack("<H", d[fb + 2:fb + 4])[0]


def pcm24(d, body, sz, n_ch):
    a = np.frombuffer(d, np.uint8, count=sz, offset=body).reshape(-1, 3)
    v = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return (v.astype(np.float32) / (1 << 23)).reshape(-1, n_ch)


def to_pcm24(x):
    v = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32) & 0xFFFFFF
    flat = v.reshape(-1)
    b = np.empty((flat.size, 3), np.uint8)
    b[:, 0] = flat & 0xFF
    b[:, 1] = (flat >> 8) & 0xFF
    b[:, 2] = (flat >> 16) & 0xFF
    return b.tobytes()


def write_rf64(path, fmt, pcm24_bytes, extra):
    """Write an RF64 BWF keeping the given extra chunks (axml / chna / bext)."""
    data_sz = len(pcm24_bytes)
    n_ch = struct.unpack("<H", fmt[2:4])[0]
    frames = data_sz // (n_ch * 3)
    ds64_tmp = struct.pack("<QQQI", 0, data_sz, frames, 0)
    body = b"ds64" + struct.pack("<I", len(ds64_tmp)) + ds64_tmp
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt + (b"\0" if len(fmt) & 1 else b"")
    body += b"data" + struct.pack("<I", 0xFFFFFFFF) + pcm24_bytes + (b"\0" if data_sz & 1 else b"")
    for cid, payload in extra:
        body += cid + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
    riff = 4 + len(body)
    ds = struct.pack("<QQQI", riff, data_sz, frames, 0)
    body = body.replace(ds64_tmp, ds, 1)
    with open(path, "wb") as f:
        f.write(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + body)


def _tsec(s):
    h, m, r = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(r)


# ADM speakerLabel -> (x, y) for DirectSpeakers beds that carry no position (x: -1 left..+1 right, y: -1 back..+1 front)
_LABEL_POS = {"L": (-1, 1), "R": (1, 1), "C": (0, 1), "LS": (-1, 0), "RS": (1, 0), "LRS": (-1, -1), "RRS": (1, -1),
              "LB": (-1, -1), "RB": (1, -1), "LSS": (-1, 0), "RSS": (1, 0), "LW": (-1, .5), "RW": (1, .5)}


def label_to_xy(label):
    """'M+030' (azimuth degrees, + = left), 'RC_L', 'L', 'Ls' ... -> (x, y) or None"""
    if not label:
        return None
    lab = label.upper().replace("RC_", "")
    m = re.match(r"[MBU]([+-]\d{3})$", lab)
    if m:
        az = np.deg2rad(float(m.group(1)))
        return (float(-np.sin(az)), float(np.cos(az)))
    return _LABEL_POS.get(lab)


def parse_adm(d, ch):
    """-> (chfmt, tracks)
    chfmt:  {audioChannelFormatID: (typeDefinition, [(rtime_s, dur_s, X, Y, Z, interp_len_s, speakerLabel), ...])}
    tracks: [(wav channel number, 1-based, audioChannelFormatID)]"""
    b, s = ch[b"axml"]
    xml = d[b:b + s].decode("utf8", "ignore")
    xml = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", xml)
    xml = re.sub(r"\sxsi:\w+=\"[^\"]*\"", "", xml)
    root = ET.fromstring(xml)
    chfmt = {}
    for cf in root.iter("audioChannelFormat"):
        blocks = []
        for bf in cf.findall("audioBlockFormat"):
            pos = {p.get("coordinate"): float(p.text) for p in bf.findall("position")}
            rt = _tsec(bf.get("rtime", "00:00:00.0"))
            du = _tsec(bf.get("duration", "00:00:00.0")) if bf.get("duration") else None
            jp = bf.find("jumpPosition")
            il = float(jp.get("interpolationLength", "0.005")) if jp is not None else 0.005
            label = bf.findtext("speakerLabel")
            if not pos and label:                       # bed channel without a position: take it from the label
                xy = label_to_xy(label)
                if xy:
                    pos = {"X": xy[0], "Y": xy[1], "Z": 0.0}
            blocks.append((rt, du, pos.get("X", 0.0), pos.get("Y", 0.0), pos.get("Z", 0.0), max(il, 0.005), label))
        chfmt[cf.get("audioChannelFormatID")] = (cf.get("typeDefinition"), blocks)
    b, s = ch[b"chna"]
    c = d[b:b + s]
    n = struct.unpack("<H", c[:2])[0]
    tracks = []
    for i in range(n):
        r = c[4 + i * 40:4 + (i + 1) * 40]
        idx = struct.unpack("<H", r[:2])[0]
        m = re.search(r"AT_([0-9a-f]{8})_", r[2:40].decode("ascii", "ignore"))
        if m:
            tracks.append((idx, "AC_" + m.group(1)))
    return chfmt, tracks
