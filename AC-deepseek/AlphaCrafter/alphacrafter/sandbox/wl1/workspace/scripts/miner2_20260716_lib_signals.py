"""Load existing library factor signals (for pairwise correlation checks).

Library artifacts come in two forms:
  1. external .npy files (e.g. miner2_20260716_mom_10d_skip5.npy) - (n_dates, 15) float32
  2. embedded gzip+base64 float32 matrix inside signal_artifact.data_b64
Both cover the 15-name cross-asset panel; embedded ones start ~2021-01-04.
"""
import json, glob, os, base64, gzip
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def decode_b64_matrix(d):
    raw = base64.b64decode(d["data_b64"])
    arr = np.frombuffer(gzip.decompress(raw), dtype=np.float32)
    arr = arr.reshape(d["n_dates"], d["n_symbols"])
    idx = pd.date_range(d["date_start"], d["date_end"], freq="B")
    if len(idx) != d["n_dates"]:
        # fall back to positional index
        idx = pd.RangeIndex(d["n_dates"])
    return pd.DataFrame(arr, index=idx, columns=d.get("symbols", SYMBOLS))

def load_library_signals():
    signals = {}
    meta = {}
    for p in sorted(glob.glob("factors/*.json")):
        if ".bak" in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        fid = d.get("factor_id", os.path.basename(p)[:-5])
        if fid in signals:
            continue
        art = d.get("signal_artifact")
        if isinstance(art, dict) and "data_b64" in art:
            df = decode_b64_matrix(art)
            signals[fid] = df
            meta[fid] = {"type": "embedded", "shape": df.shape,
                         "range": f"{df.index.min()}..{df.index.max()}"}
        elif isinstance(art, str) and art.endswith(".npy"):
            path = os.path.join("factors", art)
            if not os.path.exists(path):
                continue
            arr = np.load(path)
            # external artifacts span the full panel index (2020-01-01..2026-07-15)
            idx = pd.date_range("2020-01-01", "2026-07-15", freq="B")
            if len(idx) != arr.shape[0]:
                idx = pd.RangeIndex(arr.shape[0])
            df = pd.DataFrame(arr, index=idx, columns=SYMBOLS)
            signals[fid] = df
            meta[fid] = {"type": "npy", "shape": arr.shape,
                         "range": f"{df.index.min()}..{df.index.max()}"}
    return signals, meta

if __name__ == "__main__":
    sig, meta = load_library_signals()
    print(f"loaded {len(sig)} library signals:")
    for fid in sorted(sig):
        print(f"  {fid:38s} {meta[fid]}")
