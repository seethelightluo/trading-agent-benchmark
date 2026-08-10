"""
Reverse-engineer the gate's pairwise correlation metric from the nclv_1d
eviction evidence (abs_spearman_rho=0.9029 vs mom_10d_skip5), then score the
new candidate body_pos_1d against the full effective library with that metric.
"""
import os, json, pickle, base64, gzip, zlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]; open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]; low = cache["low"][SYMBOLS]
idx = close.index

# ---------------- loaders ----------------
def load_npy(pid):
    M = np.load(f"factors/{pid}.npy")
    return pd.DataFrame(M, index=idx, columns=SYMBOLS)

def decode_b64zlibcsv(payload):
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    txt = raw.decode("utf-8", errors="replace")
    rows = []
    for line in txt.splitlines():
        if not line.strip() or line.startswith("date,"):
            continue
        parts = line.split(",")
        rows.append((parts[0], [float(v) if v not in ("", "NA") else np.nan for v in parts[1:]]))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=payload.get("columns", SYMBOLS))
    df.index = pd.to_datetime(df.index)
    return df

def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = gzip.decompress(raw)
    except Exception:
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    mask = (idx >= start) & (idx <= end) & close.notna().all(axis=1)
    common = idx[mask]
    n = min(len(common), arr.shape[0])
    dates = common[-n:]
    return pd.DataFrame(arr[-n:], index=dates, columns=payload.get("symbols", SYMBOLS))

def load_lib_factor(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    if sa is None:
        v = d.get("validation") or {}
        sa = v.get("signal_artifact") or (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            return pd.DataFrame(np.load(p), index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "gzip" in fmt and "n_dates" in sa:
            try: return decode_gzipb64(sa)
            except Exception: pass
        if "base64:zlib:csv" in fmt or ("data" in sa and "n_dates" not in sa):
            try: return decode_b64zlibcsv(sa)
            except Exception: pass
    return None

lib = {}
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json"):
        continue
    df = load_lib_factor(os.path.join("factors", f))
    if df is not None and len(df) > 100:
        lib[f] = df
print("library:")
for k, v in lib.items():
    print(f"  {k:42s} {v.shape} {v.index.min().date()}..{v.index.max().date()}")

# ---------------- correlation metrics ----------------
def pooled_spearman(a, b):
    A = a.values.astype(float).ravel(); B = b.values.astype(float).ravel()
    m = np.isfinite(A) & np.isfinite(B)
    if m.sum() < 30:
        return np.nan
    return float(spearmanr(A[m], B[m]).statistic)

def perdate_spearman(a, b, signed=False):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 8:
            continue
        r = spearmanr(x[mm], y[mm]).statistic
        if np.isfinite(r):
            vals.append(r if signed else abs(r))
    return float(np.mean(vals)) if vals else np.nan

# 1) diagnose nclv vs mom with all three metrics
nclv = load_npy("miner2_20260716_nclv_1d")
mom = load_npy("miner2_20260716_mom_10d_skip5")
print("\n[nclv_1d vs mom_10d_skip5] gate said abs_spearman_rho=0.9029")
print(f"  pooled spearman        : {pooled_spearman(nclv, mom):.4f}")
print(f"  per-date mean signed   : {perdate_spearman(nclv, mom, signed=True):.4f}")
print(f"  per-date mean abs      : {perdate_spearman(nclv, mom, signed=False):.4f}")

# 2) candidate body_pos_1d
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([open_[c], close[c], high[c], low[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    rng = (df.iloc[:, 2] - df.iloc[:, 3])
    _p.loc[df.index, c] = (df.iloc[:, 1] - df.iloc[:, 0]) / rng.replace(0, np.nan)
body = _p

print("\n[body_pos_1d vs library] pooled spearman:")
for k, v in lib.items():
    r = pooled_spearman(body, v)
    print(f"  vs {k:42s} pooled_rho={r:.4f} {'<<< CONFLICT' if abs(r) >= 0.5 else 'ok'}")

# also per-date signed for reference
print("\n[body_pos_1d vs library] per-date mean signed spearman:")
for k, v in lib.items():
    r = perdate_spearman(body, v, signed=True)
    print(f"  vs {k:42s} rho={r:.4f}")
