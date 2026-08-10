"""Miner2 deep validation of sign_streak (consecutive same-sign day streak reversal).

Screened: IC1=-0.0609, ICIR1=-0.186, hit1=0.42, n1=994 dates, cov=0.599.
Checks:
  1. by-year IC1/ICIR1 stability
  2. per-symbol-group robustness (drop crypto, drop rates, drop China)
  3. min_names sensitivity (8/10/12)
  4. full-coverage sample (all 15 names valid) vs sparse sample
  5. decay: IC for h=1,2,3,5,10,20
  6. gate-style pairwise |spearman| vs effective library (threshold 0.5)
"""
import sys, time, json, os, pickle, base64, gzip, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner2_fast as F

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
CP = cache["close"]
RET = cache["ret"]
idx = CP.index
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10, 20)}


def sign_streak(ret):
    cols = {}
    for s in ret.columns:
        r = ret[s].values
        out = np.full(len(r), np.nan)
        prev = 0.0
        for i in range(len(r)):
            if not np.isfinite(r[i]):
                prev = 0.0
                continue
            sg = 1.0 if r[i] > 0 else (-1.0 if r[i] < 0 else 0.0)
            if sg == 0:
                prev = 0.0
                out[i] = 0.0
            elif np.sign(prev) == sg or prev == 0.0:
                prev = prev + sg if np.sign(prev) == sg else sg
                out[i] = prev
            else:
                prev = sg
                out[i] = sg
        cols[s] = out
    return pd.DataFrame(cols, index=ret.index)


STRK = sign_streak(RET)
print("panel built", STRK.shape, "coverage=%.3f" % (STRK.notna().sum().sum() / STRK.size))

# ---- 1. by-year ----
print("\n[by-year]")
for yr in range(2020, 2027):
    m = (idx.year == yr) & STRK.notna().any(axis=1)
    if m.sum() < 50:
        continue
    sub = STRK[m]
    fw = fwd[1].reindex(sub.index)
    r = F.fast_ic(sub, fw)
    print(f"  {yr}: n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f} hit={r['hit']:.2f}")

# ---- 2. drop groups ----
print("\n[group robustness]")
groups = {
    "all": SYMBOLS,
    "no_crypto": [s for s in SYMBOLS if s not in ("BTC", "ETH")],
    "no_rates": [s for s in SYMBOLS if s not in ("US10Y", "CN10Y")],
    "no_china": [s for s in SYMBOLS if s not in ("000300.SH", "000688.SH")],
    "equities_only": ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"],
    "crypto_only": ["BTC", "ETH"],
}
for gname, syms in groups.items():
    r = F.fast_ic(STRK[syms], fwd[1].reindex(idx)[syms])
    print(f"  {gname:14s} n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")

# ---- 3. min_names ----
print("\n[min_names]")
for mn in (8, 10, 12):
    r = F.fast_ic(STRK, fwd[1], min_names=mn)
    print(f"  min_names={mn}: n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")

# ---- 4. full-coverage sample ----
print("\n[full-coverage sample]")
full = RET.notna().all(axis=1)
print("  full-coverage dates:", int(full.sum()), f"({full.mean():.2%} of rows)")
for mn in (8, 13, 15):
    r = F.fast_ic(STRK[full], fwd[1].reindex(idx)[full], min_names=mn)
    print(f"  full-sample min_names={mn}: n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")

# ---- 5. decay ----
print("\n[decay]")
for h in (1, 2, 3, 5, 10, 20):
    r = F.fast_ic(STRK, fwd[h])
    print(f"  h={h:2d}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} n={r['n_dates']}")

# ---- 6. library correlation (gate-style per-row |spearman|) ----
def pd_rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sv = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sv[end] == sv[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pairwise_rho(A, B):
    n = min(A.shape[0], B.shape[0])
    A, B = A[-n:], B[-n:]
    vals = []
    for a, b in zip(A, B):
        fin = np.isfinite(a) & np.isfinite(b)
        if fin.sum() < 2:
            continue
        ra, rb = pd_rank(a[fin]), pd_rank(b[fin])
        if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(ra, rb)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan


def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = gzip.decompress(raw)
    except Exception:
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    mask = (idx >= start) & (idx <= end)
    dates = idx[mask]
    n = min(len(dates), arr.shape[0])
    return pd.DataFrame(arr[-n:], index=dates[-n:], columns=payload.get("symbols", SYMBOLS))


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


def load_lib(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    v = d.get("validation") or {}
    sa = sa or v.get("signal_artifact") or (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            return pd.DataFrame(np.load(p), index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "gzip" in fmt and "n_dates" in sa:
            try:
                return decode_gzipb64(sa)
            except Exception:
                pass
        if "base64:zlib:csv" in fmt or ("data" in sa and "n_dates" not in sa):
            try:
                return decode_b64zlibcsv(sa)
            except Exception:
                pass
    return None


print("\n[library gate-style rho vs sign_streak]")
cand = STRK.reindex(idx)
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json") or f.endswith(".bak"):
        continue
    libdf = load_lib(os.path.join("factors", f))
    if libdf is None or len(libdf) < 100:
        continue
    rho_lastN = pairwise_rho(cand.values, libdf.values)
    # date-aligned variant (overlap window)
    common = cand.index.intersection(libdf.index)
    if len(common) > 200:
        rho_da = pairwise_rho(cand.loc[common].values, libdf.loc[common].values)
    else:
        rho_da = np.nan
    print(f"  {f:44s} rows={len(libdf):4d} rho_lastN={rho_lastN:+.3f} rho_dateAlign={rho_da:+.3f}")

print("\ndone in %.1fs" % (time.time() - __import__("time").time() + __import__("time").time()))
