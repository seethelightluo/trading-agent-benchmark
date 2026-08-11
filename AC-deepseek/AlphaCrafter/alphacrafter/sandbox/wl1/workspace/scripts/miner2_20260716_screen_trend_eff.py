"""
miner_2 cycle 2026-07-16 (new): trend-quality family via Kaufman efficiency ratio.

Motivation: the persisted library is dominated by short-horizon mean-reversion
(rev_1d/2d/3d, nclv_1d..5d, nbody_1d, id_rev_1d, mom_10d_skip5-as-reversal).
A structurally different return driver is TREND QUALITY: the Kaufman efficiency
ratio (net displacement / gross path length) measures how directional a market
is. In trending regimes, high-ER assets tend to continue; in choppy regimes ER
is low. We test ER at several horizons for cross-sectional predictive power.

Admission gate (shared, 15-instrument universe):
    |daily paper IC|  >= 0.0070
    |daily paper ICIR| >= 0.0840
Validation: 2021-01-01 .. 2026-07-15, >=8 valid names per date.
"""
import sys, os, json, time, base64, zlib
import numpy as np
import pandas as pd

T0 = time.time()
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VALID_LO, VALID_HI = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-15")
IC_MIN, ICIR_MIN = 0.0070, 0.0840
RHO_MAX = 0.5
MIN_NAMES = 8

cache = pickle = None
import pickle
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]
idx = close.index
lret = np.log(close).diff()

# ---------------------------------------------------------------------------
def fwd_log(closes, h):
    return np.log(closes.shift(-h)) - np.log(closes)

def fast_ic(factor_df, fwd, min_names=MIN_NAMES):
    F = factor_df.values.astype(float); R = fwd.values.astype(float)
    n = np.isfinite(F) & np.isfinite(R)
    ok = n.sum(axis=1) >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    Fm = np.where(n, F, 0.0); Rm = np.where(n, R, 0.0)
    cnt = n.sum(axis=1)[ok]
    sx = Fm[ok].sum(axis=1); sy = Rm[ok].sum(axis=1)
    sxx = (Fm[ok] ** 2).sum(axis=1); syy = (Rm[ok] ** 2).sum(axis=1)
    sxy = (Fm[ok] * Rm[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        num = cnt * sxy - sx * sy
        den = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
        ic = num / den
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(cnt.sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean())}

def turnover10(factor_df, rebal=10):
    ranks = factor_df.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna(); cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        chg.append((cur[common] - prev[common]).abs().mean() / (len(common) - 1))
    return float(np.mean(chg)) if chg else np.nan

def pair_rho(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            continue
        x, y = x[m], y[m]
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(rx, ry)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

def per_symbol_dense(fn):
    out = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
    for c in SYMBOLS:
        s = close[c].dropna()
        if len(s) < 90:
            continue
        out[c] = fn(s)
    return out

# ---------------------------------------------------------------------------
# Kaufman efficiency ratio: |C_t - C_{t-n}| / sum(|C_i - C_{i-1}|, n)
# ---------------------------------------------------------------------------
def kaufman_er(s, n):
    r = s.diff()
    num = (s - s.shift(n)).abs()
    den = r.abs().rolling(n).sum()
    return num / den.replace(0, np.nan)

panels = {}
META = {}
for n in (5, 10, 20, 30, 60):
    panels[f"er_{n}"] = per_symbol_dense(lambda s, n=n: kaufman_er(s, n))
    META[f"er_{n}"] = {"name": f"Kaufman efficiency ratio {n}d",
                       "expr": f"|C_t - C_{{t-{n}}}| / sum(|C_i - C_{{i-1}}|, {n})",
                       "dep": ["close"], "params": {"win": n},
                       "tags": ["trend-quality", "efficiency-ratio"]}

print(f"panels built in {time.time()-T0:.1f}s: {list(panels.keys())}")

m = (idx >= VALID_LO) & (idx <= VALID_HI)
fwd1 = fwd_log(close, 1); fwd5 = fwd_log(close, 5); fwd10 = fwd_log(close, 10)
n_cells = len(SYMBOLS) * int(m.sum())

results = {}
for nm, p in panels.items():
    P = p.loc[m]
    ic1 = fast_ic(P, fwd1.loc[m]); ic5 = fast_ic(P, fwd5.loc[m]); ic10 = fast_ic(P, fwd10.loc[m])
    cov = float(P.notna().sum().sum()) / n_cells
    to = turnover10(p.loc[m])
    passed = (abs(ic1["ic"]) >= IC_MIN) and (abs(ic1["icir"]) >= ICIR_MIN)
    results[nm] = {"panel": p, "ic1": ic1, "ic5": ic5, "ic10": ic10, "cov": cov, "to": to, "passed": passed}
    print(f"{nm:8s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.3f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | "
          f"{'PASS' if passed else 'fail'}")

# ---------------------------------------------------------------------------
# correlation vs persisted library (canonical factor files only)
# ---------------------------------------------------------------------------
def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    dates = pd.bdate_range(start, end)[: payload["n_dates"]]
    return pd.DataFrame(arr, index=dates, columns=payload.get("symbols", SYMBOLS))

def load_lib_factor(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    if sa is None:
        v = d.get("validation") or {}
        sa = v.get("signal_artifact")
        if sa is None:
            sa = (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            M = np.load(p)
            return pd.DataFrame(M, index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        if "data" in sa and "n_dates" in sa:
            try:
                return decode_gzipb64(sa)
            except Exception as e:
                print("  decode err", path, e)
    return None

lib_paths = sorted([os.path.join("factors", f) for f in os.listdir("factors")
                    if f.endswith(".json") and ".2026" not in f and not f.endswith(".bak")])
lib = {}
for p in lib_paths:
    df = load_lib_factor(p)
    if df is not None and len(df) > 100:
        lib[os.path.basename(p)] = df
print(f"\nlibrary loaded: {len(lib)} factors")
for k, v in lib.items():
    print(f"  {k:45s} shape={v.shape} finite={np.isfinite(v.values).sum()}")

passers = [nm for nm, r in results.items() if r["passed"]]
print(f"\npassing gate: {passers}")
for nm in passers:
    row = " ".join(f"{k.split('.')[0][-12:]}:{pair_rho(results[nm]['panel'], v):.2f}" for k, v in lib.items())
    print(f"  {nm:8s} {row}")

def quality(nm):
    r = results[nm]
    return abs(r["ic1"]["ic"]) * abs(r["ic1"]["icir"])

kept = []
for nm in sorted(passers, key=lambda x: -quality(x)):
    ok = all(pair_rho(results[nm]["panel"], v) < RHO_MAX for v in lib.values()) and \
         all(pair_rho(results[nm]["panel"], results[k]["panel"]) < RHO_MAX for k in kept)
    if ok:
        kept.append(nm)
print(f"\ndiverse kept: {kept}")
for nm in kept:
    r = results[nm]
    print(f"  {nm:8s} IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} quality={quality(nm):.5f} "
          f"cov={r['cov']:.3f} to={r['to']:.3f}")

print(f"\nfinished in {time.time()-T0:.1f}s")
