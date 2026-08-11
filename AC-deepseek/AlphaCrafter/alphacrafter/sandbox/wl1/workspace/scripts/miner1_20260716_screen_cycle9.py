"""miner_1 cycle-9 screening: explore new factor families orthogonal to the
current library (mom_10d_skip5, mom_120d_skip5, vix_beta_cond_60x20,
vol_of_vol20x60).  Validation window 2021-01-01..2026-07-15 on the 15-name
cross-asset panel (BTC calendar, 2388 rows).

Admission gates: |daily paper IC| >= 0.007, |ICIR| >= 0.084.
Correlation guard: keep max |spearman rho| vs library artifacts < 0.5.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, ic_analysis, coverage, turnover, decay_analysis

VAL_START = pd.Timestamp("2021-01-01")
CUT = pd.Timestamp("2026-07-15")
panel = pd.read_pickle("scripts/panel_cache.pkl")
close = panel["close"]; open_ = panel["open"]; high = panel["high"]
low = panel["low"]; vol = panel["vol"]; ret = panel["ret"]; macro = panel["macro"]
closes = load_close()          # dict of DataFrames for ic_analysis alignment

# ---------------------------------------------------------------- candidates
def f_rev1(df):
    return -(df["close"] / df["close"].shift(1) - 1.0)

def f_rev2(df):
    return -(df["close"] / df["close"].shift(2) - 1.0)

def f_nclv1(df):
    r = np.log(df["close"] / df["close"].shift(1))
    rv = r.abs().rolling(1).sum()
    return -(r / (rv + 1e-12))

def f_intraday_pos(df, w=5):
    ip = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
    return ip.rolling(w).mean()

def f_range_contr(df, w=20):
    rng = (df["high"] - df["low"]) / df["close"]
    z = (rng - rng.rolling(w).mean()) / (rng.rolling(w).std() + 1e-12)
    return -z

def f_amihud(df, w=20):
    r = (df["close"] / df["close"].shift(1) - 1.0).abs()
    illiq = r / (df["volume"] + 1e-9)
    return illiq.rolling(w).mean()

def f_vol_asym(df, w=20):
    r = np.log(df["close"] / df["close"].shift(1))
    up = r.clip(lower=0).rolling(w).sum()
    dn = r.clip(upper=0).abs().rolling(w).sum()
    return dn / (up + dn + 1e-12) - 0.5          # >0 -> downside-heavy

def f_skew20(df, w=20):
    return -df["close"].pct_change().rolling(w).skew()

def f_highlow_dist(df, w=60):
    return df["close"] / df["close"].rolling(w).max() - 1.0   # -1 far below high

def f_volz(df, w=20):
    rv = np.log(df["close"] / df["close"].shift(1)).abs().rolling(w).sum()
    z = (rv - rv.rolling(4 * w).mean()) / (rv.rolling(4 * w).std() + 1e-12)
    return z

def f_overnight_gap(df):
    gap = df["open"] / df["close"].shift(1) - 1.0
    return -gap

def f_gap_fill(df):
    """intraday reversal: gap then fade (open/prev close - 1) * (close/open - 1)"""
    gap = df["open"] / df["close"].shift(1) - 1.0
    intra = df["close"] / df["open"] - 1.0
    return -np.sign(gap) * intra

def f_vol_ret_corr(df, w=20):
    r = df["close"].pct_change()
    dv = df["volume"].pct_change()
    return r.rolling(w).corr(dv)

def f_er_soft(df, w=20):
    net = (df["close"] / df["close"].shift(w) - 1.0).abs()
    path = np.log(df["close"] / df["close"].shift(1)).abs().rolling(w).sum()
    er = net / (path + 1e-12)
    return (1.0 - er).clip(lower=0) * -(df["close"] / df["close"].shift(5) - 1.0)

CANDIDATES = {
    "rev1":        f_rev1,
    "rev2":        f_rev2,
    "nclv1":       f_nclv1,
    "intraday_pos5": f_intraday_pos,
    "range_contr20": f_range_contr,
    "amihud20":    f_amihud,
    "vol_asym20":  f_vol_asym,
    "skew20":      f_skew20,
    "highlow60":   f_highlow_dist,
    "volz20":      f_volz,
    "overnight_gap": f_overnight_gap,
    "gap_fill":    f_gap_fill,
    "vol_ret_corr20": f_vol_ret_corr,
    "er_soft5":    f_er_soft,
}

def build_panel(fn):
    cols = {}
    for s in SYMBOLS:
        try:
            fv = fn(closes[s])
            if fv is not None and len(fv):
                cols[s] = fv
        except Exception as e:
            pass
    return pd.DataFrame(cols)

# macro-based candidates (built directly on panel)
def macro_factor(sig, w=20, sign=-1.0):
    m = macro[sig].reindex(close.index).ffill()
    chg = m / m.shift(w) - 1.0
    z = (chg - chg.rolling(4 * w).mean()) / (chg.rolling(4 * w).std() + 1e-12)
    return pd.DataFrame({s: sign * z for s in SYMBOLS})

macro_candidates = {
    "dxy_mom20": macro_factor("DXY", 20, -1.0),
    "vix_mom20": macro_factor("VIX", 20, -1.0),
    "usdjpy_mom20": macro_factor("USDJPY", 20, 1.0),
}

# ------------------------------------------------------- library artifacts
lib = {}
for f in os.listdir("factors"):
    if f.endswith(".npy"):
        lib[f] = np.load(os.path.join("factors", f))
# reconstruct no-artifact library factors on the 2388 grid
cal = close.index
def lib_mom120():
    out = {}
    for s in SYMBOLS:
        c = close[s]
        out[s] = c.shift(5) / c.shift(125) - 1.0
    return pd.DataFrame(out).reindex(cal)

def lib_vixbeta():
    vix = macro["VIX"].reindex(cal).ffill()
    vixr = vix.pct_change()
    out = {}
    for s in SYMBOLS:
        r = ret[s]
        beta = r.rolling(60).cov(vixr) / (vixr.rolling(60).var() + 1e-12)
        out[s] = -beta * (vix / vix.shift(20) - 1.0)
    return pd.DataFrame(out).reindex(cal)

def lib_vov():
    out = {}
    for s in SYMBOLS:
        rv = ret[s].rolling(20).std()
        out[s] = rv.rolling(60).std()
    return pd.DataFrame(out).reindex(cal)

lib_panels = {"mom_120d_skip5_recon": lib_mom120(), "vix_beta_cond_60x20_recon": lib_vixbeta(),
              "vol_of_vol20x60_recon": lib_vov()}

def to_art(df):
    arr = np.full((len(cal), len(SYMBOLS)), np.nan, dtype=np.float32)
    for j, s in enumerate(SYMBOLS):
        if s in df.columns:
            arr[:, j] = df[s].values
    return arr

def rho_vs_lib(arr, min_names=4):
    """daily cross-sectional spearman, mean over overlapping rows"""
    targets = {}
    for k, a in lib.items():
        targets[k] = a
    for k, df in lib_panels.items():
        targets[k] = to_art(df)
    out = {}
    for k, a in targets.items():
        n = min(arr.shape[0], a.shape[0])
        vals = []
        for i in range(n):
            x, y = arr[i], a[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= min_names:
                r = spearmanr(x[m], y[m])[0]
                if np.isfinite(r):
                    vals.append(r)
        out[k] = (float(np.mean(vals)) if vals else np.nan, len(vals))
    return out

# ------------------------------------------------------- run screen
results = {}
for name, fn in CANDIDATES.items():
    df = build_panel(fn)
    val = df[df.index >= VAL_START]
    if val.shape[1] == 0:
        print(f"== {name}: EMPTY"); continue
    ic1 = ic_analysis(val, closes, fwd_days=1)
    ic5 = ic_analysis(val, closes, fwd_days=5)
    ic10 = ic_analysis(val, closes, fwd_days=10)
    cov = coverage(val, closes)
    to = turnover(val)
    arr = to_art(df)
    rhos = rho_vs_lib(arr)
    maxrho = max((abs(v[0]) for v in rhos.values() if np.isfinite(v[0])), default=np.nan)
    results[name] = dict(ic1=ic1["ic"], icir1=ic1["icir"], hit1=ic1["hit"],
                         ic5=ic5["ic"], icir5=ic5["icir"], ic10=ic10["ic"],
                         ndates=ic1["n_dates"], nobs=ic1["n_obs"], cov=cov, to=to,
                         maxrho=maxrho, rhos={k: round(v[0], 3) for k, v in rhos.items()})
    print(f"== {name:16} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit={ic1['hit']:.3f} "
          f"| IC5={ic5['ic']:+.4f} IC10={ic10['ic']:+.4f} | cov={cov:.3f} to={to:.3f} "
          f"| ndates={ic1['n_dates']} nobs={ic1['n_obs']} | maxrho={maxrho:.3f}")

for name, df in macro_candidates.items():
    val = df[df.index >= VAL_START]
    ic1 = ic_analysis(val, closes, fwd_days=1)
    ic5 = ic_analysis(val, closes, fwd_days=5)
    ic10 = ic_analysis(val, closes, fwd_days=10)
    cov = coverage(val, closes); to = turnover(val)
    arr = to_art(df)
    rhos = rho_vs_lib(arr)
    maxrho = max((abs(v[0]) for v in rhos.values() if np.isfinite(v[0])), default=np.nan)
    results[name] = dict(ic1=ic1["ic"], icir1=ic1["icir"], hit1=ic1["hit"],
                         ic5=ic5["ic"], icir5=ic5["icir"], ic10=ic10["ic"],
                         ndates=ic1["n_dates"], nobs=ic1["n_obs"], cov=cov, to=to,
                         maxrho=maxrho, rhos={k: round(v[0], 3) for k, v in rhos.items()})
    print(f"== {name:16} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit={ic1['hit']:.3f} "
          f"| IC5={ic5['ic']:+.4f} IC10={ic10['ic']:+.4f} | cov={cov:.3f} to={to:.3f} "
          f"| ndates={ic1['n_dates']} nobs={ic1['n_obs']} | maxrho={maxrho:.3f}")

with open("scripts/miner1_cycle9_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("\n=== SUMMARY TABLE (sorted by |IC1|*|ICIR1|) ===")
for name in sorted(results, key=lambda k: -abs(results[k]["ic1"] * results[k]["icir1"])):
    r = results[name]
    q = abs(r["ic1"] * r["icir1"])
    print(f"{name:16} | q={q:.5f} | IC1={r['ic1']:+.4f} ICIR1={r['icir1']:+.3f} | IC5={r['ic5']:+.4f} "
          f"| cov={r['cov']:.3f} to={r['to']:.3f} | maxrho={r['maxrho']:.3f} | ndates={r['ndates']}")
