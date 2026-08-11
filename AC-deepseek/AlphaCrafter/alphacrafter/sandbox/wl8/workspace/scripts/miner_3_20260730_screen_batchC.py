"""miner_3 batch-C screen (2026-07-30): new orthogonal candidate families.
Targets NOT previously tried in evicted/rejected library or prior batch screens:
  - park_vol_ratio_20x60 : 20d Parkinson vol / 60d Parkinson vol (vol term-structure)
  - gold_beta_60         : 60d beta of asset returns to XAU returns
  - cn10y_beta_60        : 60d beta of asset returns to CN10Y yield changes
  - beta_asym_60         : downside beta - upside beta vs equal-weight market (60d)
  - ret_autocorr_20      : lag-1 autocorrelation of daily returns (20d)
  - volume_z_20          : 20d mean volume z-score vs trailing 60d
  - semi_vol_ratio_20    : downside dev / upside dev of daily returns (20d)
  - vol_term_60          : 20d close-vol / 60d close-vol ratio
  - skew_60              : 60d rolling skewness of daily returns (30d failed; 60d variant)
  - range_pos_20         : close position inside 20d high-low range (short-horizon)
Admission h=10: |IC|>=0.007, |ICIR|>=0.084, and max |rho| < 0.5 vs the 3 CURRENT
library factors (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
macro["__market_close__"] = close.mean(axis=1)
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}")

# ---- current library panels ----
def load_lib_artifacts(fids):
    lib = {}
    for fid in fids:
        d = json.load(open(f"factors/{fid}.json"))
        data = d["validation"]["signal_artifact"]["data"]
        raw = base64.b64decode(data)
        csv_text = zlib.decompress(raw).decode()
        panel = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib

LIB_FIDS = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]
lib = load_lib_artifacts(LIB_FIDS)

def lib_corr(panel, method="pearson"):
    out = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            out[fid] = None
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            out[fid] = None
            continue
        if method == "pearson":
            out[fid] = float(np.corrcoef(a[m], b[m])[0, 1])
        else:
            out[fid] = float(spearmanr(a[m], b[m])[0])
    vals = [abs(v) for v in out.values() if v is not None and np.isfinite(v)]
    return out, (max(vals) if vals else None)

# ---- factor functions (dense per-asset calendars) ----
def f_park_vol_ratio(c, v, o, h, l, m, s=20, long=60):
    pr = np.log(h / l)
    pv_s = (pr.rolling(s).mean() / np.sqrt(4 * np.log(2))) * np.sqrt(252)
    pv_l = (pr.rolling(long).mean() / np.sqrt(4 * np.log(2))) * np.sqrt(252)
    return (pv_s / pv_l.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_gold_beta(c, v, o, h, l, m, win=60):
    g = m["XAU_close"]
    ri = c.pct_change(); rg = g.pct_change()
    beta = ri.rolling(win).cov(rg) / rg.rolling(win).var()
    return beta.replace([np.inf, -np.inf], np.nan)

def f_cn10y_beta(c, v, o, h, l, m, win=60):
    y = m["CN10Y_close"]
    ri = c.pct_change(); ry = y.pct_change()
    beta = ri.rolling(win).cov(ry) / ry.rolling(win).var()
    return beta.replace([np.inf, -np.inf], np.nan)

def f_beta_asym(c, v, o, h, l, m, win=60):
    mkt = m["__market_close__"]
    ri = c.pct_change(); rm = mkt.pct_change()
    df = pd.concat([ri, rm], axis=1).dropna()
    if len(df) < win:
        return pd.Series(np.nan, index=ri.index)
    rb = df.iloc[:, 0]; rm_ = df.iloc[:, 1]
    down = rm_ < 0; up = rm_ >= 0
    bd = rb[down].rolling(win).cov(rm_[down]) / rm_[down].rolling(win).var()
    bu = rb[up].rolling(win).cov(rm_[up]) / rm_[up].rolling(win).var()
    return (bd - bu).reindex(ri.index).replace([np.inf, -np.inf], np.nan)

def f_ret_autocorr(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: x.autocorr() if len(x) > 3 else np.nan, raw=False)

def f_volume_z(c, v, o, h, l, m, win=20, base=60):
    vv = v.replace(0, np.nan)
    mu = vv.rolling(base).mean(); sd = vv.rolling(base).std()
    return ((vv - mu) / sd.replace(0, np.nan)).rolling(win).mean()

def f_semi_vol_ratio(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    dd = r.where(r < 0, np.nan).rolling(win).std()
    ud = r.where(r > 0, np.nan).rolling(win).std()
    return (dd / ud.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_vol_term(c, v, o, h, l, m, s=20, long=60):
    r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(long).std()
    return (vs / vl.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_skew_60(c, v, o, h, l, m, win=60):
    return c.pct_change().rolling(win).skew()

def f_range_pos_20(c, v, o, h, l, m, win=20):
    hi = h.rolling(win).max(); lo = l.rolling(win).min()
    return ((c - lo) / (hi - lo).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# XAU / CN10Y close series for beta factors
macro["XAU_close"] = close["XAU"].astype(float)
macro["CN10Y_close"] = close["CN10Y"].astype(float)

CANDIDATES = [
    ("park_vol_ratio_20x60", f_park_vol_ratio, "20d/60d Parkinson vol ratio"),
    ("gold_beta_60", f_gold_beta, "60d beta to XAU returns"),
    ("cn10y_beta_60", f_cn10y_beta, "60d beta to CN10Y yield changes"),
    ("beta_asym_60", f_beta_asym, "downside beta - upside beta (60d)"),
    ("ret_autocorr_20", f_ret_autocorr, "lag-1 return autocorrelation (20d)"),
    ("volume_z_20", f_volume_z, "20d mean volume z-score vs 60d"),
    ("semi_vol_ratio_20", f_semi_vol_ratio, "downside/upside dev ratio (20d)"),
    ("vol_term_60", f_vol_term, "20d/60d close-vol ratio"),
    ("skew_60", f_skew_60, "60d rolling return skewness"),
    ("range_pos_20", f_range_pos_20, "close position in 20d high-low range"),
]

HORIZONS = (1, 2, 3, 5, 10, 20)
results = {}
for name, fn, desc in CANDIDATES:
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay = {}
    for h in HORIZONS:
        ic = ic_series(panel, fwd_returns(close, h))
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic10 = ic_series(panel, fwd_returns(close, 10))
    ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic10 < 0).mean())
    pe_map, maxpe = lib_corr(panel, "pearson")
    sp_map, maxsp = lib_corr(panel, "spearman")
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE and maxpe < 0.5
    results[name] = dict(ic=ic, icir=icir, hit=hit, n=len(ic10), cov_ad=cov_ad,
                         cov_ge8=cov_ge8, to=to, decay=decay, pe=pe_map, sp=sp_map,
                         maxpe=maxpe, maxsp=maxsp, ok=ok)
    print(f"\n=== {name} [{desc}] ===")
    print(f"  IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} cov_ad={cov_ad:.3f} "
          f"cov8={cov_ge8:.3f} turnover={to:.3f}")
    print(f"  decay: { {str(k): round(v, 4) for k, v in decay.items()} }")
    print(f"  pearson vs lib: { {k: round(v, 3) if v is not None else None for k, v in pe_map.items()} } max|rho|={maxpe:.3f}")
    print(f"  GATE(IC>={IC_GATE}, ICIR>={ICIR_GATE}, rho<0.5): {'PASS' if ok else 'FAIL'}")

print(f"\n===== SUMMARY (h=10) =====  elapsed={time.time()-t0:.1f}s")
for name, r in results.items():
    print(f"{name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['hit']:.3f} "
          f"n={r['n']} cov={r['cov_ad']:.3f} to={r['to']:.2f} maxpe={r['maxpe']:.3f} -> {'PASS' if r['ok'] else 'fail'}")
