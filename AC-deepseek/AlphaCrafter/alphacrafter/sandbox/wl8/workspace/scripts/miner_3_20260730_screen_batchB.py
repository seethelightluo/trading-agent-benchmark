"""miner_3 batch-B quick screen (2026-07-30): extra orthogonal candidates.
month_seasonal (fixed), eurusd_beta_60, vix_cond_mom_20x60 (regime-switched
momentum), usdcny_mom_cond_60 (directional CNY-beta carry).
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}
HORIZONS = (1, 2, 3, 5, 10, 20)


def _align(series, c):
    return series.reindex(c.index).ffill()


def f_fx_beta(c, v, o, h, l, m, name, win=60):
    fx = _align(m[name], c)
    r = c.pct_change()
    fxr = fx.pct_change()
    return (r.rolling(win).cov(fxr) / fxr.rolling(win).var()).replace([np.inf, -np.inf], np.nan)


def f_month_seasonal(c, v, o, h, l, m, min_years=2):
    r = c.pct_change()
    df = pd.DataFrame({"r": r, "m": c.index.month})
    g = df.groupby("m")["r"].transform("mean")
    cnt = df.groupby("m")["r"].transform("count")
    return g.where(cnt >= 20 * 12 * min_years)


def f_vix_cond_mom(c, v, o, h, l, m, short=20, skip=5, vixwin=60):
    vix = _align(m["VIX"], c)
    mom = c.shift(skip) / c.shift(skip + short) - 1.0
    z = (vix - vix.rolling(vixwin).mean()) / vix.rolling(vixwin).std()
    sign = np.where(z > 1.0, -1.0, 1.0)
    return (mom * sign).replace([np.inf, -np.inf], np.nan)


def f_usdcny_mom_cond(c, v, o, h, l, m, win=60, momwin=20):
    fx = _align(m["USDCNY"], c)
    r = c.pct_change()
    fxr = fx.pct_change()
    beta = (r.rolling(win).cov(fxr) / fxr.rolling(win).var())
    fxmom = fx / fx.shift(momwin) - 1.0
    return (beta * np.sign(fxmom)).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "month_seasonal": {"fn": f_month_seasonal, "params": {"min_years": 2}},
    "eurusd_beta_60": {"fn": f_fx_beta, "params": {"name": "EURUSD", "win": 60}},
    "vix_cond_mom_20x60": {"fn": f_vix_cond_mom, "params": {"short": 20, "skip": 5, "vixwin": 60}},
    "usdcny_mom_cond_60": {"fn": f_usdcny_mom_cond, "params": {"win": 60, "momwin": 20}},
}

# library panels
def load_lib_panels():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib

lib = load_lib_panels()

def spearman_pooled(a_panel, b_panel):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), int(m.sum())


fwd = {h: fwd_returns(close, h) for h in HORIZONS}
for fid, spec in FACTORS.items():
    t1 = time.time()
    panel = factor_panel(spec["fn"], close, vol, open_, high, low, macro, **spec["params"])
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay = {}
    for h in HORIZONS:
        ic = ic_series(panel, fwd[h])
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    icm = ic_series(panel, fwd[10])
    ic = float(icm.mean()) if len(icm) else np.nan
    icir = float(icm.mean() / icm.std()) if len(icm) > 2 else np.nan
    hit = float((icm > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((icm < 0).mean())
    rho_map = {}
    for lfid, lp in lib.items():
        r, _ = spearman_pooled(panel, lp)
        rho_map[lfid] = r
    maxrho = max((abs(r) for r in rho_map.values() if np.isfinite(r)), default=np.nan)
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    ortho = np.isfinite(maxrho) and maxrho < 0.5
    print(f"\n=== {fid} === gate={'PASS' if gate else 'FAIL'} ortho={'OK' if ortho else 'RISK'} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(h): round(decay[h],4) for h in HORIZONS} }", flush=True)
    print(f"  lib rho: { {k: (round(v,4) if np.isfinite(v) else None) for k,v in rho_map.items()} } max={maxrho:.4f}", flush=True)

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
