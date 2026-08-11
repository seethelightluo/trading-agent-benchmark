"""miner_1 screen batch B: drawup variants + fixed idio_vol + volume/range candidates."""
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


def f_drawup(c, v, o, h, l, m, win=60):
    roll = c.rolling(win)
    maxup = (c / roll.max() - 1.0).abs()
    maxdn = (c / roll.min() - 1.0).abs()
    denom = (maxup + maxdn).replace(0, np.nan)
    return (maxup / denom)


def f_dist_high(c, v, o, h, l, m, win=60):
    """Signed distance from rolling max: close/rolling_max - 1 (<=0, 0 at highs)."""
    return (c / c.rolling(win).max() - 1.0)


def f_dist_low(c, v, o, h, l, m, win=60):
    """Signed distance from rolling min: close/rolling_min - 1 (>=0, 0 at lows)."""
    return (c / c.rolling(win).min() - 1.0)


def f_idio_vol(c, v, o, h, l, m, bench="SPX", win=60):
    """Residual vol after regression on benchmark close (from tradable panel)."""
    b = close[bench].reindex(c.index)
    r = c.pct_change()
    br = b.pct_change()
    beta = (r.rolling(win).cov(br) / br.rolling(win).var())
    resid = r - beta * br
    return resid.rolling(win).std()


def f_vol_trend(c, v, o, h, l, m, short=20, long_=60):
    """Volume trend: 20d mean volume / 60d mean volume."""
    if v is None:
        return pd.Series(np.nan, index=c.index)
    vs = v.rolling(short).mean()
    vl = v.rolling(long_).mean()
    return (vs / vl).replace([np.inf, -np.inf], np.nan)


def f_vol_z(c, v, o, h, l, m, win=60):
    """Volume z-score vs 60d mean/std."""
    if v is None:
        return pd.Series(np.nan, index=c.index)
    mu = v.rolling(win).mean()
    sd = v.rolling(win).std()
    return ((v - mu) / sd).replace([np.inf, -np.inf], np.nan)


def f_eff_ratio(c, v, o, h, l, m, win=60):
    """Kaufman efficiency ratio: |close - close.shift(win)| / sum(|ret|, win)."""
    r = c.pct_change().abs()
    path = r.rolling(win).sum()
    net = (c - c.shift(win)).abs()
    return (net / path).replace([np.inf, -np.inf], np.nan)


def f_range_vol(c, v, o, h, l, m, win=20):
    """Mean (high-low)/close over window."""
    return ((h - l) / c).rolling(win).mean()


def f_hl_vol_corr(c, v, o, h, l, m, win=60):
    """Correlation between daily range and volume over 60d."""
    if v is None:
        return pd.Series(np.nan, index=c.index)
    rng = (h - l) / c
    return rng.rolling(win).corr(v)


FACTORS = {
    "drawup_40": {"fn": f_drawup, "params": {"win": 40}},
    "drawup_120": {"fn": f_drawup, "params": {"win": 120}},
    "dist_high_60": {"fn": f_dist_high, "params": {"win": 60}},
    "dist_low_60": {"fn": f_dist_low, "params": {"win": 60}},
    "idio_vol_60_spx": {"fn": f_idio_vol, "params": {"bench": "SPX", "win": 60}},
    "vol_trend_20x60": {"fn": f_vol_trend, "params": {}},
    "vol_z_60": {"fn": f_vol_z, "params": {"win": 60}},
    "eff_ratio_60": {"fn": f_eff_ratio, "params": {"win": 60}},
    "range_20": {"fn": f_range_vol, "params": {"win": 20}},
    "hl_vol_corr_60": {"fn": f_hl_vol_corr, "params": {"win": 60}},
}

d = json.load(open("factors/usdcny_beta_60.json"))
raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
lib_panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
lib_panel.index = pd.DatetimeIndex(lib_panel.index)


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
    icm = ic_series(panel, fwd[10])
    ic = float(icm.mean()) if len(icm) else np.nan
    icir = float(icm.mean() / icm.std()) if len(icm) > 2 else np.nan
    hit = float((icm > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((icm < 0).mean())
    decay = {}
    for hh in HORIZONS:
        ic_h = ic_series(panel, fwd[hh])
        decay[hh] = float(ic_h.mean()) if len(ic_h) else np.nan
    rho, nrho = spearman_pooled(panel, lib_panel)
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    ortho = np.isfinite(rho) and abs(rho) < 0.5
    print(f"\n=== {fid} === gate={'PASS' if gate else 'FAIL'} rho_vs_lib={'OK' if ortho else 'RISK' if np.isfinite(rho) else 'NA'} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(hh): round(decay[hh],4) for hh in HORIZONS} }", flush=True)
    print(f"  lib_spearman_rho={rho if not np.isfinite(rho) else round(rho,4)} (n={nrho})", flush=True)

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
