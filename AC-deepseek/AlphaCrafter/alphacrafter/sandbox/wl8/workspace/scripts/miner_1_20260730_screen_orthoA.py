"""miner_1 screen batch A: volatility-shape / time-series-structure factors
orthogonal to the library's usdcny_beta_60 (FX-beta) signal.
Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at admission horizon 10.
Also report pooled Spearman rho vs library artifact (eviction gate < 0.5).
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE, ASSETS,
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


def f_autocorr_20(c, v, o, h, l, m):
    """Lag-1 autocorrelation of daily returns over 20d window."""
    r = c.pct_change()
    x = r.copy()
    x1 = r.shift(1)
    mu = r.rolling(20).mean()
    s2 = r.rolling(20).var()
    cov = ((r - mu) * (x1 - x1.rolling(20).mean())).rolling(20).mean()
    return (cov / s2).replace([np.inf, -np.inf], np.nan)


def f_idio_vol_60(c, v, o, h, l, m, bench="SPX"):
    """Residual vol after 60d regression on benchmark (orthogonal risk)."""
    b = _align(m[bench], c)
    r = c.pct_change()
    br = b.pct_change()
    beta = (r.rolling(60).cov(br) / br.rolling(60).var())
    resid = r - beta * br
    return resid.rolling(60).std()


def f_overnight_ratio_20(c, v, o, h, l, m):
    """Overnight contribution: |open/prev_close-1| / (|open/prev_close-1| + |close/open-1|), 20d mean."""
    prev_close = c.shift(1)
    ovn = (o / prev_close - 1.0).abs()
    intra = (c / o - 1.0).abs()
    denom = (ovn + intra).replace(0, np.nan)
    return (ovn / denom).rolling(20).mean()


def f_vol_term_10x60(c, v, o, h, l, m):
    """Realized vol term structure: 10d vol / 60d vol (contango vs inversion)."""
    r = c.pct_change()
    v10 = r.rolling(10).std()
    v60 = r.rolling(60).std()
    return (v10 / v60).replace([np.inf, -np.inf], np.nan)


def f_jump_freq_20(c, v, o, h, l, m, k=2.0, win=60):
    """Fraction of last 20 days with |r| > k*rolling std(60)."""
    r = c.pct_change()
    sd = r.rolling(win).std()
    jump = (r.abs() > k * sd).astype(float)
    return jump.rolling(20).mean()


def f_drawup_60(c, v, o, h, l, m):
    """Max run-up share of total max-run range over 60d."""
    roll = c.rolling(60)
    maxup = (c / roll.max() - 1.0).abs()
    maxdn = (c / roll.min() - 1.0).abs()
    denom = (maxup + maxdn).replace(0, np.nan)
    return (maxup / denom)


def f_parkinson_ratio_20(c, v, o, h, l, m):
    """Parkinson vol (intraday range) / close-close vol, 20d."""
    r = c.pct_change()
    cc = r.rolling(20).std()
    hl = np.log(h / l)
    park = np.sqrt((hl ** 2).rolling(20).mean() / (4 * np.log(2)))
    return (park / cc).replace([np.inf, -np.inf], np.nan)


def f_skew_60(c, v, o, h, l, m):
    """Rolling skewness of daily returns, 60d."""
    return c.pct_change().rolling(60).skew()


def f_vol_cluster_20(c, v, o, h, l, m):
    """Autocorr of squared returns (vol clustering) over 20d."""
    r2 = (c.pct_change() ** 2)
    r2l = r2.shift(1)
    mu = r2.rolling(20).mean()
    var = r2.rolling(20).var()
    cov = ((r2 - mu) * (r2l - r2l.rolling(20).mean())).rolling(20).mean()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def f_candle_asym_20(c, v, o, h, l, m):
    """Candle asymmetry: mean upper shadow / (upper+lower shadow) over 20d.
    Upper shadow = high - max(open,close); lower = min(open,close) - low."""
    up_sh = h - np.maximum(o, c)
    lo_sh = np.minimum(o, c) - l
    denom = (up_sh + lo_sh).replace(0, np.nan)
    return (up_sh / denom).rolling(20).mean()


FACTORS = {
    "autocorr_20": {"fn": f_autocorr_20, "params": {}},
    "idio_vol_60_spx": {"fn": f_idio_vol_60, "params": {"bench": "SPX"}},
    "overnight_ratio_20": {"fn": f_overnight_ratio_20, "params": {}},
    "vol_term_10x60": {"fn": f_vol_term_10x60, "params": {}},
    "jump_freq_20": {"fn": f_jump_freq_20, "params": {}},
    "drawup_60": {"fn": f_drawup_60, "params": {}},
    "parkinson_ratio_20": {"fn": f_parkinson_ratio_20, "params": {}},
    "skew_60": {"fn": f_skew_60, "params": {}},
    "vol_cluster_20": {"fn": f_vol_cluster_20, "params": {}},
    "candle_asym_20": {"fn": f_candle_asym_20, "params": {}},
}

# library panel (current effective)
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
print(f"close panel {close.shape}, dates {close.index.min()}..{close.index.max()}", flush=True)
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
    for h in HORIZONS:
        ic_h = ic_series(panel, fwd[h])
        decay[h] = float(ic_h.mean()) if len(ic_h) else np.nan
    rho, nrho = spearman_pooled(panel, lib_panel)
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    ortho = np.isfinite(rho) and abs(rho) < 0.5
    print(f"\n=== {fid} === gate={'PASS' if gate else 'FAIL'} rho_vs_lib={'OK' if ortho else 'RISK' if np.isfinite(rho) else 'NA'} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(h): round(decay[h],4) for h in HORIZONS} }", flush=True)
    print(f"  lib_spearman_rho={rho if not np.isfinite(rho) else round(rho,4)} (n={nrho})", flush=True)

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
