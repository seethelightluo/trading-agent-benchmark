"""miner_2 2026-07-30 -- Batch C: structure/dynamics factor families orthogonal to
library (esp. yield_beta_cond_60x20 gatekeeper).

Families:
  1. Return autocorrelation (trend persistence / mean reversion)
  2. Trend sign-alignment (consistency of daily signs with trend)
  3. Volume concentration (Herfindahl of daily volume shares)
  4. Volume autocorrelation (serial dependence of participation)
  5. Overnight gap mean (information-arrival direction)
  6. Price acceleration (momentum-of-momentum with skip)
  7. Range squeeze (5d/60d mean daily range)
  8. OBV slope (accumulated volume flow)
  9. Day-of-week return spread (calendar microstructure)

Library correlation: Pearson AND Spearman vs ALL 3 effective factors
(mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).
"""
import sys
import json
import base64
import zlib
import io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets", flush=True)

EFFECTIVE = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]


def decode(fid):
    d = json.load(open(f"factors/{fid}.json"))
    raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return d, p


lib = {}
for fid in EFFECTIVE:
    _, lib[fid] = decode(fid)
print(f"Library loaded: {list(lib.keys())}", flush=True)


def lib_corr(panel, method="spearman"):
    """Max |corr| vs each library panel on common dates/assets."""
    out = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 30 or len(cols) < 3:
            out[fid] = np.nan
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            out[fid] = np.nan
            continue
        if method == "spearman":
            rho, _ = spearmanr(a[m], b[m])
        else:
            rho = float(np.corrcoef(a[m], b[m])[0, 1])
        out[fid] = abs(float(rho))
    return out


# ---------------- candidate factor functions ----------------

def f_autocorr_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: x.iloc[:-1].corr(x.iloc[1:]) if x.notna().sum() >= 10 else np.nan, raw=False)


def f_autocorr_60(c, v, o, h, l, m, win=60):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: x.iloc[:-1].corr(x.iloc[1:]) if x.notna().sum() >= 30 else np.nan, raw=False)


def f_trend_align_20(c, v, o, h, l, m, win=20):
    """Fraction of daily returns in window with same sign as window return."""
    r = c.pct_change()
    tot = r.rolling(win).sum()
    sgn = np.sign(tot)
    align = r.rolling(win).apply(lambda x: (np.sign(x) == np.sign(x.sum())).mean() if x.notna().sum() >= 10 else np.nan, raw=True)
    # multiply by sign of window return so positive = consistent up-trend
    return align * sgn


def f_vol_conc_20(c, v, o, h, l, m, win=20):
    """Herfindahl index of daily volume shares (concentration of activity)."""
    v = v.replace(0, np.nan)
    sh = v / v.rolling(win).sum()
    return (sh ** 2).rolling(win).sum()


def f_vol_autocorr_20(c, v, o, h, l, m, win=20):
    v = v.replace(0, np.nan)
    lv = np.log(v)
    return lv.rolling(win).apply(lambda x: x.iloc[:-1].corr(x.iloc[1:]) if x.notna().sum() >= 10 else np.nan, raw=False)


def f_gap_mean_20(c, v, o, h, l, m, win=20):
    """Mean overnight gap (open/prev_close - 1) over window."""
    gap = o / c.shift(1) - 1.0
    return gap.rolling(win).mean()


def f_accel_20(c, v, o, h, l, m, skip=5, span=10):
    """Price acceleration: momentum(span,skip) - momentum(span,skip) shifted by span."""
    mom = c.shift(skip) / c.shift(skip + span) - 1.0
    return mom - mom.shift(span)


def f_range_squeeze_5x60(c, v, o, h, l, m, short=5, long=60):
    rng = (h - l) / c
    return rng.rolling(short).mean() / rng.rolling(long).mean()


def f_obv_slope_20(c, v, o, h, l, m, win=20):
    """Slope of OBV (sum sign*volume) over window, normalized by mean volume."""
    r = c.pct_change()
    obv = (np.sign(r) * v).rolling(win).sum()
    return obv / v.rolling(win).mean()


def f_dow_spread_60(c, v, o, h, l, m, win=60):
    """Mean return on 'worst avg weekday' minus 'best avg weekday' over trailing window
    (calendar microstructure persistence)."""
    r = c.pct_change()
    df = pd.DataFrame({"r": r.values, "dow": r.index.dayofweek}).dropna()
    out = pd.Series(np.nan, index=r.index)
    for i in range(win, len(df)):
        w = df.iloc[i - win:i]
        g = w.groupby("dow")["r"].mean()
        if len(g) >= 4:
            out.iloc[i] = g.min() - g.max()
    return out.reindex(r.index)


def f_upvol_ratio_20(c, v, o, h, l, m, win=20):
    """Up-volume / total volume ratio over window (buying pressure participation)."""
    r = c.pct_change()
    upv = (v * (r > 0)).rolling(win).sum()
    return upv / v.rolling(win).sum()


def f_williams_r_30(c, v, o, h, l, m, win=30):
    """Williams %R: distance of close from 30d high/low (price location alt)."""
    hi = h.rolling(win).max()
    lo = l.rolling(win).min()
    return (hi - c) / (hi - lo).replace(0, np.nan)


CANDIDATES = [
    ("autocorr_20", f_autocorr_20, dict(win=20), "lag-1 return autocorr 20d"),
    ("autocorr_60", f_autocorr_60, dict(win=60), "lag-1 return autocorr 60d"),
    ("trend_align_20", f_trend_align_20, dict(win=20), "trend sign-alignment 20d"),
    ("vol_conc_20", f_vol_conc_20, dict(win=20), "volume Herfindahl 20d"),
    ("vol_autocorr_20", f_vol_autocorr_20, dict(win=20), "log-volume autocorr 20d"),
    ("gap_mean_20", f_gap_mean_20, dict(win=20), "mean overnight gap 20d"),
    ("accel_20", f_accel_20, dict(skip=5, span=10), "price acceleration 10d/10d"),
    ("range_squeeze_5x60", f_range_squeeze_5x60, dict(short=5, long=60), "range squeeze 5/60"),
    ("obv_slope_20", f_obv_slope_20, dict(win=20), "OBV slope 20d"),
    ("dow_spread_60", f_dow_spread_60, dict(win=60), "weekday spread 60d"),
    ("upvol_ratio_20", f_upvol_ratio_20, dict(win=20), "up-volume share 20d"),
    ("williams_r_30", f_williams_r_30, dict(win=30), "Williams %R 30d"),
]

print("\n===== BATCH C SCREEN =====", flush=True)
for name, fn, prm, desc in CANDIDATES:
    res = validate_factor(fn, close, vol, open_, high, low, macro, **prm)
    panel = res["panel"]
    rho_s = lib_corr(panel, "spearman")
    rho_p = lib_corr(panel, "pearson")
    max_s = max([v for v in rho_s.values() if np.isfinite(v)], default=0.0)
    max_p = max([v for v in rho_p.values() if np.isfinite(v)], default=0.0)
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE and max_s < 0.5
    print(f"[{'PASS' if ok else 'fail'}] {name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} "
          f"hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']:5d} cov={res['coverage_asset_days']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} | maxRho_s={max_s:.3f} maxRho_p={max_p:.3f}", flush=True)
    print(f"      decay={res['decay_ic_by_horizon']}", flush=True)
    print(f"      rho_s={ {k: (round(v,3) if np.isfinite(v) else None) for k,v in rho_s.items()} }", flush=True)
print("\ndone", flush=True)
