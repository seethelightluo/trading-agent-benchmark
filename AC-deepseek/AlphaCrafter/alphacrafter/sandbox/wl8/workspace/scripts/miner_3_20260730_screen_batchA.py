"""miner_3 batch-A exploration screen (2026-07-30).
Tests 9 novel candidate factor families on the 15-asset cross-asset universe.
No persistence here; only exploration to identify promising ideas.
Admission gate (shared): |IC|>=0.0070, |ICIR|>=0.0840 at h=10; orthogonality rho<0.5 vs library.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
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


# ---------------- candidate factor definitions ----------------
def f_autocorr_20x60(c, v, o, h, l, m, win=60, lag=1):
    r = c.pct_change()
    a = r.rolling(win).apply(lambda x: pd.Series(x).autocorr(lag), raw=False)
    return a.replace([np.inf, -np.inf], np.nan)


def f_intraday_vol_share_20(c, v, o, h, l, m, win=20):
    total = c.pct_change()
    intra = (c / o - 1.0)
    return (intra.rolling(win).std() / total.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


def f_var_ratio_10x120(c, v, o, h, l, m, q=10, win=120):
    r = c.pct_change()
    num = r.rolling(win).mean().rolling(q).sum()  # placeholder, replaced below
    # variance ratio: Var(q-day returns)/(q*Var(1-day returns))
    var1 = r.rolling(win).var()
    rq = r.rolling(q).sum()
    varq = rq.rolling(win).var()
    return (varq / (q * var1)).replace([np.inf, -np.inf], np.nan)


def f_fx_beta(c, v, o, h, l, m, name, win=60):
    fx = _align(m[name], c)
    r = c.pct_change()
    fxr = fx.pct_change()
    cov = r.rolling(win).cov(fxr)
    var = fxr.rolling(win).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def f_risk_adj_mom_20x60(c, v, o, h, l, m, short=20, skip=5, volwin=60):
    mom = c.shift(skip) / c.shift(skip + short) - 1.0
    rv = c.pct_change().rolling(volwin).std()
    return (mom / rv).replace([np.inf, -np.inf], np.nan)


def f_month_seasonal(c, v, o, h, l, m, min_years=2):
    r = c.pct_change()
    df = pd.DataFrame({"r": r, "m": c.index.month})
    g = df.groupby("m")["r"].transform("mean") * 21  # approx monthly mean return in pct
    # require at least min_years of history for the month
    cnt = df.groupby("m")["r"].transform("count")
    return g.where(cnt >= 250 * min_years)


def f_ret_skew_30(c, v, o, h, l, m, win=30):
    return c.pct_change().rolling(win).skew()


def f_vol_trend_20x60(c, v, o, h, l, m, short=20, long=60):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    return (v.rolling(short).mean() / v.rolling(long).mean()).replace([np.inf, -np.inf], np.nan)


def f_overnight_share_20(c, v, o, h, l, m, win=20):
    # share of total 20d absolute move coming from overnight gaps
    ovn = (o / c.shift(1) - 1.0).abs()
    tot = c.pct_change().abs()
    return (ovn.rolling(win).sum() / tot.rolling(win).sum()).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "autocorr_20x60": {"fn": f_autocorr_20x60, "params": {"win": 60, "lag": 1}},
    "intraday_vol_share_20": {"fn": f_intraday_vol_share_20, "params": {"win": 20}},
    "var_ratio_10x120": {"fn": f_var_ratio_10x120, "params": {"q": 10, "win": 120}},
    "fx_beta_usdjpy_60": {"fn": f_fx_beta, "params": {"name": "USDJPY", "win": 60}},
    "fx_beta_usdcny_60": {"fn": f_fx_beta, "params": {"name": "USDCNY", "win": 60}},
    "risk_adj_mom_20x60": {"fn": f_risk_adj_mom_20x60, "params": {"short": 20, "skip": 5, "volwin": 60}},
    "month_seasonal": {"fn": f_month_seasonal, "params": {"min_years": 2}},
    "ret_skew_30": {"fn": f_ret_skew_30, "params": {"win": 30}},
    "vol_trend_20x60": {"fn": f_vol_trend_20x60, "params": {"short": 20, "long": 60}},
    "overnight_share_20": {"fn": f_overnight_share_20, "params": {"win": 20}},
}

# forward returns per horizon (compute once)
fwd = {h: fwd_returns(close, h) for h in HORIZONS}

print(f"universe: {len(ASSETS)} assets, panel dates {close.index.min().date()}..{close.index.max().date()}", flush=True)
for fid, spec in FACTORS.items():
    t1 = time.time()
    panel = factor_panel(spec["fn"], close, vol, open_, high, low, macro, **spec["params"])
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay, ic_by_h = {}, {}
    for h in HORIZONS:
        ic = ic_series(panel, fwd[h])
        ic_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    icm = ic_by_h[10]
    ic = float(icm.mean()) if len(icm) else np.nan
    icir = float(icm.mean() / icm.std()) if len(icm) > 2 else np.nan
    hit = float((icm > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((icm < 0).mean())
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"\n=== {fid} === gate={'PASS' if gate else 'FAIL'} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(h): round(decay[h],4) for h in HORIZONS} }", flush=True)
    print(f"  panel shape={panel.shape} valid={int(panel.notna().sum().sum())}", flush=True)

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
