"""miner_2 batch-E screen v2 (2026-07-30): orthogonal-to-FX-beta factor ideas.
All factors vectorized for speed. Gate: |IC|>=0.007, |ICIR|>=0.084 at h=10.
Library member: usdcny_beta_60.
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


def f_autocorr(c, v, o, h, l, m, win=10):
    r = c.pct_change()
    num = r.rolling(win).cov(r.shift(1))
    den = r.rolling(win).var()
    return (num / den).replace([np.inf, -np.inf], np.nan)


def f_vol_ratio(c, v, o, h, l, m, short=5, long_=60):
    r = c.pct_change()
    return (r.rolling(short).std() / r.rolling(long_).std()).replace([np.inf, -np.inf], np.nan)


def f_abn_vol(c, v, o, h, l, m, win=20):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    return (v / v.rolling(win).mean()).replace([np.inf, -np.inf], np.nan)


def f_abn_vol_z(c, v, o, h, l, m, win=20):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    return ((v - v.rolling(win).mean()) / v.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


def f_skew_20(c, v, o, h, l, m, win=20):
    return c.pct_change().rolling(win).skew()


def f_up_ratio(c, v, o, h, l, m, win):
    r = c.pct_change()
    return (r > 0).rolling(win).mean()


def f_gap_20(c, v, o, h, l, m, win=20):
    if o is None:
        return pd.Series(np.nan, index=c.index)
    gap = (o - c.shift(1)).abs()
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return (gap.rolling(win).mean() / tr.rolling(win).mean()).replace([np.inf, -np.inf], np.nan)


def f_body_10(c, v, o, h, l, m, win=10):
    if o is None or h is None or l is None:
        return pd.Series(np.nan, index=c.index)
    rng = (h - l).replace(0, np.nan)
    return ((c - o).abs() / rng).rolling(win).mean()


def f_hl_pos_10(c, v, o, h, l, m, win=10):
    if h is None or l is None:
        return pd.Series(np.nan, index=c.index)
    rng = h.rolling(win).max() - l.rolling(win).min()
    return ((c - l.rolling(win).min()) / rng).replace([np.inf, -np.inf], np.nan)


def f_dd_vol_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    roll_max = c.rolling(win).max()
    dd = (c / roll_max - 1.0).rolling(win).min()
    return (dd / r.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


def f_fx_beta(c, v, o, h, l, m, name, win=60):
    fx = _align(m[name], c)
    r = c.pct_change()
    fxr = fx.pct_change()
    return (r.rolling(win).cov(fxr) / fxr.rolling(win).var()).replace([np.inf, -np.inf], np.nan)


def f_weekday(c, v, o, h, l, m, lookback=252):
    r = c.pct_change()
    g = pd.DataFrame({"r": r, "d": c.index.dayofweek})
    means = g.groupby("d")["r"].transform("mean")
    cnt = g.groupby("d")["r"].transform("count")
    return means.where(cnt >= 100)


def f_ret_rev_3(c, v, o, h, l, m, win=3):
    return -c.pct_change(win)


def f_daily_range_20(c, v, o, h, l, m, win=20):
    if h is None or l is None:
        return pd.Series(np.nan, index=c.index)
    r = c.pct_change()
    rng = (h - l) / c
    return (rng.rolling(win).mean() / r.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "autocorr_10": {"fn": f_autocorr, "params": {"win": 10}},
    "autocorr_60": {"fn": f_autocorr, "params": {"win": 60}},
    "vol_ratio_5x60": {"fn": f_vol_ratio, "params": {"short": 5, "long_": 60}},
    "abn_vol_20": {"fn": f_abn_vol, "params": {"win": 20}},
    "abn_vol_z_20": {"fn": f_abn_vol_z, "params": {"win": 20}},
    "skew_20": {"fn": f_skew_20, "params": {"win": 20}},
    "up_ratio_20": {"fn": f_up_ratio, "params": {"win": 20}},
    "up_ratio_60": {"fn": f_up_ratio, "params": {"win": 60}},
    "gap_20": {"fn": f_gap_20, "params": {"win": 20}},
    "body_10": {"fn": f_body_10, "params": {"win": 10}},
    "hl_pos_10": {"fn": f_hl_pos_10, "params": {"win": 10}},
    "dd_vol_20": {"fn": f_dd_vol_20, "params": {"win": 20}},
    "wti_beta_60": {"fn": f_fx_beta, "params": {"name": "WTI", "win": 60}},
    "xau_beta_60": {"fn": f_fx_beta, "params": {"name": "XAU", "win": 60}},
    "dxy_beta_60": {"fn": f_fx_beta, "params": {"name": "DXY", "win": 60}},
    "weekday_ret": {"fn": f_weekday, "params": {"lookback": 252}},
    "ret_rev_3": {"fn": f_ret_rev_3, "params": {"win": 3}},
    "daily_range_20": {"fn": f_daily_range_20, "params": {"win": 20}},
}

# market correlation factor (panel context)
r_all = close.pct_change()
mkt = r_all.mean(axis=1)
mkt_corr_panel = r_all.rolling(60).corr(mkt).replace([np.inf, -np.inf], np.nan)


def load_lib():
    lib = {}
    d = json.load(open("factors/usdcny_beta_60.json"))
    art = d["validation"]["signal_artifact"]
    raw = base64.b64decode(art["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    lib["usdcny_beta_60"] = p
    return lib


lib = load_lib()


def spearman_pooled(a_panel, b_panel):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan, int(m.sum())
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), int(m.sum())


fwd = {h: fwd_returns(close, h) for h in HORIZONS}
results = []
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
    rho_sp, n_sp = spearman_pooled(panel, lib["usdcny_beta_60"])
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"=== {fid} === gate={'PASS' if gate else 'FAIL'} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(h): round(decay[h],4) for h in HORIZONS} }", flush=True)
    print(f"  rho_vs_usdcny_beta: spearman={rho_sp:.4f}(n={n_sp})", flush=True)
    results.append({"fid": fid, "ic": ic, "icir": icir, "hit": hit, "n": len(icm),
                    "cov": cov_ad, "cov8": cov_ge8, "to": to, "decay": decay,
                    "rho_sp": rho_sp, "gate": gate})

# market corr factor
t1 = time.time()
panel = mkt_corr_panel
cov_ad, cov_ge8 = coverage(panel)
to = turnover_rank(panel)
decay = {}
for h in HORIZONS:
    ic = ic_series(panel, fwd[h])
    decay[h] = float(ic.mean()) if len(ic) else np.nan
icm = ic_series(panel, fwd[10])
ic = float(icm.mean()) if len(icm) else np.nan
icir = float(icm.mean() / icm.std()) if len(icm) > 2 else np.nan
rho_sp, n_sp = spearman_pooled(panel, lib["usdcny_beta_60"])
gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
print(f"=== mkt_corr_60 === gate={'PASS' if gate else 'FAIL'} ({time.time()-t1:.1f}s)", flush=True)
print(f"  ic={ic:+.4f} icir={icir:+.4f} n_ic={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f}", flush=True)
print(f"  decay={ {str(h): round(decay[h],4) for h in HORIZONS} }", flush=True)
print(f"  rho_vs_usdcny_beta: spearman={rho_sp:.4f}(n={n_sp})", flush=True)
results.append({"fid": "mkt_corr_60", "ic": ic, "icir": icir, "n": len(icm),
                "cov": cov_ad, "cov8": cov_ge8, "to": to, "rho_sp": rho_sp, "gate": gate})

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
