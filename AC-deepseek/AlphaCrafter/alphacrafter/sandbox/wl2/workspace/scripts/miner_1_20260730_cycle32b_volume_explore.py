"""miner_1 cycle 32b: explore VOLUME / OHLC-structure factors orthogonal to the FX-beta family.

Cycle32 showed the driver-beta family is saturated (only EURUSD passes thresholds but
maxlib 0.839 vs DXY). Volume columns ARE populated (SPX ~1.8e9 shares/day, 000300.SH
~1.8e10) and NO volume-based factor is in the active library -> strongest orthogonal
opportunity. Also test range-position / drawdown / overnight-gap / tail structures.

Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5, turnover vs cadence.
Validation date 2026-07-30 (data visible through 2026-07-29).
"""
import sys, json
from itertools import groupby
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_asset, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor)

close_panel = load_panel()
union_idx = close_panel.index

frames = {a: load_asset(a) for a in TRADABLES}

def own_series(a, col):
    s = frames[a][col].astype(float)
    s.index = pd.to_datetime(frames[a]["date"].values)
    return s

vol = {a: own_series(a, "volume") for a in TRADABLES}
opn = {a: own_series(a, "open") for a in TRADABLES}
hi = {a: own_series(a, "high") for a in TRADABLES}
lo = {a: own_series(a, "low") for a in TRADABLES}

def reindex_to_panel(series_dict):
    out = {}
    for a, s in series_dict.items():
        s.index = pd.to_datetime(s.index)
        out[a] = s.reindex(union_idx)
    return pd.DataFrame(out, index=union_idx)

cands = {}

# 1. volume trend: 20d avg volume / 60d avg volume - 1 (per-asset normalized)
vt = {}
for a in TRADABLES:
    v = vol[a]
    vt[a] = (v.rolling(20, min_periods=10).mean() / v.rolling(60, min_periods=30).mean() - 1.0)
cands["vol_trend_20x60"] = reindex_to_panel(vt)

# 2. volume level z-score: (v20 - mean(v60))/std(v60)
vz = {}
for a in TRADABLES:
    v = vol[a]
    vz[a] = ((v.rolling(20, min_periods=10).mean() - v.rolling(60, min_periods=30).mean())
             / (v.rolling(60, min_periods=30).std() + 1e-9))
cands["vol_z_20x60"] = reindex_to_panel(vz)

# 3. Amihud illiquidity (log): mean(|ret|)/mean(volume) over 60d
am = {}
for a in TRADABLES:
    c = close_panel[a]
    v = vol[a].reindex(c.index)
    illiq = (c.pct_change().abs() / v).rolling(60, min_periods=30).mean()
    am[a] = np.log(illiq + 1e-12)
cands["amihud_60"] = reindex_to_panel(am)

# 4. up/down volume balance over 20d
ud = {}
for a in TRADABLES:
    c = close_panel[a]
    v = vol[a].reindex(c.index)
    r = c.pct_change()
    up = v.where(r > 0)
    dn = v.where(r < 0)
    s20 = v.rolling(20, min_periods=10).sum()
    ud[a] = (up.rolling(20, min_periods=10).sum() - dn.rolling(20, min_periods=10).sum()) / (s20 + 1e-9)
cands["updown_vol_20"] = reindex_to_panel(ud)

# 5. volume-confirmed momentum: mom20_skip5 * (1 + 2*tanh(vol_trend))
vcm = {}
for a in TRADABLES:
    c = close_panel[a]
    mom = c.shift(5) / c.shift(25) - 1.0
    t = vt[a].reindex(c.index)
    vcm[a] = mom * (1.0 + 2.0 * np.tanh(t.clip(-2, 2)))
cands["vol_confirmed_mom20"] = reindex_to_panel(vcm)

# 6. daily range ratio: mean((high-low)/close, 20d)
rr = {}
for a in TRADABLES:
    c = close_panel[a]
    rng = ((hi[a].reindex(c.index) - lo[a].reindex(c.index)) / c)
    rr[a] = rng.rolling(20, min_periods=10).mean()
cands["range_ratio_20"] = reindex_to_panel(rr)

# 7. close position within 20d range
cp = {}
for a in TRADABLES:
    c = close_panel[a]
    h20 = hi[a].reindex(c.index).rolling(20, min_periods=10).max()
    l20 = lo[a].reindex(c.index).rolling(20, min_periods=10).min()
    cp[a] = (c - l20) / (h20 - l20 + 1e-9)
cands["close_pos_20"] = reindex_to_panel(cp)

# 8. drawdown depth from 60d high (negative = deeper drawdown)
dd = {}
for a in TRADABLES:
    c = close_panel[a]
    dd[a] = -(1.0 - c / c.rolling(60, min_periods=30).max())
cands["dd_60"] = reindex_to_panel(dd)

# 9. overnight gap consistency (mean open/prev_close - 1 over 20d)
og = {}
for a in TRADABLES:
    c = close_panel[a]
    o = opn[a].reindex(c.index)
    gap = o / c.shift(1) - 1.0
    og[a] = gap.rolling(20, min_periods=10).mean()
cands["gap_consistency_20"] = reindex_to_panel(og)

# 10. tail ratio: 95th pctile |ret| over 60d / 60d std
tr = {}
for a in TRADABLES:
    c = close_panel[a]
    r = c.pct_change().abs()
    q95 = r.rolling(60, min_periods=30).quantile(0.95)
    sd = r.rolling(60, min_periods=30).std()
    tr[a] = q95 / (sd + 1e-9)
cands["tail_ratio_60"] = reindex_to_panel(tr)

# ---------------------------------------------------------------------------
# Library (broad: 5 ensemble + recent references) for correlation gate
# ---------------------------------------------------------------------------
def beta_cond(asset_close, driver_close, w=60, m=20, minp_frac=0.5):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar = asset_close.pct_change()
    dr = dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    minp = max(int(w * minp_frac), 15)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["d"])
    var = df["d"].rolling(w, min_periods=minp).var()
    beta = cov / var
    mom = dcs / dcs.shift(m) - 1.0
    return beta * mom.reindex(beta.index)

def mom20_volproxy60(s):
    mom = s.shift(5) / s.shift(25) - 1.0
    proxy = s.shift(5) / s.shift(65) - 1.0
    return mom / (1.0 + proxy.abs())

def calmness_20(s):
    r = s.pct_change()
    return r.abs().rolling(20, min_periods=10).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)

def gain_loss_20(s):
    r = s.pct_change()
    g = r.clip(lower=0).rolling(20, min_periods=10).sum()
    l = (-r.clip(upper=0)).rolling(20, min_periods=10).sum()
    return (g - l) / (g + l + 1e-9)

def intraday_drift_20(close_s):
    o = opn[close_s.name].reindex(close_s.index)
    return (close_s / o - 1.0).rolling(20, min_periods=10).mean()

def downbeta_spx_60(s):
    spx = macro_series("SPX").reindex(s.index).ffill()
    ar, sr = s.pct_change(), spx.pct_change()
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    neg = df[df["s"] < 0]
    b = neg["a"].rolling(60, min_periods=15).cov(neg["s"]) / neg["s"].rolling(60, min_periods=15).var()
    return b.reindex(df.index)

def max_consec_gain_20(s):
    r = (s.pct_change() > 0)
    def _maxrun(x):
        if len(x) < 10:
            return np.nan
        mx = 0
        cur = 0
        for v in x:
            cur = cur + 1 if v else 0
            mx = max(mx, cur)
        return float(mx)
    return r.rolling(20, min_periods=10).apply(_maxrun, raw=True)

lib = {}
lib["mom20_volproxy60"] = per_asset(close_panel, mom20_volproxy60)
lib["dxy_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("DXY"), 60, 20)
lib["calmness_20"] = per_asset(close_panel, calmness_20)
lib["usdjpy_beta_cond_120x60"] = per_asset(close_panel, beta_cond, macro_series("USDJPY"), 120, 60)
lib["downbeta_spx_60"] = per_asset(close_panel, downbeta_spx_60)
lib["gain_loss_20"] = per_asset(close_panel, gain_loss_20)
lib["intraday_drift_20"] = per_asset(close_panel, intraday_drift_20)
lib["vol_of_vol20x60"] = per_asset(close_panel, lambda s: s.pct_change().rolling(20).std().rolling(60).std())
lib["mom_20d_skip5"] = per_asset(close_panel, lambda s: s.shift(5) / s.shift(25) - 1.0)
lib["max_consec_gain_20"] = per_asset(close_panel, max_consec_gain_20)

fwd = {str(h): forward_returns(close_panel, h) for h in (1, 2, 3, 5, 10, 20)}

results = {}
for name, sig in cands.items():
    m = validate_factor(sig, close_panel, library=lib, fwd_cache=fwd)
    ic_ser = compute_ic(sig, fwd["10"]).dropna()
    reg = {}
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29"),
                   ("2026-03-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 20:
            sd = sub.std()
            reg[f"{r0[:4]}-{r1[:4]}"] = {"ic": round(sub.mean(), 4),
                                          "icir": round(sub.mean() / sd, 3) if sd > 0 else 0.0,
                                          "n": int(len(sub))}
    results[name] = {"metrics": m, "regime": reg}
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and m["max_abs_library_correlation"] < 0.5
    print(f"[{name:26s}] IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} to={m['turnover_10_rank']} "
          f"maxlib={m['max_abs_library_correlation']:.4f} => {'PASS' if passed else 'fail'}")
    if passed:
        reg_s = " | ".join(f"{k}:{v['ic']:+.4f}/{v['icir']:+.3f}(n{v['n']})" for k, v in reg.items())
        print(f"      regime: {reg_s}")

json.dump(results, open("scripts/_miner1_cycle32b_volume_results.json", "w"), indent=1, default=str)
print("\nDONE cycle32b explore")
