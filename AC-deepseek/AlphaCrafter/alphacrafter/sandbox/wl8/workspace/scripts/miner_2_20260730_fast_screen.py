"""miner_2 2026-07-30 -- FAST screening for new factor families.
Vectorized Spearman IC: pre-rank panels row-wise (cross-section), then per-date
Pearson on ranks == Spearman. Forward returns precomputed once per horizon.
Admission horizon = 10. Gate: |IC|>=0.007, |ICIR|>=0.084.
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   load_library_panels, max_library_corr,
                                   IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
for anchor in ["SPX", "XAU", "BTC", "WTI", "NDX", "US10Y"]:
    macro[anchor] = close[anchor].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}", flush=True)

# ---- precompute forward returns and their cross-sectional ranks (per asset own calendar) ----
def fwd_ret_panel(horizon):
    out = {}
    for a in close.columns:
        c = close[a].dropna()
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)

HORIZONS = [1, 2, 3, 5, 10, 20]
fwd = {h: fwd_ret_panel(h) for h in HORIZONS}
fwd_rank = {h: fwd[h].rank(axis=1) for h in HORIZONS}
print("forward returns precomputed", flush=True)


def fast_validate(panel):
    """Return dict of metrics using precomputed forward rank panels."""
    pr = panel.rank(axis=1)
    out = {}
    for h in HORIZONS:
        fr = fwd_rank[h]
        ics = []
        for dt in panel.index:
            x = pr.loc[dt].values
            y = fr.loc[dt].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= MIN_ASSETS_PER_DATE:
                xv, yv = x[m], y[m]
                if xv.std() == 0 or yv.std() == 0:
                    continue
                ics.append(float(np.corrcoef(xv, yv)[0, 1]))
        out[h] = np.array(ics)
    ic10 = out[10]
    ic = float(ic10.mean()) if len(ic10) else np.nan
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic10 < 0).mean())
    n_total = float(panel.notna().sum().sum())
    denom = panel.shape[0] * panel.shape[1]
    cov_ad = n_total / denom
    cov8 = float((panel.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    ranks = panel.rank(axis=1)
    to = float(ranks.diff(10).abs().mean(axis=1).dropna().mean())
    return {
        "panel": panel,
        "ic": ic, "icir": icir, "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ic10)),
        "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov8, 4),
        "turnover_10d_rank": round(to, 4),
        "decay_ic_by_horizon": {str(h): round(float(out[h].mean()), 4) if len(out[h]) else np.nan for h in HORIZONS},
    }


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


# ---------- candidate factor functions (dense per-asset series) ----------
def f_intraday_pos_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    return _safe_div(c - l, rng).rolling(win).mean()

def f_body_ratio_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    return _safe_div((c - o).abs(), rng).rolling(win).mean()

def f_upper_shadow_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    return _safe_div(h - np.maximum(o, c), rng).rolling(win).mean()

def f_lower_shadow_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    return _safe_div(np.minimum(o, c) - l, rng).rolling(win).mean()

def f_overnight_gap_20(c, v, o, h, l, m, win=20):
    return (_safe_div(o, c.shift(1)) - 1.0).rolling(win).mean()

def f_overnight_gap_60(c, v, o, h, l, m, win=60):
    return (_safe_div(o, c.shift(1)) - 1.0).rolling(win).mean()

def f_overnight_gap_abs_20(c, v, o, h, l, m, win=20):
    return (_safe_div(o, c.shift(1)) - 1.0).abs().rolling(win).mean()

def f_mom_accel_10x20(c, v, o, h, l, m):
    return (c / c.shift(10) - 1.0) - (c / c.shift(20) - 1.0)

def f_mom_accel_20x60(c, v, o, h, l, m):
    return (c / c.shift(20) - 1.0) - (c / c.shift(60) - 1.0)

def f_mom_accel_10x60(c, v, o, h, l, m):
    return (c / c.shift(10) - 1.0) - (c / c.shift(60) - 1.0)

def f_dd_depth_60(c, v, o, h, l, m, win=60):
    return c / c.rolling(win).max() - 1.0

def f_dd_depth_120(c, v, o, h, l, m, win=120):
    return c / c.rolling(win).max() - 1.0

def f_dd_depth_252(c, v, o, h, l, m, win=252):
    return c / c.rolling(win).max() - 1.0

def f_dd_speed_60x120(c, v, o, h, l, m):
    return (c / c.rolling(60).max() - 1.0) - (c / c.rolling(120).max() - 1.0)

def _m(c, m, key):
    return m[key].reindex(c.index).ffill()

def _beta(asset_ret, anchor_ret, win):
    return asset_ret.rolling(win).cov(anchor_ret) / anchor_ret.rolling(win).var()

def make_anchor_cond(key, win=60, look=20):
    def fn(c, v, o, h, l, m, win=win, look=look):
        a = _m(c, m, key)
        b = _beta(c.pct_change(), a.pct_change(), win)
        return b * (a / a.shift(look) - 1.0)
    fn.__name__ = f"{key.lower()}_beta_cond_{win}x{look}"
    return fn

def f_mom_vol_adj_20(c, v, o, h, l, m, win=20):
    return (c / c.shift(win) - 1.0) / c.pct_change().rolling(win).std()

def f_mom_vol_adj_60(c, v, o, h, l, m, win=60):
    return (c / c.shift(win) - 1.0) / c.pct_change().rolling(win).std()

def f_range_ratio_10x60(c, v, o, h, l, m):
    rng = (h - l) / c
    return rng.rolling(10).mean() / rng.rolling(60).mean()

def f_vol_ratio_20x60(c, v, o, h, l, m):
    vv = v.replace(0, np.nan)
    return vv.rolling(20).mean() / vv.rolling(60).mean()

def f_vol_z_60(c, v, o, h, l, m, win=60):
    vv = np.log(v.replace(0, np.nan))
    return (vv - vv.rolling(win).mean()) / vv.rolling(win).std()

cands = [
    ("intraday_pos_20", f_intraday_pos_20, "20d mean close location in day range"),
    ("body_ratio_20", f_body_ratio_20, "20d mean |body|/range"),
    ("upper_shadow_20", f_upper_shadow_20, "20d mean upper shadow/range"),
    ("lower_shadow_20", f_lower_shadow_20, "20d mean lower shadow/range"),
    ("overnight_gap_20", f_overnight_gap_20, "20d mean overnight gap"),
    ("overnight_gap_60", f_overnight_gap_60, "60d mean overnight gap"),
    ("overnight_gap_abs_20", f_overnight_gap_abs_20, "20d mean |overnight gap|"),
    ("mom_accel_10x20", f_mom_accel_10x20, "10d ret - 20d ret"),
    ("mom_accel_20x60", f_mom_accel_20x60, "20d ret - 60d ret"),
    ("mom_accel_10x60", f_mom_accel_10x60, "10d ret - 60d ret"),
    ("dd_depth_60", f_dd_depth_60, "close/60d high - 1"),
    ("dd_depth_120", f_dd_depth_120, "close/120d high - 1"),
    ("dd_depth_252", f_dd_depth_252, "close/252d high - 1"),
    ("dd_speed_60x120", f_dd_speed_60x120, "60d dd depth - 120d dd depth"),
    ("spx_beta_cond_60x20", make_anchor_cond("SPX"), "60d beta to SPX x 20d SPX move"),
    ("xau_beta_cond_60x20", make_anchor_cond("XAU"), "60d beta to XAU x 20d XAU move"),
    ("btc_beta_cond_60x20", make_anchor_cond("BTC"), "60d beta to BTC x 20d BTC move"),
    ("wti_beta_cond_60x20", make_anchor_cond("WTI"), "60d beta to WTI x 20d WTI move"),
    ("mom_vol_adj_20", f_mom_vol_adj_20, "20d ret / 20d vol"),
    ("mom_vol_adj_60", f_mom_vol_adj_60, "60d ret / 60d vol"),
    ("range_ratio_10x60", f_range_ratio_10x60, "10d/60d daily range ratio"),
    ("vol_ratio_20x60", f_vol_ratio_20x60, "20d/60d volume ratio"),
    ("vol_z_60", f_vol_z_60, "log-vol z-score 60d"),
]

results = {}
t0 = time.time()
for i, (name, fn, desc) in enumerate(cands):
    t1 = time.time()
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    res = fast_validate(panel)
    res["max_abs_library_correlation"] = round(max_library_corr(panel, lib), 4)
    results[name] = res
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-HI-CORR" if ok else "fail")
    print(f"[{i+1}/{len(cands)}] {name:24s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} "
          f"hit={res['ic_hit_ratio']:.3f} cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} "
          f"decay10={res['decay_ic_by_horizon']['10']:+.4f} -> {flag}  ({time.time()-t1:.1f}s)", flush=True)

print(f"\n===== SUMMARY (gate |IC|>=%.4f |ICIR|>=%.4f libcorr<0.5) total %.1fs =====" % (IC_GATE, ICIR_GATE, time.time() - t0))
for name, res in sorted(results.items()):
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-HI-CORR" if ok else "fail")
    dec = res["decay_ic_by_horizon"]
    print(f"{name:26s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} "
          f"decay{{1,5,10,20}}={dec['1']:+.3f}/{dec['5']:+.3f}/{dec['10']:+.3f}/{dec['20']:+.3f} -> {flag}")
