"""miner_1 cycle-14 batch-H (2026-07-30).
Novel factor families NOT previously tried (checked evicted/rejected library and
miner_1/2/3 screen logs):
  - lev_effect_60    : 60d corr(r_{t-1}, r_t^2)  -> leverage/asymmetry effect
  - vol_cluster_60   : 60d autocorr of squared returns (vol-shock persistence)
  - rng_cluster_60   : 60d autocorr of daily range (high-low)/close
  - btc_lag_lead_60  : 60d corr(asset ret, lag-1 BTC ret)  (crypto lead-lag)
  - cn_lag_lead_60   : 60d corr(asset ret, lag-1 CSI300 ret) (China lead-lag)
  - tail_ratio_20    : max daily gain / |min daily loss| over 20d (robust skew)
  - up_down_vol_60   : std(pos ret days)/std(neg ret days) 60d (vol asymmetry)
  - vol_slope_60     : 60d OLS slope of log volume, normalized (liquidity trend)
  - vol_mom_10x5     : log-volume momentum skip-5 analog of mom_10d_skip5
  - market_corr_60   : 60d corr with equal-weight cross-asset market return
  - sk_ratio_20      : (p90-p50)/(p50-p10) quantile skewness of returns, 20d

All factors are per-asset (no macro data -> no calendar misalignment NaN risk).
Admission gates at h=10: |IC|>=0.0070, |ICIR|>=0.0840; orthogonality rho<0.5 vs
the CURRENT effective library (mom_10d_skip5, vix_beta_cond_60x20,
yield_beta_cond_60x20), computed from real signal artifacts (Spearman).
"""
import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, ic_series, fwd_returns, coverage,
    turnover_rank, IC_GATE, ICIR_GATE, artifact_b64,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}", flush=True)

# ---- current effective library (from real signal artifacts) ----
LIB_FIDS = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]
lib = {}
for fid in LIB_FIDS:
    try:
        d = json.load(open(f"factors/{fid}.json"))
        data = d["validation"]["signal_artifact"]["data"]
        import base64, zlib, io
        raw = base64.b64decode(data)
        csv_text = zlib.decompress(raw).decode()
        p = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
        p.index = pd.DatetimeIndex(p.index)
        lib[fid] = p
        print(f"[lib] {fid}: shape={p.shape} status={d['validation']['status']}", flush=True)
    except Exception as e:
        print(f"[warn] cannot load {fid}: {e}")


def compute_panel(fn, **params):
    """Per-asset factor on dense calendar, reindexed onto union panel."""
    out = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        c = close[a].reindex(idx)
        v = None if vol is None else vol[a].reindex(idx)
        o = None if open_ is None else open_[a].reindex(idx)
        h = None if high is None else high[a].reindex(idx)
        l = None if low is None else low[a].reindex(idx)
        try:
            s = fn(c, v, o, h, l, **params)
            out[a] = pd.Series(np.asarray(s), index=idx).reindex(close.index)
        except Exception as e:
            print(f"  [err] {a}: {e}")
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)


def lev_effect_60(c, v, o, h, l, win=60):
    r = c.pct_change()
    r2 = r ** 2
    return r.shift(1).rolling(win).corr(r2)


def vol_cluster_60(c, v, o, h, l, win=60):
    r2 = c.pct_change() ** 2
    return r2.rolling(win).corr(r2.shift(1))


def rng_cluster_60(c, v, o, h, l, win=60):
    rng = (h - l) / c
    return rng.rolling(win).corr(rng.shift(1))


def _lag_lead(c, ref_ret, win=60):
    r = c.pct_change()
    return r.rolling(win).corr(ref_ret.shift(1))


def btc_lag_lead_60(c, v, o, h, l, win=60):
    btc = close["BTC"].reindex(c.index).pct_change()
    return _lag_lead(c, btc, win)


def cn_lag_lead_60(c, v, o, h, l, win=60):
    cn = close["000300.SH"].reindex(c.index).pct_change()
    return _lag_lead(c, cn, win)


def tail_ratio_20(c, v, o, h, l, win=20):
    r = c.pct_change()
    return r.rolling(win).max() / r.rolling(win).min().abs()


def up_down_vol_60(c, v, o, h, l, win=60):
    r = c.pct_change()
    up = r.where(r > 0, np.nan)
    dn = r.where(r < 0, np.nan)
    up_std = up.rolling(win).std()
    dn_std = dn.rolling(win).std()
    return (up_std / dn_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def vol_slope_60(c, v, o, h, l, win=60):
    lv = np.log(v.replace(0, np.nan))
    x = np.arange(win)
    def slope(y):
        m = np.isfinite(y)
        if m.sum() < win * 0.5:
            return np.nan
        return np.polyfit(x[m], y[m], 1)[0]
    return lv.rolling(win).apply(slope, raw=True)


def vol_mom_10x5(c, v, o, h, l, look=10, skip=5):
    lv = np.log(v.replace(0, np.nan))
    return lv.shift(skip) - lv.shift(skip + look)


def market_corr_60(c, v, o, h, l, win=60):
    mkt = pd.DataFrame({a: close[a].pct_change() for a in ASSETS}).mean(axis=1)
    mkt = mkt.reindex(c.index)
    r = c.pct_change()
    return r.rolling(win).corr(mkt)


def sk_ratio_20(c, v, o, h, l, win=20):
    r = c.pct_change()
    p90 = r.rolling(win).quantile(0.90)
    p50 = r.rolling(win).quantile(0.50)
    p10 = r.rolling(win).quantile(0.10)
    return (p90 - p50) / (p50 - p10).replace(0, np.nan)


CANDIDATES = [
    ("lev_effect_60", lev_effect_60, {}),
    ("vol_cluster_60", vol_cluster_60, {}),
    ("rng_cluster_60", rng_cluster_60, {}),
    ("btc_lag_lead_60", btc_lag_lead_60, {}),
    ("cn_lag_lead_60", cn_lag_lead_60, {}),
    ("tail_ratio_20", tail_ratio_20, {}),
    ("up_down_vol_60", up_down_vol_60, {}),
    ("vol_slope_60", vol_slope_60, {}),
    ("vol_mom_10x5", vol_mom_10x5, {}),
    ("market_corr_60", market_corr_60, {}),
    ("sk_ratio_20", sk_ratio_20, {}),
]

HORIZONS = (1, 2, 3, 5, 10, 20)
results = {}
for name, fn, params in CANDIDATES:
    panel = compute_panel(fn, **params)
    cov_ad, cov_ge8 = coverage(panel)
    decay, ic_series_by_h = {}, {}
    for h in HORIZONS:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        ic_series_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic_main = ic_series_by_h[10]
    ic = float(ic_main.mean())
    icir = float(ic_main.mean() / ic_main.std()) if len(ic_main) > 2 else np.nan
    hit = float((ic_main > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic_main < 0).mean())

    # Spearman rho vs each library panel (same gate as post-Miner audit)
    lib_corr = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            lib_corr[fid] = np.nan
            continue
        a = panel.loc[common, cols]
        b = lp.loc[common, cols]
        # date-wise mean rank panel correlation (matches audit gate approach)
        ra = a.rank(axis=1)
        rb = b.rank(axis=1)
        rr = []
        for dt in common:
            x, y = ra.loc[dt], rb.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 5:
                rr.append(x[m].corr(y[m], method="pearson"))
        lib_corr[fid] = float(np.nanmean(rr)) if rr else np.nan
    max_abs = max([abs(v) for v in lib_corr.values() if np.isfinite(v)], default=0.0)

    res = {
        "ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 4),
        "n_ic_dates": int(len(ic_main)), "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov_ge8, 4), "turnover_10d_rank": round(turnover_rank(panel), 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in HORIZONS},
        "lib_corr": {k: round(v, 4) if np.isfinite(v) else None for k, v in lib_corr.items()},
        "max_abs_library_correlation": round(max_abs, 4),
    }
    results[name] = res
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"\n=== {name} ===  GATE: {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.0f}s)", flush=True)
    for k in ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "coverage_asset_days",
              "coverage_dates_ge8", "turnover_10d_rank"]:
        print(f"  {k}: {res[k]}")
    print(f"  decay: {res['decay_ic_by_horizon']}")
    print(f"  lib_corr: {res['lib_corr']}  maxAbs={res['max_abs_library_correlation']}")

json.dump(results, open("scripts/_miner1_cycle14_batchH_results.json", "w"), indent=1)
print(f"\nDONE in {time.time()-t0:.0f}s; results -> scripts/_miner1_cycle14_batchH_results.json")
