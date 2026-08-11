"""miner_1 2027-02-01: batch H exploration, FULLY VECTORIZED (no per-asset loops).

Admission gate (h=10): |IC| >= 0.0070 AND |ICIR| >= 0.0840, n_ic_dates >= 200,
max_abs_library_correlation < 0.5.
Cross-section: 15 tradable instruments; date valid if >= 8 valid values.
Data through previous completed trading day (2027-01-29), no lookahead.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, coverage_metrics, turnover_rank,
                                 max_library_corr)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = ret_panel(panels)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
MIN_IC_DATES = 200
print(f"panels={len(panels)} closes={closes.shape} "
      f"{closes.index[0].date()}..{closes.index[-1].date()} "
      f"load={time.time()-t0:.1f}s", flush=True)


def rank_ic_series_vec(factor_panel, fwd, min_valid=MIN_VALID):
    fr = factor_panel.rank(axis=1, method="average", na_option="keep")
    rr = fwd.rank(axis=1, method="average", na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1).astype(float)
    f = fr.where(valid)
    r = rr.where(valid)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = fm.mul(rm).sum(axis=1, min_count=1)
    den = np.sqrt(fm.pow(2).sum(axis=1, min_count=1) * rm.pow(2).sum(axis=1, min_count=1))
    ic = num.div(den.replace(0, np.nan))
    ic = ic.where((n >= min_valid) & (n >= 2) & (den > 1e-14))
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def summarize(ic_series):
    ic = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    return {"ic": round(ic, 4), "icir": round(ic / std, 4) if std > 0 else 0.0,
            "ic_hit_ratio": round(float((np.sign(ic_series) != 0).mean()), 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(std, 4)}


def rolling_beta_vec(asset_ret, driver_ret, win=60, min_obs=40):
    """beta_i = cov(r_i, m)/var(m) fully vectorized across assets."""
    m = driver_ret
    m_mean = m.rolling(win, min_periods=min_obs).mean()
    a_mean = asset_ret.rolling(win, min_periods=min_obs).mean()
    cov_am = asset_ret.mul(m, axis=0).rolling(win, min_periods=min_obs).mean() - a_mean * m_mean
    var_m = m.rolling(win, min_periods=min_obs).var()
    return cov_am.div(var_m.replace(0, np.nan))


t1 = time.time()
# ---------------- library signals (4 effective factors) ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
dxy = panels["DXY"]["close"].astype(float)
jpy = panels["USDJPY"]["close"].astype(float)
vol = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)

lib = {
    "vol_price_corr_20": rets.rolling(20, min_periods=10).corr(vol),
    "eurusd_beta_60d": rolling_beta_vec(rets, eur.pct_change(), 60),
    "rate_beta_cn10y_60d": rolling_beta_vec(rets, cn10.pct_change(), 60),
    "dn_mkt_beta_60d": rolling_beta_vec(rets, dn, 60),
}
lib = {k: v.reindex(closes.index) for k, v in lib.items()}
print(f"library signals built {time.time()-t1:.1f}s", flush=True)

# ---------------- candidate panels (all vectorized, no per-asset loops) ----------------
t2 = time.time()
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)

def er_panel(win):
    num = (closes - closes.shift(win)).abs()
    den = rets.abs().rolling(win, min_periods=win // 2).sum()
    return num.div(den.replace(0, np.nan))

def autocorr_panel(win, minp):
    r1 = rets.shift(1)
    return rets.rolling(win, min_periods=minp).corr(r1)

candidates = {
    "er_20": er_panel(20),
    "er_10": er_panel(10),
    "autocorr_1_60": autocorr_panel(60, 40),
    "autocorr_1_20": autocorr_panel(20, 12),
    "skew_60": rets.rolling(60, min_periods=40).skew(),
    "down_ratio_60": rets.where(rets < 0, 0.0).rolling(60, min_periods=40).std()
                     .div(rets.rolling(60, min_periods=40).std().replace(0, np.nan)),
    "upday_ratio_60": (rets > 0).astype(float).rolling(60, min_periods=40).mean(),
    "amplitude_20": (hi - lo).div(closes).rolling(20, min_periods=10).mean(),
    "dxy_beta_60": rolling_beta_vec(rets, dxy.pct_change(), 60),
    "usdjpy_beta_60": rolling_beta_vec(rets, jpy.pct_change(), 60),
    "range_pos_20": (closes - lo.rolling(20, min_periods=10).min()) /
                    (hi.rolling(20, min_periods=10).max() - lo.rolling(20, min_periods=10).min()),
    "rel_mom_20": (closes / closes.shift(20) - 1.0).sub(
        (closes / closes.shift(20) - 1.0).rolling(252, min_periods=60).mean()),
    "sharpe_60": rets.rolling(60, min_periods=40).mean()
                 .div(rets.rolling(60, min_periods=40).std().replace(0, np.nan)),
    "max_dd_60": (closes / closes.rolling(60, min_periods=40).max() - 1.0)
                 .rolling(60, min_periods=40).min(),
}
candidates = {k: v.reindex(closes.index) for k, v in candidates.items()}
print(f"candidate panels built {time.time()-t2:.1f}s ({len(candidates)} candidates)", flush=True)

# ---------------- evaluate ----------------
t3 = time.time()
fwds = {h: forward_returns(closes, h) for h in HORIZONS}
results = {}
for name, panel in candidates.items():
    ics = rank_ic_series_vec(panel, fwds[H_ADM])
    m = summarize(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    decay = {}
    for h in HORIZONS:
        ic_h = rank_ic_series_vec(panel, fwds[h])
        decay[str(h)] = round(float(ic_h.mean()), 4) if len(ic_h) else None
    m["decay_ic_by_horizon"] = decay
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    m["expected_sign"] = 1 if m["ic"] >= 0 else -1
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES and corr < 0.5)
    print(f"{name:18s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m.get('turnover_10d_rank')} rho={corr:.3f}({key}) "
          f"decay10={m['decay_ic_by_horizon'].get('10')} {'-> PASS' if gate else ''}", flush=True)
    results[name] = m

with open("scripts/_miner1_20270201_batchH.json", "w") as fh:
    json.dump({"window": {"first": str(closes.index[0].date()), "last": str(closes.index[-1].date()),
                          "n_dates": int(len(closes)), "n_assets": int(closes.shape[1])},
               "results": results}, fh, indent=1, default=str)
print(f"saved scripts/_miner1_20270201_batchH.json elapsed={time.time()-t0:.1f}s")
