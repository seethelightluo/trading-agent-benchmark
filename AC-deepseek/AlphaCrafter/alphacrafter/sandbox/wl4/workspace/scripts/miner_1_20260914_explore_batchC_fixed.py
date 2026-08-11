"""miner_1 2026-09-14: batch C (fixed) - trend efficiency, lottery/tail, range, liquidity, vol-structure.

FIX vs 20260831 run:
  * All rolling-window factors computed PER-ASSET on each asset's own continuous
    calendar (dropna), then reindexed to the union calendar. Union-calendar
    rolling polluted equity rows with weekend NaNs (only ~15 IC dates).
  * vol_price_corr_20 re-validation uses min_periods=10 per its persisted params.
  * trend_r2_60 vectorized per-asset (no polyfit, no ndarray/DataFrame alignment).
  * Admission gate additionally requires n_ic_dates >= 200 to avoid tiny-sample passes.

Candidates (h=10 admission, 15-asset tradable universe, min 8 valid per date):
  1. kaufman_eff_20   - Kaufman efficiency ratio |P_t-P_{t-20}| / sum(|ret|,20)   [+1]
  2. skew_20d         - rolling skewness of 20d returns (lottery demand)          [-1]
  3. max_ret_20d      - max daily return over 20d (MAX lottery effect)            [-1]
  4. range_pos_20     - mean((close-low)/(high-low)) over 20d (range position)    [+1]
  5. amihud_20        - mean(|ret|/volume) over 20d (illiquidity proxy)           [-1]
  6. vol_ratio_5_60   - vol5/vol60 (vol term-structure proxy)                     [-1]
  7. trend_r2_60      - R^2 of 60d linear trend fit (trend consistency)           [+1]
  8. rev_5d_skip1     - -(ret over days t-6..t-1) short-term reversal              [+1]
  9. close_sma200     - close/SMA200-1 (long-term trend)                          [+1]
 10. corr_mkt_60      - 60d rolling correlation with equal-weight market          [+1]
 11. drawup_60        - close/rolling_min(close,60)-1 (distance from 60d low)     [+1]
 12. vol_trend_60     - SMA(vol20,10)/SMA(vol20,60)-1 (rising-vol regime)         [-1]

No lookahead: factor at t uses data <= t; forward returns close[t+h]/close[t]-1.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
lib = library_signals(panels, closes, rets)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
MIN_IC_DATES = 200
print(f"panels: {len(panels)} | closes {closes.shape} | "
      f"dates {closes.index[0].date()}..{closes.index[-1].date()}", flush=True)


# ---------- per-asset continuous-calendar factor framework ----------
def per_asset(build):
    """build(s, a) -> per-asset factor Series indexed like s (asset's own calendar)."""
    cols = {}
    for a in closes.columns:
        s = closes[a].dropna()
        if len(s) < 260:  # need >= ~1y for 200d windows + burn-in
            continue
        f = build(s, a)
        if f is not None and len(f):
            cols[a] = pd.Series(f, index=s.index).reindex(s.index)
    return pd.DataFrame(cols).reindex(closes.index)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


def evaluate(name, panel, exp_sign):
    panel = panel.reindex(closes.index)
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, MIN_VALID, exp_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES)
    print(f"{name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) decay={ {k: round(v,3) for k,v in m['decay_ic_by_horizon'].items()} } "
          f"-> {'PASS' if gate else ''}", flush=True)
    return m


def sma(panel, w):
    return panel.rolling(w).mean()


# ============ 0. ACTIVE LIBRARY RE-VALIDATION (drift check) ============
print("\n=== 0. ACTIVE LIBRARY RE-VALIDATION (drift check) ===", flush=True)
mkt = rets.mean(axis=1)
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)

# vol_price_corr_20: rolling corr(ret, volume, 20, min_obs=10) per asset
def vpc_build(s, a):
    r = s.pct_change()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    z = pd.concat([r.rename("a"), v.rename("v")], axis=1).dropna()
    return z["a"].rolling(20, min_periods=10).corr(z["v"])

active = {
    "vol_price_corr_20": (per_asset(vpc_build), 1),
    "eurusd_beta_60d": (rolling_beta(rets, eur.pct_change(), 60), -1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, cn10.pct_change(), 60), -1),
    "dn_mkt_beta_60d": (rolling_beta(rets, mkt.where(mkt < 0).fillna(0.0), 60), 1),
}
for name, (panel, es) in active.items():
    evaluate(f"[active] {name}", panel, es)

# ============ 1. NEW CANDIDATES ============
print("\n=== 1. NEW CANDIDATES ===", flush=True)

# 1. kaufman_eff_20
def f_ke(s, a):
    r = s.pct_change()
    return (s - s.shift(20)).abs() / r.abs().rolling(20).sum()
evaluate("kaufman_eff_20", per_asset(f_ke), 1)

# 2. skew_20d
def f_sk(s, a):
    return s.pct_change().rolling(20).skew()
evaluate("skew_20d", per_asset(f_sk), -1)

# 3. max_ret_20d
def f_mx(s, a):
    return s.pct_change().rolling(20).max()
evaluate("max_ret_20d", per_asset(f_mx), -1)

# 4. range_pos_20
def f_rp(s, a):
    hi = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    rng = (s - lo) / (hi - lo).replace(0, np.nan)
    return rng.rolling(20).mean()
evaluate("range_pos_20", per_asset(f_rp), 1)

# 5. amihud_20
def f_am(s, a):
    r = s.pct_change().abs()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    return (r / v.replace(0, np.nan)).rolling(20).mean()
evaluate("amihud_20", per_asset(f_am), -1)

# 6. vol_ratio_5_60
def f_vr(s, a):
    r = s.pct_change()
    return r.rolling(5).std() / r.rolling(60).std()
evaluate("vol_ratio_5_60", per_asset(f_vr), -1)

# 7. trend_r2_60 - vectorized R^2 of log-price on time over 60d per asset
def f_tr2(s, a):
    lp = np.log(s)
    n = len(lp)
    t = pd.Series(np.arange(n, dtype=float), index=lp.index)
    tm = t.rolling(60).mean()
    tv = (t * t).rolling(60).mean() - tm ** 2
    xt = lp * t
    xm = lp.rolling(60).mean()
    xv = lp.rolling(60).var(ddof=0)
    cov = xt.rolling(60).mean() - xm * tm
    r2 = (cov ** 2) / (xv * tv)
    return r2.where(xv > 1e-14)
evaluate("trend_r2_60", per_asset(f_tr2), 1)

# 8. rev_5d_skip1
def f_rev(s, a):
    return -(s.shift(1) / s.shift(6) - 1.0)
evaluate("rev_5d_skip1", per_asset(f_rev), 1)

# 9. close_sma200
def f_c200(s, a):
    return s / s.rolling(200).mean() - 1.0
evaluate("close_sma200", per_asset(f_c200), 1)

# 10. corr_mkt_60
def f_cm(s, a):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), mkt.reindex(r.index).rename("m")], axis=1).dropna()
    return z["a"].rolling(60).corr(z["m"])
evaluate("corr_mkt_60", per_asset(f_cm), 1)

# 11. drawup_60
def f_du(s, a):
    return s / s.rolling(60).min() - 1.0
evaluate("drawup_60", per_asset(f_du), 1)

# 12. vol_trend_60
def f_vt(s, a):
    v20 = s.pct_change().rolling(20).std()
    return v20.rolling(10).mean() / v20.rolling(60).mean() - 1.0
evaluate("vol_trend_60", per_asset(f_vt), -1)

print("\nDone.", flush=True)
