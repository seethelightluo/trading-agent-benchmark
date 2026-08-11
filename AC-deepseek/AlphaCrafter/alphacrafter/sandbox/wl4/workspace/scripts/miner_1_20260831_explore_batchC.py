"""miner_1 2026-08-31: explore batch C - trend efficiency, lottery/tail, range, liquidity, vol-structure.

First re-validates the 4 active library factors (drift check), then evaluates
new candidates (h=10 admission, 15-asset tradable universe, min 8 valid per date).

Candidates (novel vs all prior batches):
  1. kaufman_eff_20   - Kaufman efficiency ratio |P_t-P_{t-20}| / sum(|ret|,20)   (trend efficiency)  [+1]
  2. skew_20d         - rolling skewness of 20d returns (lottery demand)          [-1]
  3. max_ret_20d      - max daily return over 20d (MAX lottery effect)            [-1]
  4. range_pos_20     - mean((close-low)/(high-low)) over 20d (range position)    [+1]
  5. amihud_20        - mean(|ret|/volume) over 20d (illiquidity proxy)           [-1]
  6. vol_ratio_5_60   - vol5/vol60 (vol term-structure proxy)                     [-1]
  7. trend_r2_60      - R^2 of 60d linear trend fit (trend consistency, vectorized) [+1]
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
print(f"panels: {len(panels)} | closes {closes.shape} | "
      f"dates {closes.index[0].date()}..{closes.index[-1].date()}", flush=True)


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
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    print(f"{name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) decay={ {k: round(v,3) for k,v in m['decay_ic_by_horizon'].items()} } "
          f"-> {'PASS' if gate else ''}", flush=True)
    return m


# ============ 0. Re-validation of 4 active library factors ============
print("\n=== 0. ACTIVE LIBRARY RE-VALIDATION (drift check) ===", flush=True)
mkt = rets.mean(axis=1)

def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)

eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)

vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
vpc = rets.rolling(20).corr(vol_panel)

active = {
    "vol_price_corr_20": (vpc, 1),
    "eurusd_beta_60d": (rolling_beta(rets, eur.pct_change(), 60), 1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, cn10.pct_change(), 60), 1),
    "dn_mkt_beta_60d": (rolling_beta(rets, mkt.where(mkt < 0).fillna(0.0), 60), 1),
}
for name, (panel, es) in active.items():
    evaluate(f"[active] {name}", panel, es)

# ============ 1. New candidates ============
print("\n=== 1. NEW CANDIDATES ===", flush=True)

def sma(panel, w):
    return panel.rolling(w).mean()

# 1. kaufman_eff_20
ke = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
evaluate("kaufman_eff_20", ke, 1)

# 2. skew_20d
sk = rets.rolling(20).skew()
evaluate("skew_20d", sk, -1)

# 3. max_ret_20d
mx = rets.rolling(20).max()
evaluate("max_ret_20d", mx, -1)

# 4. range_pos_20
hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
rp = ((closes - lo) / (hi - lo).replace(0, np.nan)).rolling(20).mean()
evaluate("range_pos_20", rp, 1)

# 5. amihud_20
amihud = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean()
evaluate("amihud_20", amihud, -1)

# 6. vol_ratio_5_60
vr = rets.rolling(5).std() / rets.rolling(60).std()
evaluate("vol_ratio_5_60", vr, -1)

# 7. trend_r2_60 - vectorized R^2 of log-price on time over 60d: R2 = corr(x,t)^2
def rolling_r2_vec(panel, win=60):
    n = panel.shape[0]
    t = np.arange(n, dtype=float)
    tm = pd.Series(t).rolling(win).mean().to_numpy()
    tv = pd.Series(t * t).rolling(win).mean().to_numpy() - tm ** 2
    lp = np.log(panel)
    xt = lp.mul(t, axis=0)
    xm = lp.rolling(win).mean()
    cov = xt.rolling(win).mean() - xm * tm
    xv = lp.rolling(win).var(ddof=0)
    r2 = (cov ** 2) / (xv * tv)
    return r2.where(xv > 1e-14)

tr2 = rolling_r2_vec(closes, 60)
evaluate("trend_r2_60", tr2, 1)

# 8. rev_5d_skip1
rev = -(closes.shift(1) / closes.shift(6) - 1.0)
evaluate("rev_5d_skip1", rev, 1)

# 9. close_sma200
c200 = closes / sma(closes, 200) - 1.0
evaluate("close_sma200", c200, 1)

# 10. corr_mkt_60
cm = rets.rolling(60).corr(mkt)
evaluate("corr_mkt_60", cm, 1)

# 11. drawup_60
du = closes / closes.rolling(60).min() - 1.0
evaluate("drawup_60", du, 1)

# 12. vol_trend_60
vt = rets.rolling(20).std().rolling(10).mean() / rets.rolling(20).std().rolling(60).mean() - 1.0
evaluate("vol_trend_60", vt, -1)

print("\nDone.", flush=True)
