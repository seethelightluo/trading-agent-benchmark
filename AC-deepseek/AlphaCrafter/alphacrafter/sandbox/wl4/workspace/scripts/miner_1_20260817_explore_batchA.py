"""miner_1 2026-08-17: explore macro-sensitivity beta & price-shape factor batch.

Candidates (h=10 admission, 15-asset tradable universe, min 8 valid per date):
  Macro-sensitivity betas (complementary to existing eurusd/cn10y/mkt beta):
    1. dxy_beta_60d        - 60d beta of asset ret on DXY ret (USD sensitivity)
    2. vix_beta_60d        - 60d beta of asset ret on VIX ret (risk sensitivity, unconditional)
    3. us10y_beta_60d      - 60d beta of asset ret on US10Y yield change
    4. copper_beta_60d     - 60d beta on COPPER returns (commodity-cycle sensitivity)
    5. xau_beta_60d        - 60d beta on XAU returns (safe-haven sensitivity)
  Price-shape / time-series:
    6. bollinger_pos_20    - (close - SMA20) / (2*std20), mean-reversion position
    7. drawdown_60d        - close / rolling_max(close,60) - 1 (distance from high)
    8. ma_cross_20_60      - SMA20/SMA60 - 1 (trend slope)
    9. ret_autocorr_5d     - rolling 5d autocorrelation of daily returns (persistence)
   10. vol_percentile_120  - percentile rank of 20d vol within trailing 120d history
   11. downside_capture_60 - mean(asset ret | mkt<0) / mean(mkt ret | mkt<0) (downside capture)
   12. upside_capture_60   - mean(asset ret | mkt>0) / mean(mkt ret | mkt>0) (upside capture)

No lookahead: factor at t uses data <= t; forward returns close[t+h]/close[t]-1.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr)

panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
lib = library_signals(panels, closes, rets)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
print(f"panels: {len(panels)} | closes {closes.shape} | "
      f"dates {closes.index[0].date()}..{closes.index[-1].date()}", flush=True)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


def capture_ratio(asset_ret, driver_ret, win=60, direction="down"):
    """Mean(asset ret | driver cond) / mean(driver ret | driver cond) over win."""
    if direction == "down":
        mask = (driver_ret < 0)
    else:
        mask = (driver_ret > 0)
    num = asset_ret.where(mask).rolling(win).mean()
    den = driver_ret.where(mask).rolling(win).mean()
    return num.div(den.replace(0, np.nan))


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
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) decay={m['decay_ic_by_horizon']} "
          f"-> {'PASS' if gate else ''}", flush=True)
    return m


def sma(panel, w):
    return panel.rolling(w).mean()


# ---- market (equal-weight cross-asset) for conditional factors ----
mkt = rets.mean(axis=1)

# ---- 1. dxy_beta_60d ----
dxy = panels["DXY"]["close"].astype(float)
evaluate("dxy_beta_60d", rolling_beta(rets, dxy.pct_change(), 60), 1)

# ---- 2. vix_beta_60d (unconditional) ----
vix = panels["VIX"]["close"].astype(float)
evaluate("vix_beta_60d", rolling_beta(rets, vix.pct_change(), 60), -1)

# ---- 3. us10y_beta_60d ----
us10 = panels["US10Y"]["close"].astype(float)
evaluate("us10y_beta_60d", rolling_beta(rets, us10.pct_change(), 60), -1)

# ---- 4. copper_beta_60d ----
cop = panels["COPPER"]["close"].astype(float)
evaluate("copper_beta_60d", rolling_beta(rets, cop.pct_change(), 60), 1)

# ---- 5. xau_beta_60d ----
xau = panels["XAU"]["close"].astype(float)
evaluate("xau_beta_60d", rolling_beta(rets, xau.pct_change(), 60), 1)

# ---- 6. bollinger_pos_20 ----
bp = (closes - sma(closes, 20)) / (2 * closes.rolling(20).std())
evaluate("bollinger_pos_20", bp, -1)

# ---- 7. drawdown_60d ----
dd = closes / closes.rolling(60).max() - 1.0
evaluate("drawdown_60d", dd, 1)

# ---- 8. ma_cross_20_60 ----
mc = sma(closes, 20) / sma(closes, 60) - 1.0
evaluate("ma_cross_20_60", mc, 1)

# ---- 9. ret_autocorr_5d ----
ac = rets.rolling(5).apply(lambda r: r.autocorr() if len(r) >= 5 and r.std() > 1e-14 else np.nan, raw=False)
evaluate("ret_autocorr_5d", ac, 1)

# ---- 10. vol_percentile_120 ----
vol20 = rets.rolling(20).std()
vp = vol20.rolling(120).rank(pct=True)
evaluate("vol_percentile_120", vp, -1)

# ---- 11/12. downside / upside capture vs equal-weight market ----
dc = capture_ratio(rets, mkt, 60, "down")
evaluate("downside_capture_60", dc, 1)
uc = capture_ratio(rets, mkt, 60, "up")
evaluate("upside_capture_60", uc, 1)

print("\nDone.", flush=True)
