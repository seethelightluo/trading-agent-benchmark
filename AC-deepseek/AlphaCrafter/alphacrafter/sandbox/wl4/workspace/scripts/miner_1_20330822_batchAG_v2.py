"""miner_1 (2033-08-22): batch AG v2 - optimized cross-asset factor candidates.

Universe: 15 tradable cross-asset instruments (equity idx, commodities, crypto, yields).
Observation-only macro: DXY, USDCNY, USDJPY, EURUSD, VIX (never traded).
Data loaded through previous completed trading day (visible_through=2033-08-19), no lookahead.

Candidates (new / distinct from library {vol_adj_mom_accel_20x60, rate_beta_cn10y_60d,
dn_mkt_beta_60d} and evicted families):
  A. gap_ratio_20d         - mean(|open/prev_close-1|,20) / std(ret,20): info-arrival gaps per unit risk
  B. close_range_pos_20d   - (close - min(low,20)) / (max(high,20) - min(low,20)): 20d range position
  C. autocorr_ret_20d      - lag-1 autocorrelation of daily returns over 20d (reversal/continuation)
  D. rel_strength_20d      - 20d return minus cross-sectional mean 20d return (peer-relative strength)
  E. dxy_beta_60d          - rolling beta of asset ret on DXY ret (dollar sensitivity)
  F. usdjpy_beta_60d       - rolling beta of asset ret on USDJPY ret (carry/risk proxy)
  G. trend_r2_20d          - signed R^2 of linear trend fit over 20d (trend quality)
  H. updown_vol_spread_20d - (std(up-days) - std(down-days)) / std(ret,20): vol asymmetry
  I. candle_body_pos_20d  - mean((close-open)/(high-low),20): average candle body position
  J. vix_regime_mom20      - 20d momentum * (VIX < 60d median): momentum only in calm regime
  K. mom5_vol20            - 5d cumulative return / 20d vol (short risk-adjusted momentum)
  L. btc_beta_60d          - rolling beta of asset ret on BTC ret (crypto beta)
  M. volume_trend_corr_20d - rolling corr(volume, close) over 20d (FIXED vectorized impl)

Admission gates (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Also re-validates the 3 library factors for drift tracking. Yearly IC split for regime robustness.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
    library_signals, full_eval,
)

panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape={closes.shape} range={closes.index.min().date()} -> {closes.index.max().date()}")
print(f"assets={closes.shape[1]} (15-asset cross-asset universe)")

lib_sig = library_signals(panels, closes, rets)
H = 10  # admission horizon
fwd = forward_returns(closes, H)


def clean(fp):
    return fp.replace([np.inf, -np.inf], np.nan)


def rolling_beta(a, b, win, min_obs):
    cov = a.rolling(win, min_periods=min_obs).cov(b)
    var = b.rolling(win, min_periods=min_obs).var()
    return cov / var


def roll_corr_panel(x_panel, y_panel, win, min_obs):
    out = {}
    for a in x_panel.columns:
        z = pd.concat([x_panel[a].rename("x"), y_panel[a].rename("y")], axis=1)
        out[a] = z["x"].rolling(win, min_periods=min_obs).corr(z["y"])
    return pd.DataFrame(out, index=x_panel.index)


def roll_autocorr1_panel(x_panel, win, min_obs):
    """Vectorized lag-1 autocorrelation via rolling cov(x, x.shift(1))/var(x)."""
    xs = x_panel.shift(1)
    cov = x_panel.rolling(win, min_periods=min_obs).cov(xs)
    var = x_panel.rolling(win, min_periods=min_obs).var()
    return cov / var


# ---------------- macro / panels ----------------
us10y_ret = panels["US10Y"]["close"].astype(float).pct_change()
dxy_ret = panels["DXY"]["close"].astype(float).pct_change()
usdjpy_ret = panels["USDJPY"]["close"].astype(float).pct_change()
btc_ret = rets["BTC"]
vix = panels["VIX"]["close"].astype(float)

open_p = pd.concat({a: panels[a]["open"].astype(float) for a in closes.columns}, axis=1).sort_index()
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()

cands = {}

# A. gap ratio 20d
gaps = (open_p / closes.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
cands["gap_ratio_20d"] = clean(gaps.abs().rolling(20, min_periods=15).mean() / rets.rolling(20, min_periods=15).std())

# B. close range position 20d
rng_hi = hi.rolling(20, min_periods=15).max()
rng_lo = lo.rolling(20, min_periods=15).min()
cands["close_range_pos_20d"] = clean((closes - rng_lo) / (rng_hi - rng_lo))

# C. lag-1 autocorrelation of returns 20d (vectorized)
cands["autocorr_ret_20d"] = clean(roll_autocorr1_panel(rets, 20, 15))

# D. peer-relative strength 20d
m20 = closes / closes.shift(20) - 1.0
cands["rel_strength_20d"] = clean(m20 - m20.mean(axis=1))

# E. DXY beta 60d
cands["dxy_beta_60d"] = clean(rolling_beta(rets, dxy_ret, 60, 40))

# F. USDJPY beta 60d
cands["usdjpy_beta_60d"] = clean(rolling_beta(rets, usdjpy_ret, 60, 40))

# G. signed trend R^2 over 20d (corr(log close, time) * |corr|)
t_idx = pd.Series(np.arange(len(closes)), index=closes.index)
logc = np.log(closes)
corr_t = roll_corr_panel(logc, pd.DataFrame({a: t_idx for a in closes.columns}, index=closes.index), 20, 15)
cands["trend_r2_20d"] = clean(corr_t * corr_t.abs())

# H. up/down vol spread 20d
up = rets.where(rets > 0)
dn = rets.where(rets < 0)
up_vol = up.rolling(20, min_periods=10).std()
dn_vol = dn.rolling(20, min_periods=10).std()
cands["updown_vol_spread_20d"] = clean((up_vol - dn_vol) / rets.rolling(20, min_periods=15).std())

# I. candle body position 20d
body = ((closes - open_p) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
cands["candle_body_pos_20d"] = clean(body.rolling(20, min_periods=15).mean())

# J. VIX-regime conditional momentum 20d
vix_med60 = vix.rolling(60, min_periods=40).median()
calm = (vix < vix_med60).astype(float)
cands["vix_regime_mom20"] = clean(m20 * calm)

# K. short risk-adjusted momentum 5/20
m5 = closes / closes.shift(5) - 1.0
cands["mom5_vol20"] = clean(m5 / rets.rolling(20, min_periods=15).std())

# L. BTC beta 60d
cands["btc_beta_60d"] = clean(rolling_beta(rets, btc_ret, 60, 40))

# M. volume-trend correlation 20d (fixed impl)
cands["volume_trend_corr_20d"] = clean(roll_corr_panel(vol_panel, closes, 20, 15))

# ---------------- evaluate ----------------
print("\n=== LIBRARY FACTORS (drift re-validation, h=10) ===")
for name, fp in lib_sig.items():
    m, ics = full_eval(fp, closes, library=lib_sig, expected_sign=1)
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']} cov_days={m['coverage_asset_days']:.2f} cov8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')}")

print("\n=== CANDIDATE FACTORS (h=10, gates |IC|>=0.0070, |ICIR|>=0.0840) ===")
rows = []
for name, fp in cands.items():
    ics = rank_ic_series(fp, fwd, min_valid=8)
    if len(ics) < 30:
        print(f"{name:26s} INSUFFICIENT dates ({len(ics)})")
        continue
    m = summarize_ic(ics, expected_sign=1)
    m.update(coverage_metrics(fp, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(fp, 10)
    m["decay_ic_by_horizon"] = decay_profile(fp, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(fp, lib_sig)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    # yearly IC split for regime robustness
    yr = ics.groupby(ics.index.year).mean()
    yr_str = " ".join(f"{y}:{v:+.3f}" for y, v in yr.items())
    gate = "PASS" if (abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840) else "FAIL"
    rows.append((name, m, gate))
    dec = m["decay_ic_by_horizon"]
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']} "
          f"cov_days={m['coverage_asset_days']:.2f} cov8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']} "
          f"maxcorr={corr:.3f}({key}) decay1/10/20={dec.get('1')}/{dec.get('10')}/{dec.get('20')} => {gate}")
    print(f"{'':26s} yearly_IC: {yr_str}")

print("\n=== PASS SUMMARY ===")
for name, m, gate in rows:
    if gate == "PASS":
        print(f"{name} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} maxcorr={m['max_abs_library_correlation']:.3f} turn={m['turnover_10d_rank']}")
