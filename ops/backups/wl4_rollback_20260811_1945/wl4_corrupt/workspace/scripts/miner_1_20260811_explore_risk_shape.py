"""miner_1 2026-08-11: explore risk-shape & systemic-structure factor batch.

Candidates (all h=10 admission, 15-asset tradable universe, min 8 valid per date):
  1. skew_20d               - 20d realized skewness (higher moment)
  2. vol_ratio_20_60        - volatility regime acceleration (20d vol / 60d vol)
  3. avg_pair_corr_60d      - avg pairwise return correlation vs all other assets (systemic crowding)
  4. rate_spread_beta_60d   - beta of asset returns on (US10Y ret - CN10Y ret) differential
  5. wti_beta_60d           - 60d beta of asset returns on WTI returns (energy beta)

No lookahead: factor at t uses data <= t; forward returns close[t+h]/close[t]-1.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr, full_eval)

panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
lib = library_signals(panels, closes, rets)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
print(f"panels: {len(panels)} | closes {closes.shape} | dates {closes.index[0].date()}..{closes.index[-1].date()}\n")


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()).where(
            z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


def realized_skew(rets, win=20, min_obs=15):
    out = {}
    for a in rets.columns:
        r = rets[a]
        mu = r.rolling(win).mean()
        sd = r.rolling(win).std()
        m3 = ((r - mu) ** 3).rolling(win).mean()
        sk = (m3 / sd ** 3).where(r.rolling(win).count() >= min_obs)
        out[a] = sk
    return pd.DataFrame(out, index=rets.index)


def avg_pair_corr(rets, win=60, min_obs=40):
    out = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for a in rets.columns:
        parts = []
        for b in rets.columns:
            if b == a:
                continue
            z = pd.concat([rets[a].rename("a"), rets[b].rename("b")], axis=1).dropna()
            c = z["a"].rolling(win).corr(z["b"]).where(z["a"].rolling(win).count() >= min_obs)
            parts.append(c)
        out[a] = pd.concat(parts, axis=1).mean(axis=1)
    return out


def evaluate(name, panel, exp_sign, desc):
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
          f"-> {'PASS' if gate else ''}")
    return m


# ---- 1. skew_20d ----
skew = realized_skew(rets, 20)
evaluate("skew_20d", skew, 1, "20d realized skewness")

# ---- 2. vol_ratio_20_60 ----
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vr = (vol20 / vol60)
evaluate("vol_ratio_20_60", vr, -1, "20d/60d vol regime ratio")

# ---- 3. avg_pair_corr_60d ----
apc = avg_pair_corr(rets, 60)
evaluate("avg_pair_corr_60d", apc, -1, "avg pairwise 60d correlation")

# ---- 4. rate_spread_beta_60d ----
us10 = panels["US10Y"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
spread_ret = us10.pct_change() - cn10.pct_change()
rsb = rolling_beta(rets, spread_ret, 60)
evaluate("rate_spread_beta_60d", rsb, -1, "beta on US10Y-CN10Y return differential")

# ---- 5. wti_beta_60d ----
wti = panels["WTI"]["close"].astype(float)
wti_beta = rolling_beta(rets, wti.pct_change(), 60)
evaluate("wti_beta_60d", wti_beta, 1, "60d beta on WTI returns")

print("\nDone.")
