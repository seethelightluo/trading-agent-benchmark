"""miner_2 exploration batch 1 (2026-08-11): screen new factor families.

Candidates (all distinct from existing library of momentum / vol-of-vol /
conditional-beta factors):
  A risk_adj_mom_60d     : 60d momentum scaled by 20d vol (trend quality)
  B parkinson_vol_inv_20d: inverse Parkinson range-vol (low range-vol wins)
  C skew_60d             : realized skewness of daily returns (60d)
  D us10y_beta_60d       : rolling beta to US10Y yield changes
  E usdjpy_beta_60d      : rolling beta to USDJPY returns (risk-on/off)
  F dxy_beta_60d         : rolling beta to DXY returns
  G max_dd_60d           : distance from 60d rolling max (drawdown)
  H downside_vol_ratio_60d: downside std / total std (60d)
  I range_ratio_20d      : mean((high-low)/close) over 20d (negated)
  J vol_surprise_5d      : volume / 60d mean volume (5d avg, negated)
  K bollinger_pos_20d    : (close-sma20)/std20
  L spread_beta_60d      : beta to (US10Y-CN10Y) spread changes

Gate (benchmark worldline): |IC|>=0.007, |ICIR|>=0.084 at 10d horizon,
max_abs_library_correlation < 0.5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr, TRADABLE,
)

HORIZON = 10
MIN_VALID = 8
WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-07-15"))

panels = load_panels(3000)
closes = close_panel(panels)
rets = closes.pct_change()
# restrict to validation window
closes = closes.loc[(closes.index >= WINDOW[0]) & (closes.index <= WINDOW[1])]
rets = rets.loc[(rets.index >= WINDOW[0]) & (rets.index <= WINDOW[1])]

vix = panels["VIX"]["close"].astype(float)
dxy = panels["DXY"]["close"].astype(float)
usdjpy = panels["USDJPY"]["close"].astype(float)
eurusd = panels["EURUSD"]["close"].astype(float)

# ---------- library signals (existing 7 effective factors) ----------
lib = {}
# 4 formula-based
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
vix_ret = vix.pct_change()
beta_vix = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
    beta_vix[a] = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
beta_vix_df = pd.DataFrame(beta_vix, index=rets.index)
lib["vix_beta_cond_60x20"] = -beta_vix_df * (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = lib["vix_beta_cond_60x20"].reindex(closes.index)

# 3 CSV-persisted beta factors (pivot long -> wide)
for name in ["dn_mkt_beta_60d", "eurusd_beta_60d", "rate_beta_cn10y_60d"]:
    p = Path("factors") / f"{name}_signal.csv"
    if p.exists():
        long_df = pd.read_csv(p, index_col=0)
        wide = long_df.pivot_table(index="date", columns="symbol", values="value")
        wide.index = pd.to_datetime(wide.index)
        lib[name] = wide.reindex(closes.index)
        print(f"loaded library csv: {name} {wide.shape}")
    else:
        print(f"MISSING library csv: {name}")

lib = {k: v.reindex(closes.index) for k, v in lib.items()}

# ---------- candidate factors ----------
cand = {}

# A: risk-adjusted momentum: 60d return / 20d vol
mom60 = closes.shift(5) / closes.shift(65) - 1.0
vol20 = rets.rolling(20).std()
cand["risk_adj_mom_60d"] = mom60 / vol20

# B: inverse Parkinson vol 20d
hl = np.log(panels["XAU"]["high"])  # placeholder to silence; real below
park = {}
for a in TRADABLE:
    h = panels[a]["high"].astype(float).reindex(closes.index)
    l = panels[a]["low"].astype(float).reindex(closes.index)
    park[a] = ((np.log(h / l) ** 2).rolling(20).mean())
park_df = pd.DataFrame(park, index=closes.index)
cand["parkinson_vol_inv_20d"] = -park_df.apply(np.sqrt, axis=0)

# C: realized skewness 60d
cand["skew_60d"] = rets.rolling(60).skew()

# D/E/F/L: rolling beta to macro yield/fx series
def rolling_beta(y_panel, x_series, win=60):
    xr = x_series.pct_change()
    out = {}
    for a in y_panel.columns:
        z = pd.concat([y_panel[a].rename("y"), xr.rename("x")], axis=1).dropna()
        b = z["y"].rolling(win).cov(z["x"]) / z["x"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out, index=y_panel.index)

cand["us10y_beta_60d"] = rolling_beta(rets, panels["US10Y"]["close"].astype(float), 60)
cand["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy, 60)
cand["dxy_beta_60d"] = rolling_beta(rets, dxy, 60)
spread = panels["US10Y"]["close"].astype(float) - panels["CN10Y"]["close"].astype(float)
cand["spread_beta_60d"] = rolling_beta(rets, spread, 60)

# G: max drawdown distance 60d (negated: deeper drawdown -> lower value)
cand["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0

# H: downside vol ratio 60d
def downside_ratio(r, win=60):
    neg = r.where(r < 0, 0.0)
    return neg.rolling(win).std() / r.rolling(win).std()

cand["downside_vol_ratio_60d"] = -downside_ratio(rets, 60)

# I: range ratio 20d (negated: low range -> high value)
rng = {}
for a in TRADABLE:
    h = panels[a]["high"].astype(float).reindex(closes.index)
    l = panels[a]["low"].astype(float).reindex(closes.index)
    c = closes[a]
    rng[a] = ((h - l) / c).rolling(20).mean()
cand["range_ratio_20d"] = -pd.DataFrame(rng, index=closes.index)

# J: volume surprise 5d / 60d mean (negated: low surprise -> high value)
vsurp = {}
for a in TRADABLE:
    v = panels[a]["volume"].astype(float).reindex(closes.index)
    vsurp[a] = (v / v.rolling(60).mean()).rolling(5).mean()
cand["vol_surprise_5d"] = -pd.DataFrame(vsurp, index=closes.index)

# K: bollinger position 20d
sma20 = closes.rolling(20).mean()
std20 = closes.rolling(20).std()
cand["bollinger_pos_20d"] = (closes - sma20) / std20

# ---------- evaluate ----------
fwd = forward_returns(closes, HORIZON)
rows = []
for name, panel in cand.items():
    panel = panel.reindex(closes.index)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), MIN_VALID, 1)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    pass_ic = abs(m["ic"]) >= 0.007
    pass_icir = abs(m["icir"]) >= 0.084
    pass_corr = corr < 0.5
    rows.append((name, m, pass_ic, pass_icir, pass_corr))
    print(f"\n=== {name} ===")
    print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']}")
    print(f"  cov_asset={m['coverage_asset_days']:.3f} cov_dates_ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']}")
    print(f"  decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} }")
    print(f"  max_lib_corr={corr:.3f} ({key})  GATES: IC={pass_ic} ICIR={pass_icir} CORR={pass_corr}")

print("\n===== SUMMARY =====")
for name, m, pi, pir, pc in rows:
    flag = "PASS" if (pi and pir and pc) else "FAIL"
    print(f"{flag:4s} {name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} corr={m['max_abs_library_correlation']:.3f}")
