"""miner_2 exploration cycle 2026-07-30: test multiple NEW candidate factors
on the 15-asset tradable universe, visible window only (<= 2026-07-29)."""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split, load_panel)

close = closes_panel()
macro = macro_closes()
frames = load_panel()
ret = close.pct_change()

# --- data helpers ---
vol = pd.DataFrame({s: df.set_index("date")["volume"].astype(float) for s, df in frames.items()}).sort_index()
vol = vol.reindex(close.index)
high = pd.DataFrame({s: df.set_index("date")["high"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
low = pd.DataFrame({s: df.set_index("date")["low"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)

def beta_panel(asset_ret, mkt_ret, win):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win).cov(pair["m"]) / pair["m"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)

def eval_factor(name, factor, verbose=True):
    fr = forward_returns(close, 10)
    ics = ic_series(factor, fr)
    if verbose:
        print(f"=== {name} === n_ic={len(ics)}")
    if len(ics) < 30:
        print(f"  INSUFFICIENT IC dates: {len(ics)}")
        return None, None
    m = summary_metrics(ics, factor, fr, close, h=10)
    if m is None:
        print("  INSUFFICIENT")
        return None, None
    lib_ics = library_ic_series_map(close)
    rho = max_abs_library_corr(ics, lib_ics)
    m["max_abs_library_correlation"] = rho
    print("  ic:", m["ic"], "icir:", m["icir"], "hit:", m["ic_hit_ratio"],
          "cov_dates_ge8:", m["coverage_dates_ge8"], "turn:", m["turnover_10d_rank"])
    rs = regime_split(ics)
    print("  regimes:", {k: (v["ic"], v["icir"], v["n"]) for k, v in rs.items()})
    print("  decay:", m["decay_ic_by_horizon"])
    print("  max_abs_lib_corr:", rho)
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and len(ics) >= 30
    print("  PASSES GATE:", passed)
    return m, ics

# ---- Candidate 1: Range position (intraday buying pressure) 10d avg ----
rp = (close - low) / (high - low).replace(0, np.nan)
range_pos = rp.rolling(10).mean()
eval_factor("RANGE_POS_10D", range_pos)

# ---- Candidate 2: Vol compression ratio 10x60 ----
rv10 = ret.rolling(10).std()
rv60 = ret.rolling(60).std()
vol_ratio = rv10 / rv60 - 1.0
eval_factor("VOL_RATIO_10X60", vol_ratio)

# ---- Candidate 3: BTC/ETH spread beta conditional ----
btc = close["BTC"]; eth = close["ETH"]
spread = btc / eth
spread_ret = spread.pct_change()
b_spread = beta_panel(ret, spread_ret, 60)
spread_mom = spread / spread.shift(20) - 1.0
crypto_cond = b_spread.mul(spread_mom, axis=0)
eval_factor("CRYPTO_SPREAD_BETA_60X20", crypto_cond)

# ---- Candidate 4: Rate spread (US10Y - CN10Y) beta conditional ----
us10 = close["US10Y"]; cn10 = close["CN10Y"]
spread10 = us10 - cn10
spread10_ret = spread10.pct_change()
b_rate = beta_panel(ret, spread10_ret, 60)
rate_mom = spread10 / spread10.shift(20) - 1.0
rate_cond = b_rate.mul(rate_mom, axis=0)
eval_factor("RATE_SPREAD_BETA_60X20", rate_cond)

# ---- Candidate 5: 252d high distance (trend) ----
high252 = close.rolling(252).max()
dist_high = close / high252 - 1.0
eval_factor("DIST_52W_HIGH", dist_high)

# ---- Candidate 6: Volume trend ratio 20x120 ----
v20 = vol.rolling(20).mean()
v120 = vol.rolling(120).mean()
vol_tr = v20 / v120 - 1.0
eval_factor("VOL_TREND_20X120", vol_tr)

# ---- Candidate 7: Downside vol (semi-deviation) 60d ----
neg = ret.where(ret < 0, 0.0)
down_vol = np.sqrt((neg ** 2).rolling(60).mean())
eval_factor("DOWN_VOL_60", down_vol)

# ---- Candidate 8: Efficiency ratio 20d ----
chg = (close - close.shift(20)).abs()
path = ret.abs().rolling(20).sum()
eff = chg / path.replace(0, np.nan)
eval_factor("EFF_RATIO_20", eff)

# ---- Candidate 9: DXY trend x asset DXY-beta (dollar regime momentum) ----
dxy = macro["DXY"]
dxy_ret = dxy.pct_change()
b_dxy = beta_panel(ret, dxy_ret, 60)
dxy_mom = dxy / dxy.shift(60) - 1.0
dxy_cond = b_dxy.mul(dxy_mom, axis=0)
eval_factor("DXY_BETA_TREND_60X60", dxy_cond)
