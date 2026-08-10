"""miner_1 exploration 2026-07-30: screen candidate factor families on the 15-asset
cross-asset universe. Data visible through 2026-07-29. Reports IC/ICIR/coverage/
turnover/decay/library-corr for each candidate so we can pick ideas for rigorous
single-idea validation."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_lib import (load_panel, macro_series, validate_factor, load_library_signals,
                         report, TRADABLES, VISIBLE_THROUGH)

panel = load_panel()
close = panel
ret = close.pct_change()
print(f"panel: {panel.shape}, dates {panel.index.min().date()} .. {panel.index.max().date()} (visible through {VISIBLE_THROUGH})")
print(f"assets: {len(panel.columns)}")

library = load_library_signals(panel)

# ---- candidate factor families (time-series per asset, then cross-sectional IC) ----
cands = {}

# C1: range position 20d: mean of (close-low)/(high-low) - closing-pressure
def range_pos(df, w):
    rng = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return rng.rolling(w).mean()

# C2: 60d drawdown from running max (negative values)
cands["dd_from_high_60d"] = close / close.rolling(60).max() - 1.0

# C3: 120d drawdown from running max
cands["dd_from_high_120d"] = close / close.rolling(120).max() - 1.0

# C4: 20d skewness of returns
cands["skew_20d"] = ret.rolling(20).skew()

# C5: vol term structure: 5d vol / 60d vol
cands["vol_ts_5x60"] = ret.rolling(5).std() / ret.rolling(60).std()

# C6: 20d momentum risk-adjusted (t-stat): mean/std
cands["risk_adj_mom_20d"] = ret.rolling(20).mean() / ret.rolling(20).std()

# C7: 60d mean/std trend
cands["risk_adj_mom_60d"] = ret.rolling(60).mean() / ret.rolling(60).std()

# C8: volume trend: vol_20 / vol_60 (volume expansion) - cross-sectionally comparable
# only via within-asset normalization; raw ratio is unit-free
vol_df = {a: (lambda df: pd.Series(df["volume"].astype(float).values, index=pd.to_datetime(df["date"])))(pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])) for a in TRADABLES}
vol_panel = pd.concat(vol_df, axis=1).sort_index()
vol_panel = vol_panel[vol_panel.index <= pd.Timestamp(VISIBLE_THROUGH)]
vol_ratio = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean()
cands["vol_ratio_20x60"] = vol_ratio

# C9: residual (idiosyncratic) momentum 20d: asset ret - equal-weight market ret, then 20d sum
mkt = ret.mean(axis=1)
resid = ret.sub(mkt, axis=0)
cands["resid_mom_20d"] = resid.rolling(20).sum()

# C10: 20d drawup (distance above 20d low)
cands["dist_from_low_20d"] = close / close.rolling(20).min() - 1.0

# C11: US10Y rate-beta conditional (analog of vix_beta_cond with rates)
us10y = macro_series("US10Y").pct_change()
rate_beta = {}
for a in close.columns:
    df = pd.concat([ret[a].rename("a"), us10y.reindex(ret.index).rename("r")], axis=1).dropna()
    b = df["a"].rolling(60).cov(df["r"]) / df["r"].rolling(60).var()
    rate_beta[a] = b
rate_beta_panel = pd.concat(rate_beta, axis=1)
us10y_20 = macro_series("US10Y") / macro_series("US10Y").shift(20) - 1.0
cands["rate_beta_cond_60x20"] = rate_beta_panel * us10y_20.reindex(rate_beta_panel.index)

# C12: 30d momentum skip 10d
cands["mom_30d_skip10"] = close.shift(10) / close.shift(40) - 1.0

# C13: 60d momentum skip 20d
cands["mom_60d_skip20"] = close.shift(20) / close.shift(80) - 1.0

# C14: EMA crossover ratio (20/60)
cands["ema_20_60"] = close.ewm(span=20, adjust=False).mean() / close.ewm(span=60, adjust=False).mean() - 1.0

# C15: high-low range expansion: (high-low)/close 20d vs 60d ratio (needs OHLC)
hl = {}
for a in TRADABLES:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)]
    hl[a] = ((df["high"] - df["low"]) / df["close"]).set_axis(pd.to_datetime(df["date"]))
hl_panel = pd.concat(hl, axis=1).sort_index()
cands["range_ratio_20x60"] = hl_panel.rolling(20).mean() / hl_panel.rolling(60).mean()

# C16: 20d range position (buying pressure)
cands["range_pos_20d"] = None  # computed below

rng = (close / close.shift(1) - 1)  # placeholder
for a in TRADABLES:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)]
    s = pd.Series(((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)).values,
                  index=pd.to_datetime(df["date"]), name=a)
    cands["range_pos_20d"] = pd.concat([cands["range_pos_20d"], s.rolling(20).mean()], axis=1) if cands["range_pos_20d"] is not None else s.rolling(20).mean().to_frame()
cands["range_pos_20d"] = cands["range_pos_20d"].sort_index()

results = {}
for name, sig in cands.items():
    sig = sig.reindex(panel.index)
    if sig is None or sig.shape[1] != 15:
        continue
    m = validate_factor(sig, panel, horizons=(1, 2, 3, 5, 10, 20),
                        admission_horizon=10, library=library)
    results[name] = m
    report(name, m)

print("\n=== summary ===")
for name, m in sorted(results.items(), key=lambda kv: abs(kv[1]["icir"]), reverse=True):
    print(f"{name:26s} IC={m['ic']: .4f} ICIR={m['icir']: .4f} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov={m['coverage_asset_days']} maxlib={m['max_abs_library_correlation']}")
