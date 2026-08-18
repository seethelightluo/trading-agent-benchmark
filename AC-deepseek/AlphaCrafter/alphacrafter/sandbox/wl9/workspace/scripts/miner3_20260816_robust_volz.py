"""Robustness check for vol_z_20d (miner_3, 2026-08-16).

Confirms the volume-participation z-score signal before persistence:
1) window variants (10/20/60)
2) per-asset IC contribution (which assets drive the signal)
3) crypto-excluded subset (drop BTC/ETH)
4) sub-period stability (2020-2023 vs 2023-2026)
5) direction consistency (positive IC at h=10 -> volume expansion predicts gains)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

VOL_ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"]
closes = load_closes()

def load_volumes():
    out = {}
    for a in VOL_ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")["volume"].astype(float)
    return out

vols = load_volumes()

def vol_z(volume, n=20):
    mu = volume.rolling(n).mean()
    sd = volume.rolling(n).std(ddof=0).replace(0, np.nan)
    return (volume - mu) / sd

# 1) window variants
for n in (10, 20, 60):
    vals = {a: vol_z(vols[a], n) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes}
    evaluate(closes, vals, f"vol_z_{n}d", horizon=10)

# 2) per-asset IC (rank IC over time for each asset alone, h=10)
print("\n=== per-asset predictive power (asset-level rank IC, h=10) ===")
for a in VOL_ASSETS:
    s = vol_z(vols[a], 20)
    r = closes[a].shift(-10) / closes[a] - 1.0
    pair = pd.concat([s.rename("f"), r.rename("r")], axis=1).dropna()
    if len(pair) > 60:
        ic = pair["f"].rank().corr(pair["r"].rank())
        print(f"  {a}: n={len(pair)} ic={ic:.4f}")

# 3) crypto-excluded subset
print("\n=== vol_z_20d without BTC/ETH ===")
sub = [a for a in ASSETS if a not in ("BTC", "ETH")]
vals_no = {a: vol_z(vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in sub}
# reduced evaluate: use evaluate with closes filtered
closes_no = {a: closes[a] for a in sub}
evaluate(closes_no, vals_no, "vol_z_20d_no_crypto", horizon=10)

# 4) sub-period stability
def eval_period(vals, label, start, end):
    from miner3_20260730_harness import to_frame, forward_returns, rank_ic, weekday_grid
    frame = to_frame(closes, vals)
    frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
    rets = forward_returns(closes, 10)
    ret_frame = pd.DataFrame({a: rets[a].reindex(frame.index) for a in frame.columns})
    ic = rank_ic(frame, ret_frame)
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    ic_std = float(ic.std(ddof=1)) if len(ic) > 2 else float("nan")
    icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) else float("nan")
    print(f"  {label}: n_ic_dates={len(ic)} ic={ic_mean:.4f} icir={icir:.4f}")

vals20 = {a: vol_z(vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes}
print("\n=== sub-period stability (h=10) ===")
eval_period(vals20, "2020-01-01..2023-06-30", "2020-01-01", "2023-06-30")
eval_period(vals20, "2023-07-01..2026-07-15", "2023-07-01", "2026-07-15")