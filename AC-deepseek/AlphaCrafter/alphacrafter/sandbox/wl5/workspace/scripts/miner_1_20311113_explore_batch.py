# -*- coding: utf-8 -*-
"""miner_1 2031-11-13 batch exploration: new candidate factor ideas.
Visible data through 2031-11-12. Daily cross-sectional Spearman IC vs 10d forward return.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner_1_20311113_common as C

close = C.price_panel("close")
high = C.price_panel("high")
low = C.price_panel("low")
vol = C.price_panel("volume")
macro = C.macro_panel()

rets = close.pct_change()
fwd10 = close.shift(-10) / close - 1.0

cands = {}

# 1) Range position: mean of (close-low)/(high-low) over 20d (closing strength within day range)
hl = (high - low).replace(0, np.nan)
pos = (close - low) / hl
cands["range_pos_20"] = pos.rolling(20, min_periods=10).mean()

# 2) Momentum acceleration: mom_20 - mom_60 (2nd derivative proxy)
cands["mom_accel_20x60"] = (close.shift(5)/close.shift(25) - 1.0) - (close.shift(5)/close.shift(65) - 1.0)

# 3) Downside vol ratio: downside std / total std over 20d
r = close.pct_change()
down = r.where(r < 0, 0.0)
down_std = down.rolling(20, min_periods=10).std()
tot_std = r.rolling(20, min_periods=10).std()
cands["downside_vol_ratio_20"] = down_std / tot_std

# 4) Volume trend: 20d avg vol / 120d avg vol (participation expansion)
v20 = vol.rolling(20, min_periods=10).mean()
v120 = vol.rolling(120, min_periods=40).mean()
cands["vol_trend_20x120"] = v20 / v120

# 5) Amihud illiquidity: |ret| / volume, smoothed 20d (log)
amihud = (r.abs() / vol).replace([np.inf, -np.inf], np.nan)
cands["amihud_illiq_20"] = np.log(amihud.rolling(20, min_periods=10).mean())

# 6) Variance ratio (trend persistence): var(5d)/ (5*var(1d)) over 60d window
vr_num = close.pct_change(5).rolling(60, min_periods=30).var()
vr_den = r.rolling(60, min_periods=30).var()
cands["var_ratio_5x60"] = vr_num / (5.0 * vr_den)

# 7) Tech beta: 60d beta of asset to SOX (semiconductor sensitivity)
sox_r = close["SOX"].pct_change()
beta_sox = r.rolling(60, min_periods=30).cov(sox_r) / sox_r.rolling(60, min_periods=30).var()
cands["sox_beta_60"] = beta_sox

# 8) Range contraction: 20d mean of (high-low)/close (normalized range) - low = squeeze
nr = (high - low) / close
cands["range_norm_20"] = -nr.rolling(20, min_periods=10).mean()

results = {}
for name, fp in cands.items():
    ics = C.rank_ic(fp, fwd10)
    summ = C.summarize_ic(ics, 10, label=name)
    if summ is None:
        continue
    reg = C.ic_by_regime(ics)
    cov = C.coverage_report(fp)
    tov = C.turnover_rank(fp, 10)
    corr = C.library_correlation(fp, close, macro)
    maxrho = max((v["pooled_rho"] for v in corr.values() if v["pooled_rho"] is not None), default=0.0)
    gate = C.admission_check(summ["ic"], summ["icir"], label=name)
    results[name] = {
        "metrics": {**summ, "turnover_10d_rank": tov, **cov},
        "regime": reg,
        "max_abs_library_correlation": round(maxrho, 4),
        "library_rho": {k: (round(v["pooled_rho"], 3) if v["pooled_rho"] is not None else None) for k, v in corr.items()},
        "gate": gate,
    }
    print(f"  regime={reg}  cov_asset_days={cov['coverage_asset_days']:.3f} dates_ge8={cov['coverage_dates_ge8']:.3f} "
          f"turn={tov:.3f} maxlibrho={maxrho:.3f}")

with open("scripts/miner_1_20311113_explore_batch_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nSaved results json. Passed candidates:")
for k, v in results.items():
    if v["gate"]:
        print("  PASS:", k, v["metrics"]["ic"], v["metrics"]["icir"])
