"""miner_1 2026-07-30: Explore trend-efficiency / mean-reversion character family.

Idea: raw momentum magnitude (close ratio) ignores *how* the move happened. A
move achieved smoothly (high efficiency, negative autocorrelation persistence)
may be more reliable than a choppy move. Candidate signals:

  eff_ratio_20 / eff_ratio_60 : Kaufman efficiency ratio |P_t - P_{t-n}| / sum(|r|)
  autocorr_60                 : lag-1 autocorrelation of daily returns (60d window)
  var_ratio_20x5              : variance ratio var(r5)/(5*var(r1)) over 20 obs
  eff_ratio_20_x_invvol       : efficiency x inverse vol (combo)

All computed on VISIBLE data only (<= 2026-07-29). Cross-sectional Spearman IC
vs 10d forward returns, using all 15 tradable assets (>=8 valid per date).
"""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split)

VIS = "2026-07-29"
close = closes_panel(visible_through=VIS)
rets = close.pct_change()

H = 10
fr = forward_returns(close, H)

# --- build factor family ---
factors = {}

# Kaufman efficiency ratio (trend smoothness)
for n in (10, 20, 60):
    path = close.diff(n).abs()
    volsum = rets.abs().rolling(n).sum()
    factors[f"eff_ratio_{n}"] = path / volsum

# lag-1 autocorrelation of daily returns over trailing window
for w in (30, 60):
    ac = {}
    for s in close.columns:
        r = rets[s]
        # rolling autocorr via cov/var on shifted series
        r1 = r.shift(1)
        ac[s] = (r.rolling(w).cov(r1) / r.rolling(w).var()).reindex(close.index)
    factors[f"autocorr_{w}"] = pd.DataFrame(ac)

# variance ratio: var(5d)/ (5*var(1d)) over trailing window -> >1 trending, <1 mean-reverting
r5 = rets.rolling(5).sum()
vr = {}
for s in close.columns:
    vr[s] = (r5[s].rolling(20).var() / (5.0 * rets[s].rolling(20).var())).reindex(close.index)
factors["var_ratio_20x5"] = pd.DataFrame(vr)

# efficiency x inverse vol combo
invvol20 = 1.0 / rets.rolling(20).std()
factors["eff_ratio_20_x_invvol"] = factors["eff_ratio_20"] * invvol20

# --- evaluate ---
print(f"universe: {close.shape[1]} assets, {close.shape[0]} dates (visible <= {VIS})")
out = {}
for name, f in factors.items():
    f = f.reindex(close.index)
    ic = ic_series(f, fr, min_valid=8)
    m = summary_metrics(ic, f, fr, close, h=H)
    if m is None:
        out[name] = {"gate_pass": False, "reason": "insufficient IC dates",
                     "n_ic_dates": int(len(ic.dropna()))}
        print(f"\n{name}: INSUFFICIENT IC DATES ({len(ic.dropna())})")
        continue
    m["regime"] = regime_split(ic)
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = (m["icir"] or 0) >= 0.084
    m["gate_pass"] = bool(gate_ic and gate_icir)
    out[name] = m
    print(f"\n{name}")
    print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']:.3f} "
          f"cov_ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f} "
          f"GATE={'PASS' if m['gate_pass'] else 'FAIL'}")
    print(f"  decay: {m['decay_ic_by_horizon']}")
    print(f"  regime: {json.dumps(m['regime'])}")

with open("scripts/miner_1_20260730_explore_eff_autocorr_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nwrote results json")
