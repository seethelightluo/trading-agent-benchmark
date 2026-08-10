"""miner_1 2026-07-30 cycle 25 (part 2): refine BTC-beta family.

btc_beta_60d passed gates (IC +0.0622, ICIR +0.1584) but 2025-26 IC ~ 0.
Test conditional variants that gate the beta on the BTC trend (like the
successful dxy_beta_cond_60x20 construction):

  1. btc_beta_cond_60x20 : beta60(asset,BTC) x BTC 20d momentum
  2. btc_beta_cond_20x20 : beta20(asset,BTC) x BTC 20d momentum (faster beta)
  3. btc_beta_60d         : original for reference + turnover/decay detail

Also print pairwise rho vs ACTIVE ensemble only (old-library vol_of_vol20x60
is not part of the online ensemble).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, load_library_signals, report,
                         panel_rank_corr, turnover_rank, coverage_stats)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

btc = panel["BTC"]
btc_ret = btc.pct_change()
btc_mom20 = btc / btc.shift(20) - 1.0

def btc_beta_factory(window, minp=30):
    def f(s):
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), btc_ret.reindex(ar.index).rename("m")], axis=1).dropna()
        cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
        var = df["m"].rolling(window, min_periods=minp).var()
        return (cov / var).reindex(s.index)
    return f

beta60 = per_asset(panel, btc_beta_factory(60))
beta20 = per_asset(panel, btc_beta_factory(20, minp=10))

signals = {
    "btc_beta_60d": beta60,
    "btc_beta_cond_60x20": beta60.mul(btc_mom20.reindex(beta60.index), axis=0),
    "btc_beta_cond_20x20": beta20.mul(btc_mom20.reindex(beta20.index), axis=0),
}

# active ensemble only (the 6 factors actually used online)
library = {}
for fid in ["mom20_volproxy60", "mom_curve_volscale", "range_pos_120d",
            "carry_12m3m", "carry_3m1m", "dxy_beta_cond_60x20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    library[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

print("\n=== turnover / decay detail ===")
for fid, sig in signals.items():
    to = turnover_rank(sig, step=ADM_H)
    print(f"  {fid}: turnover_10d_rank={to:.3f}")
    print(f"    decay(1,2,3,5,10,20) = {results[fid]['decay_ic_by_horizon']}")
    print(f"    coverage_asset={results[fid]['coverage_asset_days']} "
          f"cov_dates_ge8={results[fid]['coverage_dates_ge8']}")

print("\n=== pairwise rho vs ACTIVE ensemble ===")
for fid, sig in signals.items():
    for lid in library:
        print(f"  {fid:24s} vs {lid:24s} = {panel_rank_corr(sig, library[lid]):+.4f}")

print("\n=== regime (10d IC/ICIR) for conditional variants ===")
for fid in ["btc_beta_cond_60x20", "btc_beta_cond_20x20"]:
    sig = signals[fid]
    parts = [fid]
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            parts.append(f"{r0[:4]}: {ic_ser.mean():+.4f}/{ic_ser.mean()/sd:+.3f}/n={len(ic_ser)}")
    print("  " + " | ".join(parts))

json.dump({k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
           for k, v in results.items()},
          open("scripts/_miner1_cycle25_btcbeta_refine.json", "w"), indent=1, default=float)
print("\nDONE")
