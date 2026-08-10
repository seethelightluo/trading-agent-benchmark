"""miner_2 cycle34: liquidity & volume-dynamics family. (fixed paired helper)

Rationale: the library has no volume/liquidity factor yet (momentum, vol,
macro-beta, path-structure, recovery are covered). Volume is a first-class
column in the simulator data. Ideas:
  - volume_trend_10x60   : mean(vol,10)/mean(vol,60) - 1  (volume expansion)
  - amihud_ratio_10x60   : mean(|ret|/vol,10)/mean(|ret|/vol,60) - 1
                           (Amihud illiquidity rising = liquidity deterioration)
  - vol_vol_corr_20      : rolling 20d corr(daily return, d(volume)) (price-volume
                           alignment; divergences flag distribution/positioning)
  - dollar_vol_z_20      : (vol_10 - mean(vol,60))/std(vol,60) (abnormal volume)

All ratios are computed within each asset (own scale) so heterogeneous volume
units across indices/crypto/yields do not distort the cross-section.

Admission gates: abs(IC)>=0.0070, abs(ICIR)>=0.0840 @10d horizon.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, load_volume_panel, per_asset,
                        compute_ic, forward_returns, validate_factor,
                        regime_breakdown, report)

close = load_close_panel()
vol = load_volume_panel()
print(f"close panel {close.shape}, volume panel {vol.shape}")
print("volume per-asset valid cells / coverage:")
for c in vol.columns:
    s = vol[c]
    print(f"  {c:12s} n_valid={int(s.notna().sum()):5d} n_zero={(s == 0).sum():5d} "
          f"min={s.min():.3g} med={s.median():.3g}")

idx = close.index
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
       "downside_dev_60", "days_since_high_60"]
lib = {}
for e in EFF:
    p = Path("factors") / f"{e}.signal.npy"
    if p.exists():
        a = np.load(p)
        if a.shape[0] == len(idx):
            lib[e] = pd.DataFrame(a, index=idx, columns=close.columns)
print(f"[lib] loaded {len(lib)} artifacts")

fwd = {str(h): forward_returns(close, h) for h in (1, 2, 3, 5, 10, 20)}


def per_asset_pair(panel_c, panel_v, func, *a, **kw):
    out = {}
    for c in panel_c.columns:
        s1 = panel_c[c].dropna()
        s2 = panel_v[c].reindex(s1.index)
        out[c] = func(s1, s2, *a, **kw).reindex(panel_c.index)
    return pd.DataFrame(out, index=panel_c.index)


def volume_trend(s, w1=10, w2=60, mp=30):
    return s.rolling(w1, min_periods=5).mean() / s.rolling(w2, min_periods=mp).mean() - 1.0


def amihud_ratio(close_s, vol_s, w1=10, w2=60, mp=30):
    r = close_s.pct_change().abs()
    am = (r / vol_s.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return am.rolling(w1, min_periods=5).mean() / am.rolling(w2, min_periods=mp).mean() - 1.0


def vol_vol_corr(close_s, vol_s, w=20, mp=10):
    r = close_s.pct_change()
    dv = vol_s.pct_change()
    return r.rolling(w, min_periods=mp).corr(dv)


def dollar_vol_z(vol_s, w1=10, w2=60, mp=30):
    m = vol_s.rolling(w2, min_periods=mp).mean()
    sd = vol_s.rolling(w2, min_periods=mp).std()
    return (vol_s.rolling(w1, min_periods=5).mean() - m) / sd.replace(0, np.nan)


cands = {
    "volume_trend_10x60": per_asset(vol, volume_trend),
    "amihud_ratio_10x60": per_asset_pair(close, vol, amihud_ratio),
    "vol_vol_corr_20": per_asset_pair(close, vol, vol_vol_corr),
    "dollar_vol_z_20": per_asset(vol, dollar_vol_z),
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, close, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", m.get("library_pairwise_corr"))
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:22s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner2_cycle34_liquidity_results.json", "w"), indent=1, default=float)
print("DONE")
