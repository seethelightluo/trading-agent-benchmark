"""miner_1 2026-07-30 cycle 25: new factor families absent from library.

Motivation: the 6-factor ensemble is momentum/trend/carry/macro-beta concentrated.
Library has NO volume-based factors, no skewness/asymmetry, no volatility
term-structure tilt, no crypto-sentiment beta. Explore:

  1. vol_term_10x60      : (20d vol / 60d vol - 1)  vol term-structure tilt
  2. vol_expansion_20x60 : (20d avg VOLUME / 60d avg VOLUME - 1)  volume expansion
  3. skew_60d            : rolling 60d skewness of daily returns
  4. downside_ratio_60d  : downside semi-dev / total std over 60d
  5. btc_beta_60d        : rolling 60d beta of asset ret on BTC ret (crypto sentiment)
  6. volconf_mom_20x20   : 20d momentum (skip5) x vol-expansion z (volume-confirmed mom)
  7. mom_accel_20x60     : 20d momentum - 60d momentum (trend acceleration)

All factors are per-asset (one value per tradable per date); cross-sectional
Spearman IC vs h-day forward returns on each asset's own calendar. Admission
horizon = 10d. Gates: |IC|>=0.007, |ICIR|>=0.084, n>=800, cov_dates>=0.5.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset, forward_returns,
                         compute_ic, validate_factor, load_library_signals, report)

panel = load_panel()
print(f"panel: {panel.shape}  {panel.index.min().date()}..{panel.index.max().date()}")
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

signals = {}
print("\n=== building candidate signals ===\n")

# 1) volatility term structure: 20d realized vol / 60d realized vol - 1
def vol_term(s, w1=20, w2=60):
    v1 = s.pct_change().rolling(w1).std()
    v2 = s.pct_change().rolling(w2).std()
    return v1 / v2 - 1.0
signals["vol_term_10x60"] = per_asset(panel, vol_term)
print("  vol_term_10x60 built")

# 2) volume expansion: 20d avg volume / 60d avg volume - 1
def vol_exp(s, w1=20, w2=60):
    v1 = s.rolling(w1).mean()
    v2 = s.rolling(w2).mean()
    return v1 / v2 - 1.0
volume_panel = None
frames = {}
for a in panel.columns:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp("2026-07-29")].sort_values("date")
    frames[a] = pd.Series(df["volume"].astype(float).values, index=pd.to_datetime(df["date"]), name=a)
volume_panel = pd.concat(frames, axis=1).sort_index()
signals["vol_expansion_20x60"] = per_asset(volume_panel, vol_exp)
print("  vol_expansion_20x60 built")

# 3) skewness 60d
def skew_60(s, w=60):
    r = s.pct_change()
    return r.rolling(w).skew()
signals["skew_60d"] = per_asset(panel, skew_60)
print("  skew_60d built")

# 4) downside ratio: downside semi-dev / total std over 60d
def downside_ratio(s, w=60):
    r = s.pct_change()
    tot = r.rolling(w).std()
    neg = r.clip(upper=0.0)
    dsd = np.sqrt((neg ** 2).rolling(w).mean())
    return dsd / tot
signals["downside_ratio_60d"] = per_asset(panel, downside_ratio)
print("  downside_ratio_60d built")

# 5) crypto-sentiment beta: beta of asset ret on BTC ret (60d)
btc = panel["BTC"]
btc_ret = btc.pct_change()
def btc_beta(s, window=60, minp=30):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), btc_ret.reindex(ar.index).rename("m")], axis=1).dropna()
    cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
    var = df["m"].rolling(window, min_periods=minp).var()
    return (cov / var).reindex(s.index)
signals["btc_beta_60d"] = per_asset(panel, btc_beta)
print("  btc_beta_60d built")

# 6) volume-confirmed momentum: 20d mom (skip5) x z-scored volume expansion
mom20 = per_asset(panel, lambda s: s.shift(5) / s.shift(25) - 1.0)
vexp_z = signals["vol_expansion_20x60"].apply(lambda col: (col - col.mean()) / col.std(), axis=0)
signals["volconf_mom_20x20"] = mom20 * vexp_z.clip(-3, 3)
print("  volconf_mom_20x20 built")

# 7) trend acceleration: 20d mom - 60d mom
mom60 = per_asset(panel, lambda s: s / s.shift(60) - 1.0)
signals["mom_accel_20x60"] = mom20 - mom60
print("  mom_accel_20x60 built")

for fid, sig in signals.items():
    print(f"    {fid}: nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# ---------------------------------------------------------------------------
print("\n=== validation (admission h=10) ===")
library = load_library_signals(panel)
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

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5]
print(f"\nPASSERS (gate + robustness): {passers}")

print("\n=== regime breakdown for passers ===")
regime_out = {}
for fid in passers:
    sig = signals[fid]
    rd = {}
    parts = [fid]
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            parts.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n_dates": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(parts))

json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                       for k, v in results.items()},
           "passers": passers, "regime": regime_out},
          open("scripts/_miner1_cycle25_explore_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle25_explore_results.json")
print("DONE")
