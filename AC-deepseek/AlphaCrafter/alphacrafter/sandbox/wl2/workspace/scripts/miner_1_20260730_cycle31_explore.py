"""miner_1 2026-07-30 cycle 31: orthogonal factor families.

Active library (3): mom20_volproxy60 (momentum), dxy_beta_cond_60x20 (dollar
beta), calmness_20 (low-vol persistence). Admission gates: |IC|>=0.007,
|ICIR|>=0.084 at 10d horizon, |rho|<0.5 vs persisted signal artifacts.

New families (NOT in prior cycles):
  1. yield_beta_cond_60x20 : 60d beta of asset ret on US10Y yield change,
                             x 20d US10Y trend (rate-regime conditioning)
  2. btc_beta_cond_60x20   : 60d beta of asset ret on BTC ret x 20d BTC mom
                             (crypto risk-appetite conditioning)
  3. avg_corr_60           : mean pairwise return correlation with other 14
                             assets (systemic integration vs idiosyncrasy)
  4. rel_strength_20       : 20d mom (skip5) minus leave-one-out mean of the
                             other 14 assets (relative strength vs global trend)
  5. ret_vol_corr_60       : rolling corr(daily ret, volume pct change, 60)
                             (volume-confirmed directionality)
  6. wk52_high_prox        : close / rolling_max(close, 252) (52w high proximity)
  7. kurt_60               : rolling 60d kurtosis of daily returns
  8. yield_beta_raw_60     : plain 60d beta to US10Y yield change (no cond)
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset, forward_returns,
                         compute_ic, validate_factor, report)

panel = load_panel()
print(f"panel: {panel.shape}  {panel.index.min().date()}..{panel.index.max().date()}")
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------- volume panel ----------
frames = {}
for a in panel.columns:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp("2026-07-29")].sort_values("date")
    frames[a] = pd.Series(df["volume"].astype(float).values,
                          index=pd.to_datetime(df["date"]), name=a)
volume_panel = pd.concat(frames, axis=1).sort_index()
print(f"volume_panel: {volume_panel.shape} nan={int(volume_panel.isna().sum().sum())}")

us10y = panel["US10Y"]
cn10y = panel["CN10Y"]
btc = panel["BTC"]

# ---------- 1) yield beta conditional ----------
def beta_to(s, mkt, window=60, minp=30):
    ar = s.pct_change()
    mr = mkt.pct_change()
    df = pd.concat([ar.rename("a"), mr.reindex(ar.index).rename("m")], axis=1).dropna()
    cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
    var = df["m"].rolling(window, min_periods=minp).var()
    return (cov / var).reindex(s.index)

us10y_20 = us10y / us10y.shift(20) - 1.0
signals = {}
signals["yield_beta_cond_60x20"] = per_asset(panel, beta_to, us10y).mul(us10y_20, axis=0)
signals["yield_beta_raw_60"] = per_asset(panel, beta_to, us10y)

# ---------- 2) BTC beta conditional ----------
btc_20 = btc / btc.shift(20) - 1.0
signals["btc_beta_cond_60x20"] = per_asset(panel, beta_to, btc).mul(btc_20, axis=0)

# ---------- 3) avg pairwise correlation 60d ----------
ret_panel = panel.pct_change()
corr_parts = {a: pd.Series(0.0, index=panel.index) for a in panel.columns}
cols = list(panel.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        ci = ret_panel[cols[i]].rolling(60, min_periods=30).corr(ret_panel[cols[j]])
        corr_parts[cols[i]] = corr_parts[cols[i]].add(ci, fill_value=0)
        corr_parts[cols[j]] = corr_parts[cols[j]].add(ci, fill_value=0)
for a in cols:
    corr_parts[a] = corr_parts[a] / 14.0
signals["avg_corr_60"] = pd.DataFrame(corr_parts, index=panel.index)

# ---------- 4) relative strength vs leave-one-out global trend ----------
mom20 = per_asset(panel, lambda s: s.shift(5) / s.shift(25) - 1.0)
n_other = pd.DataFrame({a: mom20.drop(columns=[a]).mean(axis=1) for a in panel.columns},
                       index=panel.index)
signals["rel_strength_20"] = mom20 - n_other

# ---------- 5) ret-volume correlation 60d ----------
def ret_vol_corr(s, v, w=60):
    r = s.pct_change()
    vr = v.pct_change()
    df = pd.concat([r.rename("r"), vr.reindex(r.index).rename("v")], axis=1)
    return df["r"].rolling(w, min_periods=30).corr(df["v"]).reindex(s.index)

rv_parts = {}
for a in panel.columns:
    rv_parts[a] = ret_vol_corr(panel[a].dropna(), volume_panel[a].dropna())
signals["ret_vol_corr_60"] = pd.DataFrame(rv_parts, index=panel.index)

# ---------- 6) 52w high proximity ----------
signals["wk52_high_prox"] = per_asset(panel, lambda s: s / s.rolling(252, min_periods=60).max())

# ---------- 7) kurtosis 60d ----------
def kurt_60(s, w=60):
    return s.pct_change().rolling(w).kurt()
signals["kurt_60"] = per_asset(panel, kurt_60)

# ---------- 8) yield slope (US10Y - CN10Y) momentum conditioning ----------
# slope level applied to yield beta: assets sensitive to rates x slope trend
slope_20 = (us10y / us10y.shift(20) - 1.0) - (cn10y / cn10y.shift(20) - 1.0)
signals["yieldbeta_slope_cond_60x20"] = per_asset(panel, beta_to, us10y).mul(slope_20, axis=0)

for fid, sig in signals.items():
    print(f"    {fid}: nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1) >= 8).sum())}")

# ---------- library artifacts ----------
library = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    library[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

print("\n=== validation (admission h=10) ===")
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
    parts = [fid]
    rd = {}
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
          open("scripts/_miner1_cycle31_explore_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle31_explore_results.json")
print("DONE")
