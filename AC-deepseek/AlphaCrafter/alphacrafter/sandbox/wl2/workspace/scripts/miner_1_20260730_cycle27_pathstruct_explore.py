"""miner_1 2026-07-30 cycle 27: path-structure & market-linkage factor family.

Motivation: cycles 14-26 showed that momentum, carry, range, and vol-surge
families all exceed |rho|>0.5 vs mom20_volproxy60 and get evicted at the
library correlation gate. This cycle targets genuinely ORTHOGONAL signal
dimensions built from price PATH STRUCTURE (not raw drift):

 1. er_20d          : Kaufman efficiency ratio 20d = |net change| / gross path.
                      High ER = smooth trend, low ER = choppy. Normalizes by path
                      length so it is NOT the same as raw momentum.
 2. er_60d          : same at 60d.
 3. mkt_beta_60d    : 60d rolling beta of asset return on equal-weight 15-asset
                      index return (systematic risk linkage).
 4. corr_ew_20d     : 20d rolling correlation with the EW index (linkage strength).
 5. gap_vol_ratio20 : close-close vol / mean(log(high/low)) over 20d. >1 means
                      overnight-gap-driven vol; <1 means continuous intraday vol.
 6. intraday_mom_20 : mean(close/open - 1) over 20d - persistence of the intraday
                      session component (drift within the day, excludes overnight).
 7. overnight_gap_5 : mean(open/prev_close - 1) over 5d - overnight gap drift.
 8. dd_60d          : drawdown depth from 60d rolling max (reversal/risk factor).
 9. er_ratio_20x60  : er_20 - er_60 (trend quality acceleration).

All per-asset on the asset's own calendar, reindexed to union panel; cross-sectional
Spearman IC at h=10 (admission), gates |IC|>=0.007, |ICIR|>=0.084, n>=800,
cov_dates_ge8>=0.5, max_abs_library_correlation vs persisted artifacts <0.5.
Also re-validates the 2 ACTIVE library factors for drift monitoring.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, report, panel_rank_corr)

panel = load_panel()
print(f"panel: {panel.shape}  {panel.index.min().date()}..{panel.index.max().date()}")
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---- OHLC panels (own-calendar aligned to union index) ----
def load_ohlc_panel(field):
    frames = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp("2026-07-29")].sort_values("date")
        frames[a] = pd.Series(df[field].astype(float).values,
                              index=pd.to_datetime(df["date"]), name=a)
    return pd.concat(frames, axis=1).sort_index().reindex(panel.index)

open_p = load_ohlc_panel("open")
high_p = load_ohlc_panel("high")
low_p = load_ohlc_panel("low")

# ---- 1-2) Kaufman efficiency ratio ----
def er_factory(n):
    def f(s):
        diff = s.diff().abs().rolling(n).sum()
        net = (s - s.shift(n)).abs()
        return net / diff
    return f

signals = {
    "er_20d": per_asset(panel, er_factory(20)),
    "er_60d": per_asset(panel, er_factory(60)),
}

# ---- 3) market beta 60d (EW-15 index return as market proxy) ----
rets = panel.pct_change()
ew_ret = rets.mean(axis=1, skipna=True)  # EW index return on union calendar
def mkt_beta_factory(window=60, minp=30):
    def f(s):
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), ew_ret.reindex(ar.index).rename("m")], axis=1).dropna()
        cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
        var = df["m"].rolling(window, min_periods=minp).var()
        return (cov / var).reindex(s.index)
    return f
signals["mkt_beta_60d"] = per_asset(panel, mkt_beta_factory(60))

# ---- 4) 20d rolling correlation with EW index ----
def corr_ew_factory(window=20, minp=10):
    def f(s):
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), ew_ret.reindex(ar.index).rename("m")], axis=1).dropna()
        c = df["a"].rolling(window, min_periods=minp).corr(df["m"])
        return c.reindex(s.index)
    return f
signals["corr_ew_20d"] = per_asset(panel, corr_ew_factory(20))

# ---- 5) gap vol ratio: cc-vol / parkinson-vol (20d) ----
def gap_vol_ratio(s_close, s_high, s_low, w=20):
    ar = s_close.pct_change()
    cc = ar.rolling(w).std()
    pk = np.log(s_high / s_low).rolling(w).mean()  # mean log range (proxy for intraday vol)
    return cc / pk
def gvr_apply(a):
    return gap_vol_ratio(panel[a].dropna(), high_p[a].dropna(), low_p[a].dropna())
frames = {a: gvr_apply(a).reindex(panel.index) for a in panel.columns}
signals["gap_vol_ratio20"] = pd.DataFrame(frames, index=panel.index)

# ---- 6) intraday momentum 20d: mean(close/open - 1) ----
def intraday_mom_apply(a):
    s = (panel[a] / open_p[a] - 1.0).dropna()
    return s.rolling(20).mean().reindex(panel.index)
frames = {a: intraday_mom_apply(a) for a in panel.columns}
signals["intraday_mom_20"] = pd.DataFrame(frames, index=panel.index)

# ---- 7) overnight gap 5d: mean(open/prev_close - 1) ----
def overnight_gap_apply(a, w=5):
    s = panel[a].dropna()
    o = open_p[a].dropna()
    gap = (o / s.shift(1) - 1.0).dropna()
    return gap.rolling(w).mean().reindex(panel.index)
frames = {a: overnight_gap_apply(a, 5) for a in panel.columns}
signals["overnight_gap_5"] = pd.DataFrame(frames, index=panel.index)

# ---- 8) drawdown depth 60d ----
signals["dd_60d"] = per_asset(panel, lambda s: s / s.rolling(60).max() - 1.0)

# ---- 9) ER acceleration ----
signals["er_ratio_20x60"] = signals["er_20d"] - signals["er_60d"]

for fid, sig in signals.items():
    print(f"    {fid}: nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# ---- library: persisted signal artifacts (the real gate input) ----
library = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "carry_3m1m", "carry_12m3m",
            "mom_curve_volscale", "range_pos_120d", "vol_surge_20"]:
    try:
        arr = np.load(f"factors/{fid}.signal.npy")
        library[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
    except FileNotFoundError:
        print(f"  [warn] missing artifact factors/{fid}.signal.npy")

print(f"library artifacts loaded: {list(library.keys())}")

# ---- validation ----
print("\n=== validation (admission h=10) ===")
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5
           and m.get("max_abs_library_correlation", 1.0) < 0.5]
print(f"\nPASSERS (gate + robustness + corr gate): {passers}")

# ---- regime breakdown for passers ----
print("\n=== regime breakdown for passers (10d IC / ICIR / n) ===")
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

# ---- re-validation of ACTIVE library factors (drift monitoring) ----
print("\n=== re-validation of ACTIVE library factors ===")
reval = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20"]:
    sig = library[fid]
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=None, fwd_cache=fwd_cache)
    reval[fid] = m
    report(fid + " [active]", m)
    parts = [fid]
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            parts.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
    print("  " + " | ".join(parts))

# ---- pairwise rho for passers vs active factors ----
print("\n=== pairwise rho (passers vs active artifacts) ===")
for fid in passers:
    for lid in ["mom20_volproxy60", "dxy_beta_cond_60x20"]:
        print(f"  {fid:20s} vs {lid:20s} = {panel_rank_corr(signals[fid], library[lid]):+.4f}")

json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                       for k, v in results.items()},
           "passers": passers, "regime": regime_out, "reval_active": reval},
          open("scripts/_miner1_cycle27_pathstruct_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle27_pathstruct_results.json")
print("DONE")
