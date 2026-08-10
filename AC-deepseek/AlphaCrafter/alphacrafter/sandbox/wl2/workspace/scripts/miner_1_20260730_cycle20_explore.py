"""miner_1 2026-07-30 cycle 20: explore & validate novel cross-asset factors.

Lesson from cycle 19: the post-miner gate recovers the signal matrix from a
`signal_artifact` STRING PATH (e.g. "fid.signal.npy") relative to factors/,
NOT from an embedded dict.  All admitted factors must share the exact same
matrix shape (same date grid & column order) so pairwise rho is computable.

This script:
  1. Loads the 15-asset close panel (union calendar, visible through 2026-07-29).
  2. Evaluates ~22 candidates spanning momentum, trend-location, volatility,
     cross-asset beta, reversal, efficiency, skew, and conditional macro-beta.
  3. Validates each: |IC|>=0.007, |ICIR|>=0.084 (h=10 admission), decay,
     coverage, turnover, regime splits.
  4. Computes gate-style pairwise abs Spearman among PASSERS (rank within row).
  5. Saves detailed results for the persistence step.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor,
                         panel_rank_corr, coverage_stats, turnover_rank,
                         report)

panel = load_panel()
print(f"panel shape: {panel.shape}  dates: {panel.index.min().date()}..{panel.index.max().date()}")
print(f"assets: {list(panel.columns)}")
close = panel
RET = close.pct_change()

# ---------------------------------------------------------------------------
# Candidate signal construction (all close-panel expressions; NaN preserved)
# ---------------------------------------------------------------------------
signals = {}

# --- momentum / trend family ---
signals["mom_10d_skip5"]  = per_asset(close, lambda s: s.shift(5) / s.shift(15) - 1.0)
signals["mom_20d_skip5"]  = per_asset(close, lambda s: s.shift(5) / s.shift(25) - 1.0)
signals["mom_60d_skip5"]  = per_asset(close, lambda s: s.shift(5) / s.shift(65) - 1.0)
signals["mom_120d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(125) - 1.0)
signals["mom_180d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(185) - 1.0)
signals["trend_20x60"]    = close.rolling(20).mean() / close.rolling(60).mean() - 1.0
signals["range_pos_252"]  = ((close - close.rolling(252, min_periods=30).min())
                             / (close.rolling(252, min_periods=30).max()
                                - close.rolling(252, min_periods=30).min()))
signals["dist_high_60d"]  = close / close.rolling(60).max() - 1.0

# --- volatility family ---
signals["vol_of_vol20x60"] = per_asset(close,
    lambda s: s.pct_change().rolling(20).std().rolling(60).std())
signals["vol_ratio_5x20"]  = RET.rolling(5, min_periods=3).std() / RET.rolling(20, min_periods=10).std()
signals["downside_vol_ratio_20"] = (RET.where(RET < 0, 0.0).rolling(20, min_periods=10).std()
                                    / RET.rolling(20, min_periods=10).std())
signals["vol_zscore_20x252"] = per_asset(close,
    lambda s: (s.pct_change().rolling(20).std()
               - s.pct_change().rolling(20).std().rolling(252).mean())
              / s.pct_change().rolling(20).std().rolling(252).std())

# --- cross-asset beta / correlation family ---
signals["spx_corr60"] = RET.rolling(60, min_periods=15).corr(close["SPX"].pct_change())
signals["btc_corr60"] = RET.rolling(60, min_periods=15).corr(close["BTC"].pct_change()).assign(BTC=np.nan)
signals["xau_corr60"] = RET.rolling(60, min_periods=15).corr(close["XAU"].pct_change()).assign(XAU=np.nan)
signals["wti_corr60"] = RET.rolling(60, min_periods=15).corr(close["WTI"].pct_change()).assign(WTI=np.nan)
signals["us10y_corr60"] = RET.rolling(60, min_periods=15).corr(close["US10Y"].pct_change()).assign(US10Y=np.nan)

# --- conditional macro beta (VIX) ---
vix_close = macro_series("VIX")
vix_ret = vix_close.pct_change()
vix_20 = vix_close / vix_close.shift(20) - 1.0
_beta_vix = {}
for a in close.columns:
    s = close[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), vix_ret.reindex(ar.index).rename("v")], axis=1).dropna()
    _beta_vix[a] = df["a"].rolling(60, min_periods=15).cov(df["v"]) / df["v"].rolling(60, min_periods=15).var()
beta_vix = pd.DataFrame(_beta_vix, index=panel.index)
signals["vix_beta_cond_60x20"] = -beta_vix * vix_20.reindex(beta_vix.index)

# --- reversal / mean-reversion family ---
signals["zscore_20_rev"] = -1.0 * (close - close.rolling(20).mean()) / close.rolling(20).std()
signals["bollinger_pos_20"] = (close - close.rolling(20).mean()) / (2.0 * close.rolling(20).std())
signals["rsi_14"] = per_asset(close, lambda s: 100 - 100 / (1 + s.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                              / (-s.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()))

# --- efficiency / flow family ---
signals["eff_ratio_20"] = (close - close.shift(20)).abs() / close.diff().abs().rolling(20, min_periods=10).sum()
signals["upday_ratio_20"] = (RET > 0).rolling(20, min_periods=10).mean()
signals["gain_loss_20"] = (RET.clip(lower=0).rolling(20, min_periods=10).mean()
                           / (RET.clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9))

# --- shape / auto-correlation family ---
signals["skew_20d"] = RET.rolling(20, min_periods=10).skew()
signals["kurt_20d"] = RET.rolling(20, min_periods=10).kurt()
signals["autocorr_60x5"] = RET.rolling(60, min_periods=15).corr(RET.shift(5))

# --- vol-adjusted momentum ---
signals["mom30_vol60"] = per_asset(close,
    lambda s: (s.shift(5) / s.shift(35) - 1.0) / s.pct_change().rolling(60).std())
signals["mom60_vol20"] = per_asset(close,
    lambda s: (s.shift(5) / s.shift(65) - 1.0) / s.pct_change().rolling(20).std())

# align every signal to the exact panel grid
for fid in list(signals.keys()):
    sig = signals[fid]
    if isinstance(sig, pd.Series):
        sig = sig.to_frame()
    sig = sig.reindex(index=panel.index, columns=panel.columns)
    signals[fid] = sig
    print(f"  {fid:22s} shape={sig.shape} nan={int(sig.isna().sum().sum())}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}
ret_a = fwd_cache[str(ADM_H)]

results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084]
print(f"\nPASSERS (raw gate): {passers}  ({len(passers)}/{len(results)})")

# ---------------------------------------------------------------------------
# Gate-style pairwise abs Spearman among passers (rank within row, mean over dates)
# ---------------------------------------------------------------------------
print("\n=== pairwise |rho| among PASSERS (gate style, >=0.5 = redundant) ===")
names = passers
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        r = abs(panel_rank_corr(signals[a], signals[b]))
        rho.loc[a, b] = r
        rho.loc[b, a] = r
if names:
    print("        " + "".join(f"{b[:8]:>10s}" for b in names))
    for i, a in enumerate(names):
        print(f"  {a:20s}" + "".join(f"{rho.loc[a,b]:>10.3f}" if pd.notna(rho.loc[a, b]) else f"{'-':>10s}" for b in names))

# quality ranking and redundancy check
print("\n=== passers by quality |IC|*|ICIR| ===")
qual = {fid: abs(results[fid]["ic"]) * abs(results[fid]["icir"]) for fid in passers}
for fid in sorted(passers, key=lambda f: -qual[f]):
    conflicts = [b for b in passers if b != fid and pd.notna(rho.loc[fid, b]) and rho.loc[fid, b] >= 0.5]
    print(f"  {fid:20s} q={qual[fid]:.6f} ic={results[fid]['ic']:+.4f} icir={results[fid]['icir']:+.4f} "
          f"hit={results[fid]['ic_hit_ratio']:.3f} n={results[fid]['n_ic_dates']} "
          f"cov={results[fid]['coverage_asset_days']:.3f} turn={results[fid]['turnover_10d_rank']} "
          f"redundant_vs={conflicts if conflicts else '-'}")

# ---------------------------------------------------------------------------
# Regime breakdown for passers
# ---------------------------------------------------------------------------
print("\n=== regime breakdown for passers (IC10 / ICIR10 / n) ===")
regimes = [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
           ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]
regime_out = {}
for fid in passers:
    sig = signals[fid]
    line = [fid]
    rd = {}
    for r0, r1 in regimes:
        sub_mask = (panel.index >= r0) & (panel.index <= r1)
        sig_s = sig.loc[sub_mask]
        ic_ser = compute_ic(sig_s, ret_a.loc[sub_mask]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            line.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4), "icir": round(float(icir), 4),
                          "n": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(line))

out = {"panel_shape": list(panel.shape), "visible_through": "2026-07-29",
       "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
       "passers": passers, "quality": qual,
       "pairwise_rho_passers": {a: {b: round(float(rho.loc[a, b]), 4) for b in names
                                    if pd.notna(rho.loc[a, b])} for a in names},
       "regime": regime_out}
json.dump(out, open("scripts/_miner1_cycle20_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle20_results.json")
