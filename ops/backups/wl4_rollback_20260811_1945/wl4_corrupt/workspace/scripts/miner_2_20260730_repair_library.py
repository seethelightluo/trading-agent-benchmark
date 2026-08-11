"""miner_2 (sim date 2026-07-30): rebuild factor library with recoverable .npy
signal artifacts + explore new close-only cross-asset factors.  Vectorized IC.

Context: all previously persisted factors were quarantined because their signal
artifact used panel_json_v1 (dict) which the deterministic gate's
_load_signal_artifact() does NOT recognize. This script re-persists every
gate-passing factor with a real 2D .npy matrix on a shared canonical grid, then
runs enforce_library() (admission + pairwise rho conflict + capacity), and
rebuilds factor_ensemble.json from the kept library.

Validation window: 2020-01-01 .. 2026-07-15 (warm-up), h=10, min_valid=8 assets/date.
Admission gates (15-asset universe): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank,
                                 library_signals, max_library_corr, TRADABLE)
from alphacrafter.factor_contract import FactorContract, enforce_library, validate_ensemble

VAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2026-07-15")
H = 10
MIN_VALID = 8
CONTRACT = FactorContract()

# ---------------- data ----------------
panels = load_panels(3000)
closes_full = close_panel(panels)                      # full history -> 2026-07-29
grid = closes_full.index[(closes_full.index >= VAL_START) & (closes_full.index <= VAL_END)]
closes = closes_full.reindex(grid)
rets = closes.pct_change()
print(f"data: {len(TRADABLE)} tradable assets, grid {grid.min().date()}..{grid.max().date()} "
      f"({len(grid)} dates, warm-up window)", flush=True)

vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
cn10 = panels["CN10Y"]["close"].astype(float)
eur = panels["EURUSD"]["close"].astype(float)

# ---------------- factor builders ----------------
def rolling_beta(asset_ret, driver, win=60, min_obs=40):
    driver = driver.astype(float)
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver.rename("m")], axis=1).dropna()
        if len(z) < min_obs + 5:
            beta[a] = pd.Series(np.nan, index=asset_ret.index)
            continue
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b.reindex(asset_ret.index)
    return pd.DataFrame(beta, index=asset_ret.index)

def per_asset(func):
    out = {}
    for a in TRADABLE:
        s = closes_full[a].dropna()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=grid)
            continue
        try:
            out[a] = func(s).reindex(grid)
        except Exception:
            out[a] = pd.Series(np.nan, index=grid)
    return pd.DataFrame(out, index=grid)

C = {}
# --- library incumbents (stored definitions) ---
C["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)
C["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)
C["vol_of_vol20x60"] = per_asset(lambda s: s.pct_change().rolling(20).std().rolling(60).std())
if vix is not None:
    vix_ret = vix.pct_change()
    C["vix_beta_cond_60x20"] = (-rolling_beta(rets, vix_ret, 60)
                                * (vix / vix.shift(20) - 1.0).reindex(grid))
# --- macro beta factors (batch-3a) ---
C["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10.pct_change(), 60)
C["eurusd_beta_60d"] = rolling_beta(rets, eur.pct_change(), 60)
mkt = rets.mean(axis=1)
dn_mkt = mkt.where(mkt < 0).fillna(0.0)
C["dn_mkt_beta_60d"] = rolling_beta(rets, dn_mkt, 60)
# --- batch-3 passers ---
C["down_up_vol_ratio_20"] = per_asset(
    lambda s: s.pct_change().clip(upper=0).rolling(20).std()
    / (s.pct_change().clip(lower=0).rolling(20).std() + 1e-12))
mom20 = per_asset(lambda s: s / s.shift(20) - 1.0)
C["rel_mom_20d"] = mom20.sub(mom20.mean(axis=1), axis=0)
btc = closes_full["BTC"].dropna()
btc_ret = btc.pct_change()
btc_mom = btc_ret.rolling(20).mean() * 20.0
C["crypto_beta_btc_60x20"] = rolling_beta(rets, btc_ret, 60) * btc_mom.reindex(grid)
# --- NEW close-only explorations ---
C["skew_60d"] = per_asset(lambda s: s.pct_change().rolling(60).skew())
C["vol_term_20_60"] = per_asset(
    lambda s: s.pct_change().rolling(20).std() / s.pct_change().rolling(60).std() - 1.0)
C["range_pos_20d"] = per_asset(
    lambda s: (s - s.rolling(20).min()) / (s.rolling(20).max() - s.rolling(20).min()) - 0.5)
C["dd_60d"] = per_asset(lambda s: s / s.rolling(60).max() - 1.0)

lib = library_signals(panels, closes, rets)

# ---------------- vectorized validation ----------------
def fast_ic_series(factor_panel, fwd, min_valid=MIN_VALID):
    """Daily cross-sectional Spearman IC via rank-then-Pearson (fully vectorized)."""
    f = factor_panel.rank(axis=1)
    r = fwd.rank(axis=1)
    common = f.index.intersection(r.index)
    f = f.loc[common]
    r = r.loc[common]
    valid = f.notna() & r.notna()
    n = valid.sum(axis=1)
    fv = f.where(valid)
    rv = r.where(valid)
    fm = fv.sub(fv.mean(axis=1), axis=0)
    rm = rv.sub(rv.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1, min_count=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = num / den
    ic = ic[(n >= min_valid) & (den > 1e-12)]
    return ic.dropna()

def validate(name, panel):
    panel = panel.reindex(grid)
    fwd10 = forward_returns(closes, H)
    ics = fast_ic_series(panel, fwd10)
    ics = ics[(ics.index >= VAL_START) & (ics.index <= VAL_END)]
    if len(ics) < 200:
        return None
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 0 else 0.0
    hit = float((np.sign(ics) == np.sign(ic)).mean())
    cov = coverage_metrics(panel)
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        s = fast_ic_series(panel, forward_returns(closes, h))
        s = s[(s.index >= VAL_START) & (s.index <= VAL_END)]
        if len(s) >= 100:
            decay[str(h)] = round(float(s.mean()), 4)
    corr, key = max_library_corr(panel, lib)
    return {"ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
            "n_ic_dates": int(len(ics)), "ic_std": round(float(ics.std(ddof=1)), 4),
            "coverage_asset_days": cov["coverage_asset_days"],
            "coverage_dates_ge8": cov["coverage_dates_ge8"],
            "turnover_10d_rank": turnover_rank(panel, 10),
            "decay_ic_by_horizon": decay,
            "max_abs_library_correlation": corr, "max_corr_factor": key}

results = {}
print("\n--- validation (h=10, warm-up window) ---", flush=True)
for name, panel in C.items():
    m = validate(name, panel)
    if m is None:
        print(f"{name:26s} NOTE insufficient IC dates", flush=True)
        continue
    results[name] = m
    ok = abs(m["ic"]) >= CONTRACT.ic_threshold and abs(m["icir"]) >= CONTRACT.icir_threshold
    print(f"{name:26s} IC={m['ic']:>8.4f} ICIR={m['icir']:>8.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} "
          f"covD8={m['coverage_dates_ge8']:.3f} to={m['turnover_10d_rank']:.3f} "
          f"rho={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']}) "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

# ---------------- persistence (JSON + .npy artifact) ----------------
META = {
    "mom_10d_skip5": ("Short Momentum 10d (skip 5d)", "close.shift(5) / close.shift(15) - 1.0",
                      "10-day price momentum with 5-day skip to avoid short-term reversal.",
                      ["close"], {"lookback": 10, "skip": 5}, ["momentum", "cross-asset"]),
    "mom_120d_skip5": ("Long Momentum 120d (skip 5d)", "close.shift(5) / close.shift(125) - 1.0",
                       "120-day price momentum with 5-day skip (trend persistence).",
                       ["close"], {"lookback": 120, "skip": 5}, ["momentum", "cross-asset"]),
    "vol_of_vol20x60": ("Vol-of-Vol 20x60", "rolling_std(pct_change(close),20).rolling(60).std()",
                        "Volatility of 20-day realized volatility over 60 days (vol regime instability).",
                        ["close"], {"vol_win": 20, "smooth_win": 60}, ["volatility", "cross-asset"]),
    "vix_beta_cond_60x20": ("VIX-beta conditional 60x20",
                            "-beta(asset_ret, pct_change(VIX), 60) * (VIX/VIX.shift(20)-1)",
                            "Negative VIX-beta times 20d VIX change: assets that hedge VIX spikes are favored when VIX rises.",
                            ["close", "VIX"], {"beta_win": 60, "cond_win": 20}, ["macro-beta", "volatility"]),
    "rate_beta_cn10y_60d": ("CN10Y rate beta 60d", "beta(asset_ret, pct_change(CN10Y), 60)",
                            "Rolling 60d beta of each asset's daily return on CN10Y yield changes.",
                            ["close", "CN10Y"], {"beta_win": 60, "min_obs": 40}, ["macro-beta", "rates"]),
    "eurusd_beta_60d": ("EURUSD beta 60d", "beta(asset_ret, pct_change(EURUSD), 60)",
                        "Rolling 60d beta on EURUSD changes (global risk-appetite proxy).",
                        ["close", "EURUSD"], {"beta_win": 60, "min_obs": 40}, ["macro-beta", "fx"]),
    "dn_mkt_beta_60d": ("Downside market beta 60d", "beta(asset_ret, min(0, mean(asset_ret,axis=1)), 60)",
                        "Rolling 60d beta on downside-only cross-asset market return (crash sensitivity).",
                        ["close"], {"beta_win": 60, "min_obs": 40}, ["macro-beta", "downside"]),
    "down_up_vol_ratio_20": ("Downside/upside vol ratio 20d",
                             "rolling_std(clip(r,upper=0),20) / rolling_std(clip(r,lower=0),20)",
                             "Ratio of downside to upside realized volatility over 20 days (asymmetric risk).",
                             ["close"], {"win": 20}, ["volatility", "asymmetry"]),
    "rel_mom_20d": ("Relative momentum 20d", "mom20 - cross_sectional_mean(mom20)",
                    "Cross-sectionally demeaned 20d momentum (relative strength vs peers).",
                    ["close"], {"lookback": 20}, ["momentum", "relative"]),
    "crypto_beta_btc_60x20": ("BTC-beta conditional 60x20",
                              "beta(asset_ret, pct_change(BTC), 60) * mean(pct_change(BTC),20)*20",
                              "BTC-beta times 20d BTC momentum (crypto-driven risk regime tilt).",
                              ["close", "BTC"], {"beta_win": 60, "cond_win": 20}, ["macro-beta", "crypto"]),
    "skew_60d": ("Return skewness 60d", "rolling_skew(pct_change(close), 60)",
                 "60-day skewness of daily returns (tail-risk asymmetry).",
                 ["close"], {"win": 60}, ["volatility", "tail"]),
    "vol_term_20_60": ("Vol term structure 20/60", "vol20/vol60 - 1",
                       "Short vs long realized vol ratio (vol regime slope).",
                       ["close"], {"short_win": 20, "long_win": 60}, ["volatility"]),
    "range_pos_20d": ("20d range position", "(close-min20)/(max20-min20) - 0.5",
                      "Close location inside the 20d high-low range (trend vs mean-reversion regime).",
                      ["close"], {"win": 20}, ["trend", "location"]),
    "dd_60d": ("60d drawdown depth", "close/rolling_max(close,60) - 1",
               "Distance from 60-day high (recovery/drawdown state).",
               ["close"], {"win": 60}, ["trend", "drawdown"]),
}

print("\n--- persistence ---", flush=True)
for fid, m in results.items():
    ok = abs(m["ic"]) >= CONTRACT.ic_threshold and abs(m["icir"]) >= CONTRACT.icir_threshold
    if not ok:
        continue
    fname, expr, desc, deps, params, tags = META[fid]
    exp_dir = int(np.sign(m["ic"])) if m["ic"] else 1
    panel = C[fid].reindex(grid)
    arr = panel[TRADABLE].values.astype(float)
    npy_path = Path("factors") / f"{fid}_signal.npy"
    np.save(npy_path, arr)
    payload = {
        "factor_id": fid, "factor_name": fname, "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": deps, "parameters": params, "expected_direction": exp_dir,
        "signal_artifact": npy_path.name, "signal_artifact_format": "npy",
        "signal_artifact_shape": list(arr.shape),
        "signal_artifact_grid": {"start": str(grid.min().date()), "end": str(grid.max().date()),
                                 "n_dates": int(len(grid)), "columns": TRADABLE,
                                 "note": "canonical grid shared by all library factors"},
        "validation": {"status": "EFFECTIVE",
                       "period": f"{VAL_START.date()}..{VAL_END.date()}",
                       "last_validated": "2026-07-30", "admission_horizon": H,
                       "regime_notes": ("Validated 2020-01-01..2026-07-15 (warm-up) across regimes: "
                                        "COVID crash 2020, 2020-21 bull, 2022 tightening bear, "
                                        "2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. "
                                        "Cross-sectional Spearman rank IC on the 15-asset tradable "
                                        "universe, h=10, min 8 valid instruments per date."),
                       "metrics": {**m, "signal_artifact": npy_path.name}},
        "tags": tags,
    }
    out = Path("factors") / f"{fid}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"WROTE factors/{fid}.json + {npy_path.name} | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
          f"dir={exp_dir:+d} decay10={m['decay_ic_by_horizon']['10']}", flush=True)

# ---------------- deterministic gate ----------------
print("\n--- enforce_library ---", flush=True)
result = enforce_library(Path("factors"), CONTRACT)
print(json.dumps(result, indent=2, default=str), flush=True)

with open("factor_library_audit.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"cycle": 3, **result}) + "\n")

# ---------------- ensemble ----------------
print("\n--- ensemble rebuild ---", flush=True)
kept = []
for name in result["kept_files"]:
    p = json.loads(Path("factors", name).read_text(encoding="utf-8"))
    sel = p["benchmark_admission"]["selected_metrics"]
    kept.append((name, p["factor_id"], abs(sel["ic"]) * abs(sel["icir"]),
                 p.get("expected_direction", 1)))
total_q = sum(q for _, _, q, _ in kept) or 1.0
selected = [{"factor_id": fid, "weight": round(q / total_q, 4), "direction": int(np.sign(d)) if d else 1}
            for _, fid, q, d in kept]
ensemble = {"schema_version": 1, "selected_factors": selected, "method": "quality_ic_tilt"}
Path("factors", "factor_ensemble.json").write_text(json.dumps(ensemble, indent=2), encoding="utf-8")
print(json.dumps(ensemble, indent=2), flush=True)
print("ensemble validation:", validate_ensemble("factors/factor_ensemble.json", "factors", CONTRACT), flush=True)

# ---------------- round-trip verification ----------------
print("\n--- artifact round-trip ---", flush=True)
ok_all = True
for name in result["kept_files"]:
    p = json.loads(Path("factors", name).read_text(encoding="utf-8"))
    art = np.load(Path("factors") / p["signal_artifact"], allow_pickle=False)
    shape_ok = art.shape == (len(grid), len(TRADABLE))
    ok_all = ok_all and shape_ok
    print(f"{name:28s} artifact={p['signal_artifact']} shape={art.shape} ok={shape_ok}", flush=True)
print("ALL ARTIFACTS OK" if ok_all else "ARTIFACT MISMATCH", flush=True)
