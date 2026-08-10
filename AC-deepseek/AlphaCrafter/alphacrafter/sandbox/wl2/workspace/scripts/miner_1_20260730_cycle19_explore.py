"""miner_1 2026-07-30 cycle 19: explore novel close-only factors + re-validate library.

Universe: 15 tradable cross-asset instruments. Data visible through 2026-07-29.
Gate: |IC10| >= 0.007 and |ICIR10| >= 0.084 (benchmark-wide cross-asset gates).
All candidates are pure `close`-panel expressions (recoverable signal artifact
for the deterministic post-miner gate under restricted namespace close/pd/np).
Library correlation is computed against the currently EFFECTIVE persisted
factors (mom_180d_skip5, mom_20d_skip5, range_pos_252, spx_corr60,
vol_of_vol20x60) reconstructed from their stored signal artifacts.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor,
                         load_library_signals, report, panel_rank_corr,
                         coverage_stats)

panel = load_panel()
close = panel
print(f"panel shape: {panel.shape}  date range: {panel.index.min().date()}..{panel.index.max().date()}")
print(f"assets: {len(panel.columns)}")

# ---------------------------------------------------------------------------
# Reconstruct effective library from persisted factor signal artifacts
# ---------------------------------------------------------------------------
EFFECTIVE_FILES = ["mom_180d_skip5", "mom_20d_skip5", "range_pos_252",
                   "spx_corr60", "vol_of_vol20x60"]


def artifact_to_panel(fid):
    d = json.load(open(f"factors/{fid}.json"))
    sa = d["signal_artifact"]
    dates = pd.to_datetime(sa["dates"])
    cols = sa["columns"]
    vals = np.asarray(sa["values"], dtype=float)
    return pd.DataFrame(vals.reshape(len(dates), len(cols)),
                        index=dates, columns=cols), d


library = {}
lib_meta = {}
for fid in EFFECTIVE_FILES:
    sig, meta = artifact_to_panel(fid)
    library[fid] = sig.reindex(panel.index)
    lib_meta[fid] = meta
print("effective library signals:", list(library.keys()))

HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------------------------------------------------------------------
# Candidate factors (each is a pure close-panel expression)
# ---------------------------------------------------------------------------
EXPRS = {
    "rsi_14": ("100 - 100/(1 + close.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean()"
               " / (-close.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean())"),
    "zscore_20_rev": "-1.0 * (close - close.rolling(20).mean()) / close.rolling(20).std()",
    "eff_ratio_20": "(close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum()",
    "bollinger_pos_20": "(close - close.rolling(20).mean()) / (2.0 * close.rolling(20).std())",
    "rate_beta_cond_60x20": ("close.pct_change().rolling(60).cov(close['US10Y'].pct_change())"
                             ".div(close['US10Y'].pct_change().rolling(60).var(), axis=0)"
                             ".mul(close['US10Y']/close['US10Y'].shift(20) - 1.0, axis=0)"),
    "drawup_20d": "close / close.rolling(20).min() - 1.0",
    "dist_high_20d": "close / close.rolling(20).max() - 1.0",
    "mom60_vol20": "(close.shift(5)/close.shift(65) - 1.0) / close.pct_change().rolling(20).std()",
    "skew_20d": "close.pct_change().rolling(20).skew()",
    "autocorr_60x5": "close.pct_change().rolling(60).corr(close.pct_change().shift(5))",
    "win_days_ratio_20": "(close.diff() > 0).rolling(20).mean()",
    "trend_20x60": "close.rolling(20).mean() / close.rolling(60).mean() - 1.0",
    "vol_ratio_5x20": "close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std()",
    "downside_vol_ratio_20": ("close.pct_change().where(close.pct_change() < 0, 0.0)"
                              ".rolling(20).std() / close.pct_change().rolling(20).std()"),
    "mom20_vol60": "(close.shift(5)/close.shift(25) - 1.0) / close.pct_change().rolling(60).std()",
}

signals = {}
print("\n=== eval candidates (close/pd/np namespace) ===")
for fid, exp in EXPRS.items():
    env = {"close": close, "pd": pd, "np": np}
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
    except Exception as e:
        print(f"  {fid:24s} EVAL FAIL: {e}")
        continue
    if isinstance(sig, pd.Series):
        sig = sig.to_frame()
    if not (isinstance(sig, pd.DataFrame) and sig.shape == close.shape):
        print(f"  {fid:24s} BAD SHAPE {getattr(sig, 'shape', None)}")
        continue
    signals[fid] = sig.reindex(panel.index)
    print(f"  {fid:24s} OK  nan={sig.isna().sum().sum()}")

# ---------------------------------------------------------------------------
# Validate candidates
# ---------------------------------------------------------------------------
print("\n=== candidate validation (admission h=10) ===")
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084]
print(f"\nPASSERS (raw gate): {passers}")

# ---------------------------------------------------------------------------
# Re-validation of current effective library factors (drift tracking)
# ---------------------------------------------------------------------------
print("\n=== effective library re-validation (full window, from artifacts) ===")
lib_metrics = {}
for fid, sig in library.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        fwd_cache=fwd_cache)
    lib_metrics[fid] = m
    print(f"[LIB:{fid}] IC10={m['ic']:+.4f} ICIR10={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} decay10={m['decay_ic_by_horizon']['10']:+.4f}")

print("\n=== effective library recent-window drift (2025-01-01..2026-07-29) ===")
panel_recent = panel.loc[panel.index >= "2025-01-01"]
fwd_recent = {str(h): forward_returns(panel_recent, h) for h in HORIZONS}
for fid, sig in library.items():
    sig_r = sig.reindex(panel_recent.index)
    ic_ser = compute_ic(sig_r, fwd_recent[str(ADM_H)]).dropna()
    ic = float(ic_ser.mean())
    icir = float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 and ic_ser.std() > 0 else 0.0
    print(f"[LIB:{fid}] recent IC10={ic:+.4f} ICIR10={icir:+.4f} n={len(ic_ser)} "
          f"full_IC10={lib_metrics[fid]['ic']:+.4f}")

# ---------------------------------------------------------------------------
# Regime breakdown for passers
# ---------------------------------------------------------------------------
if passers:
    print("\n=== regime breakdown for passers ===")
    regimes = [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
               ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]
    for fid in passers:
        sig = signals[fid]
        line = [fid]
        for r0, r1 in regimes:
            sub = panel.loc[(panel.index >= r0) & (panel.index <= r1)]
            sig_s = sig.reindex(sub.index)
            ic_ser = compute_ic(sig_s, fwd_cache[str(ADM_H)].reindex(sub.index)).dropna()
            if len(ic_ser) > 0:
                sd = ic_ser.std()
                line.append(f"{r0[:4]}-{r1[:4]}: ic={ic_ser.mean():+.4f} icir="
                            f"{ic_ser.mean()/sd if sd>0 else 0:+.3f} n={len(ic_ser)}")
        print("  " + " | ".join(line))

out = {"candidates": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                      for k, v in results.items()},
       "passers": passers, "library_revalidation": lib_metrics}
json.dump(out, open("scripts/_miner1_cycle19_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle19_results.json")
