"""miner_1 cycle 2026-07-30: deep validation of batch-7 winners.

gain_loss_asym_20 and beta_ndx_60 passed the h=10 admission gate. Here we:
  1) recompute panels + full IC summary (with regime splits),
  2) compute max |spearman rho| against ALL 12 effective library factor
     signal artifacts on the canonical grid,
  3) persist each passing factor JSON + .npy artifact.
"""
import sys
sys.path.insert(0, 'scripts')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, save_signal_artifact, WATCHLIST,
                           VAL_START, VAL_END, rank_ic_series, forward_returns)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

LIB_IDS = ["dd_duration_120_resid", "down_beta_60", "dxy_beta_cond_60x20",
           "eurusd_beta_cond_60x20", "hilo_pos_60", "hs300_beta_60",
           "max_ret_20d", "skew_term_20_60", "spx_beta_60",
           "vix_beta_cond_60x20", "vol_adj_mom_20_60", "vol_of_vol20x60"]


def lib_corr(panel):
    """max |daily mean cross-sectional Spearman rho| vs library signal artifacts."""
    best, best_id = 0.0, None
    pm = panel.reindex(grid)
    for fid in LIB_IDS:
        art_path = Path('factors') / f'{fid}_signal.npy'
        if not art_path.exists():
            print(f"  [missing artifact] {fid}")
            continue
        arr = np.load(art_path, allow_pickle=False)
        lib = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
        common = pm.index.intersection(lib.index)
        corrs = []
        for d in common:
            x = pm.loc[d]; y = lib.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if len(corrs) > 30:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def regime_ic(panel):
    fwd = forward_returns(prices, 10)
    ic = rank_ic_series(panel, fwd)
    ic = ic[(ic.index >= VAL_START) & (ic.index <= VAL_END)]
    out = {}
    for name, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"),
                         ("2023-2024", "2023-01-01", "2024-12-31"),
                         ("2025-2026H1", "2025-01-01", "2026-07-15")]:
        sub = ic[(ic.index >= pd.Timestamp(lo)) & (ic.index <= pd.Timestamp(hi))]
        if len(sub) > 30:
            m = float(sub.mean()); s = float(sub.std(ddof=1))
            out[name] = {"n": int(len(sub)), "ic": round(m, 4),
                         "icir": round(m / s if s > 0 else 0.0, 4)}
    return out


def f_gain_loss_asym_20(df, s):
    r = df['close'].pct_change()
    pos = r.clip(lower=0).rolling(20).mean()
    neg = r.clip(upper=0).rolling(20).mean().abs()
    return pos / neg.replace(0, np.nan)


def f_beta_ndx_60(df, s):
    ndx = prices['NDX']['close']
    r = df['close'].pct_change()
    rn = ndx.pct_change()
    z = pd.concat([r.rename('r'), rn.rename('n')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['n']) / z['n'].rolling(60).var()
    return b


CANDIDATES = {
    "gain_loss_asym_20": {
        "fn": f_gain_loss_asym_20,
        "name": "Gain/Loss Asymmetry (20d)",
        "expr": "mean(clip(r,0,None),20) / abs(mean(clip(r,None,0),20)), r=close.pct_change()",
        "desc": "Ratio of average up-day return to average down-day magnitude over 20 days. "
                "Assets whose recent daily gains are large relative to their losses (positive "
                "asymmetry) tended to earn higher forward 10d returns; a behavioral "
                "momentum/quality hybrid that captures directional skew of the return path.",
        "deps": ["close"],
        "params": {"window": 20, "admission_horizon": 10},
        "tags": ["asymmetry", "return-path", "quality", "cross-asset"],
    },
    "beta_ndx_60": {
        "fn": f_beta_ndx_60,
        "name": "NDX Beta (60d)",
        "expr": "rolling_cov(r_asset, r_ndx, 60) / rolling_var(r_ndx, 60)",
        "desc": "60-day rolling regression beta of each asset's daily returns on NDX (Nasdaq-100) "
                "returns. Captures technology/semiconductor beta exposure across the cross-asset "
                "universe; high NDX-beta assets (including crypto and semis) tended to earn higher "
                "forward 10d returns in the warm-up window.",
        "deps": ["close", "NDX close"],
        "params": {"window": 60, "admission_horizon": 10},
        "tags": ["beta", "technology", "systematic", "cross-asset"],
    },
}

for fid, spec in CANDIDATES.items():
    print(f"\n===== {fid} =====")
    panel = factor_to_panel(spec["fn"], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: validation returned None -> skip")
        continue
    rho, rho_id = lib_corr(panel)
    m["max_abs_library_correlation"] = round(rho, 4)
    m["max_corr_library_id"] = rho_id
    m["regime_ic_by_period"] = regime_ic(panel)
    ok = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    print(f"  IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f}")
    print(f"  decay10={m['decay_ic_by_horizon']['10']:+.4f} decay20={m['decay_ic_by_horizon']['20']:+.4f}")
    print(f"  max_lib_rho={rho:.4f} vs {rho_id}")
    print(f"  regime: {json.dumps(m['regime_ic_by_period'])}")
    if not ok:
        print(f"  {fid}: FAIL -> not persisted")
        continue
    if rho > 0.5:
        print(f"  {fid}: WARNING library rho {rho:.3f} > 0.5 correlation threshold (gate may evict)")
    # persist JSON + artifact
    art_path = Path('factors') / f'{fid}_signal.npy'
    arr = save_signal_artifact(panel, grid, art_path)
    payload = {
        "factor_id": fid,
        "factor_name": spec["name"],
        "version": "1.0.0",
        "calculation": {"expression": spec["expr"], "description": spec["desc"]},
        "dependencies": spec["deps"],
        "parameters": spec["params"],
        "expected_direction": "positive (IC>0)",
        "signal_artifact": art_path.name,
        "signal_artifact_format": "npy",
        "signal_artifact_shape": list(arr.shape),
        "signal_artifact_grid": {
            "start": str(grid.min().date()), "end": str(grid.max().date()),
            "n_dates": int(len(grid)), "columns": WATCHLIST,
            "note": "canonical grid shared by all library factors (see factor_common.canonical_grid)",
        },
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{VAL_START.date()}..{VAL_END.date()}",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": (f"Validated 2020-01..2026-07 warm-up across equities, commodities, "
                             f"crypto and rates; regime IC: {json.dumps(m['regime_ic_by_period'])}. "
                             f"Max library rho {rho:.3f} vs {rho_id}."),
            "metrics": {k: v for k, v in m.items() if k != "regime_ic_by_period"},
        },
        "tags": spec["tags"],
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5, "library_capacity": 30,
                         "active_top_k": 10},
            "selected_metrics": {
                "ic": m["ic"], "icir": m["icir"],
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": round(rho, 4),
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    Path('factors') / f'{fid}.json'
    out = Path('factors') / f'{fid}.json'
    out.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f"  PERSISTED -> {out}")
print("\ndone.")
