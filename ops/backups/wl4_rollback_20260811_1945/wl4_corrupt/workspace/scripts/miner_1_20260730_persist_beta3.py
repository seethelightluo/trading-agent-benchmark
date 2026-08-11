"""Persist batch-3a passing beta factors with recoverable signal artifacts (2026-07-30).

Each JSON embeds validation.signal_artifact = {format, dates, assets, values}
so the deterministic post-Miner gate can recompute pairwise rho from real
signal artifacts instead of assuming rho=0. A sidecar CSV is also written.
"""
import sys, json, base64, gzip, io
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)
HORIZONS = (1, 2, 3, 5, 10, 20)
H_ADM = 10


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


def build_factor(panel, fid, fname, expression, description, deps, params, exp_dir, tags):
    """Compute metrics, build JSON with artifact, write file + sidecar csv."""
    panel = panel.reindex(closes.index)
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, 8)
    m = summarize_ic(ics, exp_dir)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, 8, exp_dir)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key

    # --- signal artifact: dates + assets + per-asset value lists (rounded) ---
    p = panel.round(6)
    dates = [d.strftime("%Y-%m-%d") for d in p.index]
    assets = [a for a in p.columns if a in p]
    values = {a: [None if pd.isna(v) else float(v) for v in p[a].tolist()] for a in assets}
    artifact = {"format": "panel_json_v1", "n_dates": len(dates), "n_assets": len(assets),
                "dates": dates, "assets": assets, "values": values}

    rec = {
        "factor_id": fid,
        "factor_name": fname,
        "version": "1.0.0",
        "calculation": {"expression": expression, "description": description},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": exp_dir,
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{closes.index[0].date()}..{closes.index[-1].date()}",
            "last_validated": "2026-07-30",
            "admission_horizon": H_ADM,
            "regime_notes": ("Validated 2020-01-01..2026-07-29 across multiple regimes: COVID "
                             "crash 2020, recovery bull 2020-21, 2022 tightening bear, 2023-24 "
                             "AI-led equity rally, 2024-26 crypto/commodity cycles. Cross-sectional "
                             "rank IC on the 15-asset tradable universe, h=10."),
            "metrics": m,
            "signal_artifact": artifact,
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
            "selected_metrics": {"ic": m["ic"], "icir": m["icir"],
                                 "metric_path": "validation.metrics",
                                 "max_abs_library_correlation": corr,
                                 "correlation_path": "validation.metrics.max_abs_library_correlation"},
        },
    }

    out = Path(f"factors/{fid}.json")
    out.write_text(json.dumps(rec))
    # sidecar csv: long format date,symbol,value
    long = panel.reset_index().melt(id_vars="date", var_name="symbol", value_name="value").dropna()
    long.to_csv(f"factors/{fid}_signal.csv", index=False)
    print(f"WROTE factors/{fid}.json ({out.stat().st_size/1024:.0f} KB) + sidecar "
          f"({long.shape[0]} rows) | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
          f"libcorr={corr:.3f} decay10={m['decay_ic_by_horizon']['10']}")
    return rec


# --- 1. rate_beta_cn10y_60d ---
cn10 = panels["CN10Y"]["close"].astype(float)
panel1 = rolling_beta(rets, cn10.pct_change(), 60)
build_factor(panel1,
             "rate_beta_cn10y_60d",
             "CN10Y rate beta 60d",
             "beta(asset_ret, pct_change(CN10Y), 60)",
             "Rolling 60-day regression beta of each asset's daily return on the CN10Y yield "
             "change. High (less negative) beta = asset co-moves with China rates; negative IC "
             "means assets with lower CN10Y-beta (more defensive to China rate shocks) earn "
             "higher forward 10d returns.",
             ["close", "CN10Y"],
             {"beta_win": 60, "min_obs": 40},
             -1,
             ["macro-beta", "rates", "cross-asset"])

# --- 2. eurusd_beta_60d ---
eur = panels["EURUSD"]["close"].astype(float)
panel2 = rolling_beta(rets, eur.pct_change(), 60)
build_factor(panel2,
             "eurusd_beta_60d",
             "EURUSD beta 60d",
             "beta(asset_ret, pct_change(EURUSD), 60)",
             "Rolling 60-day regression beta of each asset's daily return on EURUSD changes "
             "(proxy for global risk-appetite / dollar weakness). Negative IC: assets with "
             "lower EURUSD-beta earn higher forward 10d returns.",
             ["close", "EURUSD"],
             {"beta_win": 60, "min_obs": 40},
             -1,
             ["macro-beta", "fx", "cross-asset"])

# --- 3. dn_mkt_beta_60d ---
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
panel3 = rolling_beta(rets, dn, 60)
build_factor(panel3,
             "dn_mkt_beta_60d",
             "Downside market beta 60d",
             "beta(asset_ret, min(mkt_ret,0), 60) where mkt_ret = equal-weight mean of 15 assets",
             "Rolling 60-day regression beta of each asset's return on down-market days only "
             "(equal-weight cross-asset market). Measures tail-capture / flight-to-safety "
             "behavior: assets with LOW downside beta are safe-havens. Positive IC: low "
             "downside-beta assets earn higher forward 10d returns.",
             ["close"],
             {"beta_win": 60, "min_obs": 40, "market": "equal_weight_15"},
             1,
             ["tail-risk", "beta", "cross-asset"])

print("\nAll 3 factors persisted with embedded signal artifacts.")
