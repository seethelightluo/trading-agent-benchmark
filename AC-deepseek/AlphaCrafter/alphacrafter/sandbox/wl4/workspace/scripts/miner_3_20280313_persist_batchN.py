"""miner_3 persistence for batch N passers (2028-03-13 simulation date).

Persists factors that passed the admission gate (|IC|>=0.0070, |ICIR|>=0.0840
at h=10, 15-asset universe, min_valid=8) and are INCREMENTAL vs the active
library (max_abs_library_correlation < 0.5):

  1. mom_vs_median_60d        (expected_direction -1)  full-sample 2020..2028, n=1344 IC dates
  2. us10y_cond_beta_60d      (expected_direction -1)  2020..2028, n=682 IC dates
  3. vol_adj_mom_accel_20x60  (expected_direction +1)  strong recent IC, data-density-limited coverage

NOT persisted here (gate passers but non-incremental / not robust):
  - vix_cond_dnbeta_60d  : rho=0.876 with active dn_mkt_beta_60d (near-clone)
  - rel_mom_vs_btc_60d   : numerically identical to mom_vs_median_60d (redundant)
  - skew_roll20_60d      : sign flips within its single (recent-only) regime
  - trend_r2_voladj_20d  : marginal (ic 0.0262) and recent-only coverage
  - volume_skew_20d      : n=4 IC dates - insufficient for validation

Artifact format matches existing library: base64:zlib:csv of the signal panel,
sha256 = first 16 hex of raw csv bytes.
"""
import sys, time, json, zlib, base64, hashlib
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
mkt = rets.mean(axis=1)
print(f"panels {closes.shape} {closes.index.min().date()}..{closes.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def rank_ic_series_fast(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    rf = factor_panel.rank(axis=1, method="average")
    rr = fwd.rank(axis=1, method="average")
    valid = rf.notna() & rr.notna()
    nv = valid.sum(axis=1)
    rf2 = rf.where(valid)
    rr2 = rr.where(valid)
    mu_f = rf2.sum(axis=1) / nv.replace(0, np.nan)
    mu_r = rr2.sum(axis=1) / nv.replace(0, np.nan)
    cf = rf2.sub(mu_f, axis=0).fillna(0.0)
    cr = rr2.sub(mu_r, axis=0).fillna(0.0)
    ssf = (cf ** 2).sum(axis=1)
    ssr = (cr ** 2).sum(axis=1)
    cov = (cf * cr).sum(axis=1)
    ic = cov / np.sqrt(ssf * ssr).replace(0, np.nan)
    ok = (nv >= min_valid) & (ssf > 1e-14) & (ssr > 1e-14) & ic.notna()
    return ic[ok].rename("ic")


def summarize_fast(ic_series: pd.Series):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    return {"ic": ic, "icir": ic / sd if sd > 0 else 0.0,
            "ic_hit_ratio": float((ic_series > 0).mean()), "n_ic_dates": int(len(ic_series)),
            "ic_std": sd}


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def rolling_corr_fast(a, b, win=20, min_obs=15):
    n = a.rolling(win).count()
    cov = (a * b).rolling(win).mean() - a.rolling(win).mean() * b.rolling(win).mean()
    den = a.rolling(win).std() * b.rolling(win).std()
    return (cov / den.replace(0, np.nan)).where(n >= min_obs)


# ---------- active library signals (for correlation reference) ----------
def lib_vol_price_corr_20():
    return pd.DataFrame({a: rolling_corr_fast(rets[a], vol_panel[a], 20, 15) for a in rets.columns},
                        index=rets.index)


def lib_dn_mkt_beta_60d():
    return rolling_beta(rets, mkt.where(mkt < 0), 60, 40)


def lib_rate_beta_cn10y_60d():
    return rolling_beta(rets, rets["CN10Y"], 60, 40)


LIBRARY = {
    "vol_price_corr_20": lib_vol_price_corr_20(),
    "dn_mkt_beta_60d": lib_dn_mkt_beta_60d(),
    "rate_beta_cn10y_60d": lib_rate_beta_cn10y_60d(),
}
print(f"library signals {time.time()-t0:.1f}s", flush=True)

# ---------- candidate panels (exact batch-N definitions) ----------
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
v20 = rets.rolling(20).std()

cands = {
    "mom_vs_median_60d": m60.sub(m60.median(axis=1), axis=0),
    "us10y_cond_beta_60d": rolling_beta(rets, rets["US10Y"], 60, 40) * pd.DataFrame(
        {a: np.sign(closes["US10Y"] / closes["US10Y"].shift(20) - 1.0) for a in rets.columns},
        index=rets.index).reindex(rets.index),
    "vol_adj_mom_accel_20x60": (m20 - m60) / v20.replace(0, np.nan),
}

META = {
    "mom_vs_median_60d": {
        "factor_name": "Cross-sectional relative momentum vs median 60d",
        "expression": "close/close.shift(60) - 1 - cross_sectional_median(close/close.shift(60) - 1)",
        "description": "60d momentum of each asset minus the cross-sectional median 60d momentum of the "
                       "15-asset universe. Negative IC at h=10: relative laggards (below-median momentum) "
                       "outperform over the next 10 days - short-horizon cross-sectional mean reversion.",
        "dependencies": ["close"], "parameters": {"window": 60, "median_mode": "cross_sectional"},
        "expected_direction": -1, "tags": ["momentum", "cross-asset", "mean-reversion"],
    },
    "us10y_cond_beta_60d": {
        "factor_name": "US10Y-conditional rate beta 60d",
        "expression": "sign(mom20(US10Y)) * rolling_beta(asset_ret, US10Y_ret, 60, min_obs=40)",
        "description": "60d beta of each asset to US10Y returns, sign-flipped by the 20d momentum of US10Y. "
                       "Negative IC at h=10: assets with higher rate-beta while US10Y is rising "
                       "(i.e., more exposed to rising long-end yields) underperform - rate-sensitivity penalty.",
        "dependencies": ["close"], "parameters": {"beta_win": 60, "min_obs": 40, "cond_window": 20, "driver": "US10Y"},
        "expected_direction": -1, "tags": ["rate-beta", "cross-asset", "conditional"],
    },
    "vol_adj_mom_accel_20x60": {
        "factor_name": "Volatility-adjusted momentum acceleration 20x60",
        "expression": "(close/close.shift(20) - 1 - (close/close.shift(60) - 1)) / rolling_std(ret, 20)",
        "description": "Momentum acceleration: 20d momentum minus 60d momentum, scaled by 20d return "
                       "volatility. Positive values = recently accelerating assets (per unit risk). "
                       "Positive IC at h=10 with IC rising through h=20: accelerating assets keep "
                       "outperforming (momentum continuation). Coverage limited to dense-data windows.",
        "dependencies": ["close"], "parameters": {"fast_window": 20, "slow_window": 60, "vol_window": 20},
        "expected_direction": 1, "tags": ["momentum", "volatility", "cross-asset"],
    },
}

# ---------- evaluate & persist ----------
fwd = forward_returns(closes, H_ADM)
ADMITTED_AT = "2028-03-13T00:00:00.000000"
CONTRACT = {"ic_threshold": GATE_IC, "icir_threshold": GATE_ICIR, "correlation_threshold": 0.5,
            "library_capacity": 30, "active_top_k": 10}

for fid, panel in cands.items():
    meta = META[fid]
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        ih = rank_ic_series_fast(panel, forward_returns(closes, h), MIN_VALID)
        m["decay_ic_by_horizon"][str(h)] = round(float(ih.mean()), 4) if len(ih) else None
    corr, key = max_library_corr(panel, LIBRARY)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    for cut_name, cut in (("r250", closes.index[-250]), ("r500", closes.index[-500]), ("r750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            m[f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            m[f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0

    # signal artifact: base64(zlib(csv))
    csv_bytes = panel.to_csv().encode("utf-8")
    data_b64 = base64.b64encode(zlib.compress(csv_bytes)).decode("ascii")
    sha = hashlib.sha256(csv_bytes).hexdigest()[:16]
    n_valid = int(panel.notna().sum().sum())
    artifact = {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape ({panel.shape[0]}, {panel.shape[1]})",
        "columns": TRADABLE,
        "shape": [panel.shape[0], panel.shape[1]],
        "n_valid_values": n_valid,
        "sha256": sha,
        "data": data_b64,
    }

    metrics_for_file = {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) and k != "n_ic_dates" else v)
                        for k, v in m.items()}
    doc = {
        "factor_id": fid,
        "factor_name": meta["factor_name"],
        "version": "1.0.0",
        "calculation": {"expression": meta["expression"], "description": meta["description"]},
        "dependencies": meta["dependencies"],
        "parameters": meta["parameters"],
        "expected_direction": meta["expected_direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{closes.index.min().date()}..{closes.index.max().date()}",
            "last_validated": "2028-03-13",
            "admission_horizon": H_ADM,
            "regime_notes": (
                "Validated 2020-01-01..2028-03-10 on the 15-asset cross-asset universe "
                "(min_valid=8, h=10) across COVID crash 2020, 2021-22 tightening bear, 2023-24 equity "
                "rally, 2024-26 crypto/commodity cycles and 2027-28 tape. Synthetic data has frequent "
                "missing days for non-crypto assets; rolling-window factors are only computable in "
                "dense-data windows (recent period), which narrows regime coverage for "
                f"{fid} - see coverage metrics and yearly-IC caveats. Direction sign: "
                f"{'positive' if meta['expected_direction'] > 0 else 'negative'} IC."
            ),
            "metrics": metrics_for_file,
            "signal_artifact": artifact,
        },
        "tags": meta["tags"],
        "benchmark_admission": {
            "contract": CONTRACT,
            "selected_metrics": {
                "ic": round(float(m["ic"]), 6),
                "icir": round(float(m["icir"]), 6),
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": corr,
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(float(m["ic"] * m["icir"]), 8),
            },
            "admitted_at": ADMITTED_AT,
        },
    }
    out_path = f"factors/{fid}.json"
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"WROTE {out_path} ic={m['ic']:.4f} icir={m['icir']:.4f} n={m['n_ic_dates']} "
          f"cov={m['coverage_asset_days']:.3f} libcorr={corr:.4f} valid={n_valid} sha={sha} ({time.time()-t0:.1f}s)",
          flush=True)

print("DONE", flush=True)
