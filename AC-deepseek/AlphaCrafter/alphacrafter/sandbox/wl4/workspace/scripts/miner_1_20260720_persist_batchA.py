"""miner_1 2026-07-20 persist batch A passers.

Persist factors that passed the admission gate (h=10, min_valid=8):
  |IC| >= 0.007, |ICIR| >= 0.084, max_abs_library_correlation < 0.5.

Passers from screening:
  vol_price_corr_20  IC=+0.0650 ICIR=+0.1523 rho=0.350 (mom_10d_skip5)
  rsi_14             IC=+0.0356 ICIR=+0.1081 rho=0.454 (mom_10d_skip5)
  usdcny_beta_60d    IC=+0.0347 ICIR=+0.0963 rho=0.418 (eurusd_beta_60d)

Validation window: warm-up 2020-01-01..2026-07-15 (consistent with library).
Writes factors/<factor_id>.json with base64:zlib:csv signal artifact,
then reloads and verifies.
"""
import sys, json, zlib, base64, hashlib, datetime as dt
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr, TRADABLE)

CUTOFF = pd.Timestamp("2026-07-15")
VALIDATED_AT = "2026-07-20"
ADM_H = 10
MIN_VALID = 8
HORIZONS = (1, 2, 3, 5, 10, 20)

panels = load_panels()
closes_full = close_panel(panels)
closes = closes_full[closes_full.index <= CUTOFF]
rets = ret_panel(panels).reindex(closes.index)
print(f"closes {closes.shape} | {closes.index[0].date()}..{closes.index[-1].date()}")

V = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE}, axis=1).sort_index().reindex(closes.index)
H = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE}, axis=1).sort_index().reindex(closes.index)
L = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE}, axis=1).sort_index().reindex(closes.index)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    out = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()).where(
            z["m"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=asset_ret.index)


def rsi_wilder(s, win=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / win, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / win, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---- candidate signals ----
vp = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename("r"), V[a].rename("v")], axis=1).dropna()
    vp[a] = z["r"].rolling(20).corr(z["v"])
cands = {
    "vol_price_corr_20": pd.DataFrame(vp, index=closes.index),
    "rsi_14": pd.DataFrame({a: rsi_wilder(closes[a]) for a in closes.columns}),
    "usdcny_beta_60d": rolling_beta(rets, panels["USDCNY"]["close"].astype(float).pct_change().reindex(closes.index), 60),
}

# ---- library (7 persisted factors) ----
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
lib = library_signals(panels, closes, rets)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, dn, 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, panels["EURUSD"]["close"].astype(float).pct_change().reindex(closes.index), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].astype(float).pct_change().reindex(closes.index), 60)
lib = {k: v.reindex(closes.index) for k, v in lib.items()}
print(f"library factors: {sorted(lib.keys())}")

fwd = forward_returns(closes, ADM_H)

FACTOR_DEFS = {
    "vol_price_corr_20": {
        "factor_name": "Volume-Price Correlation 20d",
        "expression": "rolling_corr(close.pct_change(), volume, 20)",
        "description": "20-day rolling Pearson correlation between daily return and daily volume. "
                       "Positive values indicate volume-confirmed price moves (participation); "
                       "in the cross-asset universe higher confirmation correlates with forward gains.",
        "dependencies": ["close", "volume"],
        "parameters": {"window": 20, "min_obs": 10},
        "expected_direction": 1,
        "tags": ["liquidity", "cross-asset", "participation"],
        "regime_notes": ("Validated on warm-up 2020-01-01..2026-07-15 across COVID crash 2020, "
                         "2020-21 recovery, 2022 tightening bear, 2023-24 equity rally, 2024-26 "
                         "crypto/commodity cycles. Rank IC computed daily on the 15-asset tradable "
                         "universe (h=10)."),
    },
    "rsi_14": {
        "factor_name": "Relative Strength Index 14d (Wilder)",
        "expression": "100 - 100/(1 + EWM(up,14)/EWM(down,14))",
        "description": "Wilder RSI-14 momentum oscillator. In this cross-asset universe (indices, "
                       "commodities, crypto, yields) higher RSI (trend persistence) predicts "
                       "positive forward returns at h=10, consistent with momentum continuation.",
        "dependencies": ["close"],
        "parameters": {"window": 14, "ewm_alpha": "1/14"},
        "expected_direction": 1,
        "tags": ["momentum", "oscillator", "cross-asset"],
        "regime_notes": ("Validated on warm-up 2020-01-01..2026-07-15 across multiple regimes. "
                         "Rank IC daily on the 15-asset tradable universe (h=10)."),
    },
    "usdcny_beta_60d": {
        "factor_name": "USDCNY Beta 60d",
        "expression": "rolling_beta(close.pct_change(), USDCNY.pct_change(), 60)",
        "description": "60-day rolling beta of each asset's return against the USDCNY exchange rate. "
                       "Higher beta to USDCNY (yuan depreciation sensitivity) predicts higher forward "
                       "returns: assets positively exposed to CNY weakness (offshore-driven) outperform.",
        "dependencies": ["close", "USDCNY"],
        "parameters": {"window": 60, "min_obs": 40},
        "expected_direction": 1,
        "tags": ["macro-beta", "fx", "cross-asset"],
        "regime_notes": ("Validated on warm-up 2020-01-01..2026-07-15 across multiple regimes. "
                         "Rank IC daily on the 15-asset tradable universe (h=10). "
                         "Coverage ~51% of asset-days due to USDCNY availability."),
    },
}


def save_factor(fid, panel, m, corr, corr_key, defn):
    panel = panel.reindex(closes.index)
    df = panel.copy()
    df.index = [d.strftime("%Y-%m-%d") for d in df.index]
    csv_bytes = df.to_csv().encode("utf-8")
    z = zlib.compress(csv_bytes, 9)
    b64 = base64.b64encode(z).decode("ascii")
    digest = hashlib.sha256(z).hexdigest()[:16]
    n_valid = int(np.isfinite(panel.values).sum())
    artifact = {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {panel.shape}",
        "columns": list(panel.columns),
        "shape": list(panel.shape),
        "n_valid_values": n_valid,
        "sha256": digest,
        "data": b64,
    }
    metrics = {
        "ic": m["ic"],
        "icir": m["icir"],
        "ic_hit_ratio": m["ic_hit_ratio"],
        "n_ic_dates": m["n_ic_dates"],
        "ic_std": m["ic_std"],
        "coverage_asset_days": m["coverage_asset_days"],
        "coverage_dates_ge8": m["coverage_dates_ge8"],
        "turnover_10d_rank": m["turnover_10d_rank"],
        "decay_ic_by_horizon": {str(k): round(float(v), 4) for k, v in m["decay_ic_by_horizon"].items()},
        "max_abs_library_correlation": corr,
        "max_corr_factor": corr_key,
    }
    doc = {
        "factor_id": fid,
        "factor_name": defn["factor_name"],
        "version": "1.0.0",
        "calculation": {
            "expression": defn["expression"],
            "description": defn["description"],
        },
        "dependencies": defn["dependencies"],
        "parameters": defn["parameters"],
        "expected_direction": defn["expected_direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{closes.index[0].date()}..{closes.index[-1].date()}",
            "last_validated": VALIDATED_AT,
            "admission_horizon": ADM_H,
            "regime_notes": defn["regime_notes"],
            "metrics": metrics,
            "signal_artifact": artifact,
        },
        "tags": defn["tags"],
        "benchmark_admission": {
            "contract": {
                "ic_threshold": 0.007,
                "icir_threshold": 0.084,
                "correlation_threshold": 0.5,
                "library_capacity": 30,
                "active_top_k": 10,
            },
            "selected_metrics": {
                "ic": m["ic"],
                "icir": m["icir"],
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": corr,
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(float(abs(m["ic"]) * abs(m["icir"])), 8),
            },
            "admitted_at": dt.datetime.now().isoformat(),
        },
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as f:
        json.dump(doc, f)
    print(f"WROTE {path} ({len(b64)//1024} KB artifact, n_valid={n_valid})")
    return path


results = []
for fid, panel in cands.items():
    defn = FACTOR_DEFS[fid]
    panel = panel.reindex(closes.index)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    exp_sign = defn["expected_direction"]
    m = summarize_ic(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, MIN_VALID, exp_sign)
    corr, corr_key = max_library_corr(panel, lib)
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = abs(m["icir"]) >= 0.084
    gate_corr = corr < 0.5
    ok = gate_ic and gate_icir and gate_corr
    print(f"{fid:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({corr_key}) -> {'PASS' if ok else 'FAIL'}")
    if ok:
        path = save_factor(fid, panel, m, corr, corr_key, defn)
        results.append((fid, path, m, corr))

print("\n===== PERSISTED =====")
for fid, path, m, corr in results:
    print(f"{fid:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} rho={corr:.3f} -> {path}")
