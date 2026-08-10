"""miner_3 cycle 34: persist the three gate-passing factors with calmness_20-style schema.

Persisted this cycle:
  downbeta_spx_60     IC=+0.0752 ICIR=+0.1871 maxrho=0.0483  dir=+1
  lagbeta_spx_60      IC=-0.0339 ICIR=-0.0996 maxrho=0.0588  dir=-1
  vol_price_corr_60   IC=+0.0533 ICIR=+0.1153 maxrho=0.2280  dir=+1

Gate-style stacked-Spearman rho (computed against active library artifacts) < 0.5 for all.
Regime notes record recent fading honestly (last250 ICs negative for downbeta & vol_price_corr,
sign flip for lagbeta) so the ensemble can weight accordingly.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor,
                         VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

ACTIVE = ["calmness_20", "dxy_beta_cond_60x20", "intraday_drift_20",
          "mom20_volproxy60", "usdjpy_beta_cond_120x60"]
lib = {}
for fid in ACTIVE:
    a = np.load(Path("factors") / f"{fid}.signal.npy")
    lib[fid] = pd.DataFrame(a, index=panel.index, columns=panel.columns)


def stacked_spearman(a, b):
    df = pd.concat([a.stack().rename("x"), b.stack().rename("y")], axis=1).dropna()
    return float(df["x"].corr(df["y"], method="spearman")) if len(df) >= 30 else 0.0


def regime_breakdown(ic_ser):
    reg = {}
    for r0, r1, tag in [("2020-01-01", "2021-12-31", "2020-21"),
                        ("2022-01-01", "2022-12-31", "2022"),
                        ("2023-01-01", "2024-12-31", "2023-24"),
                        ("2025-01-01", "2026-07-29", "2025-26")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            reg[tag] = (f"ic={sub.mean():+.4f} icir={(sub.mean()/sd if sd > 0 else 0):+.3f} n={len(sub)}")
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        reg["last250"] = f"ic={last.mean():+.4f} icir={(last.mean()/sd if sd > 0 else 0):+.3f} n={len(last)}"
    return reg


def load_ohlc():
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date").set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


OHLC = load_ohlc()
spx_ret = panel["SPX"].dropna().pct_change()

# ---- factor computations (identical to cycle 33/34 screen) ----
def down_beta(s, w=60, minp=15):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if i < w - 1:
            continue
        seg = df.iloc[max(0, i - w + 1): i + 1]
        neg = seg[seg["m"] < 0]
        if len(neg) < minp:
            continue
        v = neg["m"].var()
        if v > 0:
            out.iloc[i] = neg["a"].cov(neg["m"]) / v
    return out.reindex(panel.index)


def lag_beta(s, w=60, minp=30):
    ar = s.pct_change()
    spx_lag = spx_ret.shift(1)
    df = pd.concat([ar.rename("a"), spx_lag.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b.reindex(panel.index)


def vol_price_corr(df, w=60, minp=30):
    ar = df["close"].pct_change().abs()
    return ar.rolling(w, min_periods=minp).corr(df["volume"])


def build_ohlc(func):
    out = {}
    for a in TRADABLES:
        out[a] = func(OHLC[a].dropna()).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


factors = {
    "downbeta_spx_60": {
        "panel": per_asset(panel, down_beta),
        "name": "Downside beta to SPX 60d (crisis beta)",
        "expr": "beta(asset_ret, SPX_ret | SPX_ret<0, 60d, min 15 down-obs) per asset own calendar",
        "desc": "Rolling beta of each asset's daily returns on SPX daily returns computed only on days when SPX fell (60d window, min 15 down-observations). High downside beta = asset crashes with the market; low/negative = crisis hedge. Full-sample 10d rank IC +0.0752 (ICIR +0.187): in this 15-asset cross-market universe, HIGH downside beta assets (risk assets) earn higher forward 10d returns - a crisis-beta risk premium. Low stacked-Spearman correlation vs active library (max 0.048).",
        "deps": ["close", "SPX"],
        "params": {"window": 60, "min_down_obs": 15},
        "direction": 1,
        "tags": ["risk", "beta", "tail", "downside", "cross-asset"],
    },
    "lagbeta_spx_60": {
        "panel": per_asset(panel, lag_beta),
        "name": "Lagged-market beta to SPX 60d",
        "expr": "beta(asset_ret_t, SPX_ret_{t-1}, 60d) per asset own calendar",
        "desc": "Rolling beta of each asset's daily return on the LAGGED (t-1) SPX return: measures delayed/slow reaction to market moves (non-synchronous beta). Full-sample 10d rank IC -0.0339 (ICIR -0.0996), so expected_direction=-1: assets that react slowly to market news (low lag-beta) outperform. Interpretation: lagged beta proxies for stale pricing / microstructure frictions that are penalized in forward 10d returns. Low stacked-Spearman correlation vs active library (max 0.059).",
        "deps": ["close", "SPX"],
        "params": {"window": 60, "min_periods": 30},
        "direction": -1,
        "tags": ["beta", "microstructure", "lag", "cross-asset"],
    },
    "vol_price_corr_60": {
        "panel": build_ohlc(vol_price_corr),
        "name": "Volume-volatility feedback 60d",
        "expr": "rolling_corr(|daily_ret|, volume, 60d, min 30 obs) per asset own calendar",
        "desc": "60d rolling correlation between absolute daily returns and daily volume: whether trading activity concentrates on volatile days (positive feedback) or on quiet days (negative). High values = volume spikes accompany price moves (panic/euphoria trading). Full-sample 10d rank IC +0.0533 (ICIR +0.115): high volume-vol feedback assets earn higher forward 10d returns. NOTE: coverage limited to 9/15 assets (SOX, XAU, COPPER, WTI, US10Y, CN10Y lack usable volume series); n_ic_dates=1483 with >=8 valid names. Low stacked-Spearman correlation vs active library (max 0.228).",
        "deps": ["close", "volume"],
        "params": {"window": 60, "min_periods": 30},
        "direction": 1,
        "tags": ["volume", "liquidity", "feedback", "cross-asset"],
    },
}

results = {}
for fid, spec in factors.items():
    F = spec["panel"]
    m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    lc = {k: round(stacked_spearman(F, sig), 4) for k, sig in lib.items()}
    m["max_abs_library_correlation"] = round(max((abs(v) for v in lc.values()), default=0.0), 4)
    m["library_pairwise_corr"] = lc
    m["turnover_10d_rank"] = m.pop("turnover_10_rank", None)
    ic_ser = compute_ic(F, fwd_cache[str(ADM_H)]).dropna()
    reg = regime_breakdown(ic_ser)
    reg_note = " | ".join([f"{k}:{v}" for k, v in reg.items()])

    ic, icir = abs(m["ic"]), abs(m["icir"])
    assert ic >= 0.007 and icir >= 0.084, f"{fid} fails IC gate"
    assert abs(m["max_abs_library_correlation"]) < 0.5, f"{fid} fails correlation gate"

    doc = {
        "factor_id": fid,
        "factor_name": spec["name"],
        "version": "1.0.0",
        "calculation": {
            "expression": spec["expr"],
            "description": spec["desc"],
            "transform": "rank cross-sectionally (pct rank); portfolio uses direction=sign(IC)",
        },
        "dependencies": spec["deps"],
        "parameters": spec["params"],
        "expected_direction": spec["direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-29",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": f"15-instrument tradable cross-asset universe. {reg_note}",
            "metrics": m,
        },
        "tags": spec["tags"],
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
                "reported_max_abs_library_correlation": m["max_abs_library_correlation"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(abs(m["ic"]) * abs(m["icir"]), 8),
            },
            "admitted_at": "2026-08-11T01:20:00.000000",
        },
        "signal_artifact": f"{fid}.signal.npy",
        "artifact_provenance": {
            "format": "npy_matrix",
            "shape": list(F.shape),
            "columns": list(F.columns),
            "dates_first": str(F.index[0].date()),
            "dates_last": str(F.index[-1].date()),
            "n_nan": int((~F.notna()).sum().sum()),
        },
    }
    out = Path("factors") / f"{fid}.json"
    out.write_text(json.dumps(doc, indent=1))
    np.save(Path("factors") / f"{fid}.signal.npy", F.values)
    results[fid] = {"metrics": m, "status": "EFFECTIVE", "regime": reg}
    print(f"[persist] {fid}: ic={m['ic']} icir={m['icir']} maxrho={m['max_abs_library_correlation']} -> {out}")

    # ---- read back and verify ----
    chk = json.load(open(out))
    ok = (chk["factor_id"] == fid
          and chk["validation"]["status"] == "EFFECTIVE"
          and abs(chk["validation"]["metrics"]["ic"]) >= 0.007
          and abs(chk["validation"]["metrics"]["icir"]) >= 0.084
          and Path("factors", chk["signal_artifact"]).exists()
          and np.load(Path("factors", chk["signal_artifact"])).shape == tuple(F.shape))
    print(f"[verify] {fid}: id={chk['factor_id'] == fid} status={chk['validation']['status']} "
          f"ic={chk['validation']['metrics']['ic']} icir={chk['validation']['metrics']['icir']} "
          f"rho={chk['validation']['metrics']['max_abs_library_correlation']} "
          f"artifact={Path('factors', chk['signal_artifact']).exists()} shape_ok={ok} ALL_OK={ok}")

json.dump(results, open("scripts/_miner3_cycle34_persist_results.json", "w"), indent=1, default=float)
print("\nDONE persist cycle34")
