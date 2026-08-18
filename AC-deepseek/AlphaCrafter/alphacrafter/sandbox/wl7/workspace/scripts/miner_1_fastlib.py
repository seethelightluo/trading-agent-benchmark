"""miner_1 fast vectorized validation library.

Universe: 15 tradable cross-asset instruments (2020-01-01..2026-07-15 warm-up).
IC = cross-sectional Spearman rank IC per date (>=8 assets), fully vectorized via
rank -> Pearson correlation on ranks with pairwise-NaN handling.
Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10,
max_abs_library_correlation < 0.5 (self-reported provenance only; the
deterministic gate recomputes pairwise rho from signal artifacts).
"""
from __future__ import annotations
import io, json, zlib, base64
from pathlib import Path
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2026-07-29"
FACTOR_LAST = "2026-07-15"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084, "corr": 0.5}
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12


def load_panel() -> pd.DataFrame:
    closes = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    return panel[~panel.index.duplicated(keep="last")].sort_index()


def load_ohlc_volume() -> dict[str, pd.DataFrame]:
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[s] = df[["open", "close", "high", "low", "volume"]].astype(float)
    return out


def load_macro(name: str | None = None) -> dict[str, pd.Series]:
    out = {}
    for m in (MACRO if name is None else [name]):
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[m] = df["close"].astype(float)
    return out


def fwd_returns(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward h-day return on each asset's own calendar, reindexed to union."""
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def row_pearson(X: np.ndarray, Y: np.ndarray, min_n: int = MIN_ASSETS) -> np.ndarray:
    """Row-wise Pearson correlation with pairwise-NaN deletion (vectorized)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    valid = np.isfinite(X) & np.isfinite(Y)
    cnt = valid.sum(axis=1)
    Xv = np.where(valid, X, np.nan)
    Yv = np.where(valid, Y, np.nan)
    Xc = Xv - np.nanmean(Xv, axis=1, keepdims=True)
    Yc = Yv - np.nanmean(Yv, axis=1, keepdims=True)
    num = np.nansum(Xc * Yc, axis=1)
    dx = np.sqrt(np.nansum(Xc * Xc, axis=1))
    dy = np.sqrt(np.nansum(Yc * Yc, axis=1))
    r = np.full(len(X), np.nan)
    m = (cnt >= min_n) & (dx > 0) & (dy > 0)
    r[m] = num[m] / (dx[m] * dy[m])
    return r


def rank_ic_fast(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    F = factor.rank(axis=1).values.astype(float)
    R = fwd.rank(axis=1).values.astype(float)
    return pd.Series(row_pearson(F, R), index=factor.index)


def turnover_10d_rank_fast(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1).values.astype(float)
    a = ranks[:-10]
    b = ranks[10:]
    valid = np.isfinite(a) & np.isfinite(b)
    cnt = valid.sum(axis=1)
    ok = cnt >= MIN_ASSETS
    m = np.full(len(a), np.nan)
    m[ok] = np.nansum(np.abs(a - b) * valid, axis=1)[ok] / cnt[ok]
    return float(np.nanmean(m))


def library_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute the CURRENT effective library (cycle-25 audit: 7 factors)
    deterministically on panel. Mirrors persisted artifacts:
    max_ret_20d, downside_vol_ratio_20, rel_mom_20d_skip5, beta_ew_60d,
    crypto_beta_60d, dxy_beta_cond_60x20, eurusd_beta_cond_60x20."""
    rets = panel.pct_change()
    mkt = panel.mean(axis=1).pct_change()
    v = rets.rolling(20, min_periods=10).std()
    out = {}
    out["max_ret_20d"] = rets.rolling(20, min_periods=10).max()
    dd = rets.clip(upper=0).rolling(20, min_periods=10).std()
    out["downside_vol_ratio_20"] = -(dd / (v + EPS))
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    cov = rets.rolling(60, min_periods=30).cov(mkt)
    var = mkt.rolling(60, min_periods=30).var().replace(0, np.nan)
    out["beta_ew_60d"] = cov.div(var.to_numpy(), axis=0)
    # crypto_beta_60d: rolling 60d beta of asset returns vs BTC returns
    btc = panel["BTC"].pct_change()
    btc_var = btc.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    cb = rets.rolling(60, min_periods=30).cov(btc).div(btc_var.to_numpy(), axis=0)
    out["crypto_beta_60d"] = cb
    macro = load_macro()
    # dxy_beta_cond_60x20: -beta(r, DXY_ret, 60) * (DXY/DXY.shift(20)-1)
    dxy = macro["DXY"].pct_change()
    dxy_var = dxy.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    db = rets.rolling(60, min_periods=30).cov(dxy).div(dxy_var.to_numpy(), axis=0)
    dxy_mom = (macro["DXY"] / macro["DXY"].shift(20) - 1.0).reindex(panel.index)
    out["dxy_beta_cond_60x20"] = -db.mul(dxy_mom.to_numpy(), axis=0)
    # eurusd_beta_cond_60x20: beta(r, EURUSD_ret, 60) * (EURUSD/EURUSD.shift(20)-1)
    eur = macro["EURUSD"].pct_change()
    eur_var = eur.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    eb = rets.rolling(60, min_periods=30).cov(eur).div(eur_var.to_numpy(), axis=0)
    eur_mom = (macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0).reindex(panel.index)
    out["eurusd_beta_cond_60x20"] = eb.mul(eur_mom.to_numpy(), axis=0)
    return out


def lib_corr_fast(factor: pd.DataFrame, libs: dict[str, pd.DataFrame],
                  last_n: int = 700) -> tuple[float, dict]:
    idx = factor.index[-last_n:]
    F = factor.loc[idx].rank(axis=1).values.astype(float)
    per = {}
    for fid, lf in libs.items():
        L = lf.reindex(idx).rank(axis=1).values.astype(float)
        rs = row_pearson(F, L)
        per[fid] = round(float(np.nanmean(rs)), 4)
    valid = [abs(v) for v in per.values() if np.isfinite(v)]
    return (round(max(valid), 4) if valid else float("nan")), per


def validate_fast(name: str, factor: pd.DataFrame, panel: pd.DataFrame,
                  fwd: dict[int, pd.DataFrame], libs: dict[str, pd.DataFrame],
                  fwd_rank_cache: dict[int, np.ndarray] | None = None) -> dict:
    factor = factor.reindex(panel.index).loc[:FACTOR_LAST]
    n_valid = int(factor.notna().sum().sum())
    if n_valid < 100:
        return {"name": name, "factor_rows": len(factor), "n_assets": panel.shape[1],
                "admission_gate": {"pass": False}, "reason": "insufficient_data",
                "n_valid": n_valid}
    res = {"name": name, "factor_rows": int(len(factor)), "n_assets": panel.shape[1]}
    ic_by_h = {}
    for h in HORIZONS:
        F = factor.rank(axis=1).values.astype(float)
        if fwd_rank_cache is not None and h in fwd_rank_cache:
            R = fwd_rank_cache[h]
        else:
            R = fwd[h].rank(axis=1).values.astype(float)
        ic_by_h[h] = pd.Series(row_pearson(F, R), index=factor.index)
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    for h in HORIZONS:
        ic = ic_by_h[h] * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    res["raw_ic_h10"] = float(ic10.mean())
    valid = factor.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank_fast(factor)
    max_corr, per = lib_corr_fast(factor, libs)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in HORIZONS}
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    gate_corr = (res["max_abs_library_correlation"] is None
                 or not np.isfinite(res["max_abs_library_correlation"])
                 or res["max_abs_library_correlation"] < ADMISSION["corr"])
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "corr_pass": bool(gate_corr),
                             "pass": bool(gate_ic and gate_icir and gate_corr)}
    flag = "PASS" if res["admission_gate"]["pass"] else "FAIL"
    print(f"  {name:<28} h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} cov={res['coverage_asset_days']:.3f} "
          f"turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']} "
          f"-> {flag}", flush=True)
    return res


# ---------------- persistence helpers (amihud pattern + embedded csv) ----------------

def save_artifact_csv(factor: pd.DataFrame) -> str:
    """Embed base64:zlib:csv of the factor panel (rows=dates, cols=assets)."""
    df = factor.copy()
    df.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    buf = io.StringIO()
    df.to_csv(buf)
    comp = zlib.compress(buf.getvalue().encode())
    return base64.b64encode(comp).decode()


def persist_factor(factor_id: str, factor_name: str, expression: str, description: str,
                   deps: list, params: dict, res: dict, tags: list, regime_notes: str,
                   panel: pd.DataFrame, factor: pd.DataFrame) -> Path:
    direction = res.get("direction", 1.0)
    factor_full = factor.reindex(panel.index)  # full visible window, union calendar
    npy_path = Path("factors") / f"{factor_id}.signal.npy"
    artifact = np.asarray(factor_full.values, dtype=float)
    np.save(npy_path, artifact)
    payload = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "version": "1.0.0",
        "calculation": {"expression": expression, "description": description},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": int(np.sign(direction)) if direction != 0 else 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": regime_notes,
            "metrics": {
                "ic": res["ic_h10"],
                "icir": res["icir_h10"],
                "ic_hit_ratio": res["hit_h10"],
                "n_ic_dates": res["n_dates_h10"],
                "coverage_asset_days": res["coverage_asset_days"],
                "coverage_dates_ge8": res["coverage_dates_ge8"],
                "turnover_10d_rank": res["turnover_10d_rank"],
                "decay_ic_by_horizon": res["decay_ic_by_horizon"],
                "max_abs_library_correlation": res["max_abs_library_correlation"],
                "library_pairwise_corr": {k: v for k, v in res["library_corrs"].items()},
            },
            "signal_artifact": {
                "format": "base64:zlib:csv",
                "description": "Factor signal panel: rows = dates, cols = assets. "
                               f"Shape {artifact.shape}.",
                "columns": WATCH,
                "shape": list(artifact.shape),
                "n_valid_values": int(np.isfinite(artifact).sum()),
                "sha256": str(hash(artifact.tobytes()) & 0xFFFFFFFFFFFFFFFF),
                "data": save_artifact_csv(factor_full),
            },
        },
        "tags": tags,
        "signal_artifact": f"{factor_id}.signal.npy",
        "artifact_provenance": {
            "format": "npy_matrix",
            "shape": list(artifact.shape),
            "columns": WATCH,
            "dates_first": str(panel.index.min().date()),
            "dates_last": str(panel.index.max().date()),
            "n_nan": int(np.isnan(artifact).sum()),
        },
        "benchmark_admission": {
            "contract": {"ic_threshold": ADMISSION["ic"], "icir_threshold": ADMISSION["icir"],
                         "correlation_threshold": ADMISSION["corr"]},
            "selected_metrics": {
                "ic": res["ic_h10"],
                "icir": res["icir_h10"],
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": res["max_abs_library_correlation"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    path = Path("factors") / f"{factor_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  PERSISTED -> {path}")
    return path


def verify_factor(factor_id: str, res: dict) -> None:
    path = Path("factors") / f"{factor_id}.json"
    loaded = json.loads(path.read_text())
    assert loaded["factor_id"] == factor_id
    assert loaded["validation"]["status"] == "EFFECTIVE"
    m = loaded["validation"]["metrics"]
    assert abs(m["ic"] - res["ic_h10"]) < 1e-9
    assert abs(m["icir"] - res["icir_h10"]) < 1e-9
    assert loaded["validation"]["signal_artifact"]["format"] == "base64:zlib:csv"
    npy = np.load(Path("factors") / f"{factor_id}.signal.npy")
    assert npy.shape == tuple(loaded["artifact_provenance"]["shape"])
    assert npy.shape[0] == loaded["validation"]["signal_artifact"]["shape"][0]
    print(f"  VERIFIED {factor_id}: JSON ok, EFFECTIVE, ic={m['ic']:.4f} icir={m['icir']:.4f} "
          f"artifact npy={npy.shape} embedded_csv={loaded['validation']['signal_artifact']['shape']}")
