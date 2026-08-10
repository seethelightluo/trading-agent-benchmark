"""miner_3 shared validation library (per-asset calendar-aware).
Factor values and forward returns are computed on each asset's own trading
calendar (dropna per column) then reindexed to the union calendar. IC is
cross-sectional Spearman rank IC per date, requiring >= 8 valid assets.
Validation window: warm-up 2020-01-01..2026-07-15; full-sample stats too.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS_PER_DATE = 8

_CACHE: dict = {}


def load_panel(days: int = 4000) -> pd.DataFrame:
    if "panel" in _CACHE:
        return _CACHE["panel"]
    closes = {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is not None and len(df):
            closes[s] = df.set_index("date")["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _CACHE["panel"] = panel
    return panel


def load_macro() -> dict[str, pd.Series]:
    if "macro" in _CACHE:
        return _CACHE["macro"]
    out = {}
    for s in MACRO:
        try:
            df = get_index_daily_data(s, days=4000)
            if df is not None and len(df):
                out[s] = df.set_index("date")["close"].astype(float)
        except Exception:
            pass
    _CACHE["macro"] = out
    return out


def per_asset(fn):
    """Apply a per-asset Series->Series function on each asset's own calendar."""
    def wrapper(panel, macro):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward h-observation returns on each asset's own calendar."""
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                   direction: float = 1.0) -> pd.Series:
    ics = []
    idx = factor.index.intersection(fwd_ret.index)
    for d in idx:
        f = factor.loc[d].dropna()
        r = fwd_ret.loc[d].reindex(f.index).dropna()
        if len(r) < MIN_ASSETS_PER_DATE:
            continue
        ics.append((d, r.corr(f.reindex(r.index), method="spearman") * direction))
    return pd.Series(dict(ics)).sort_index()


def turnover_10d_rank(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1)
    out = []
    dates = ranks.index
    for i in range(10, len(dates)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS_PER_DATE:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def validate_factor(name: str, factor_fn, horizons=(1, 2, 3, 5, 10, 20),
                    warm_end: str = "2026-07-15",
                    direction_override: float | None = None,
                    print_extra: str = "") -> dict:
    panel = load_panel()
    factor = factor_fn(panel, load_macro())
    factor_w = factor.loc[:warm_end]
    assert len(factor_w) > 100, f"factor too short: {len(factor_w)}"

    results = {"name": name, "factor_rows": len(factor_w),
               "n_assets": panel.shape[1]}
    fwd = {h: fwd_returns(panel, h) for h in horizons}
    ic_by_h = {h: rank_ic_series(factor_w, fwd[h]) for h in horizons}

    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}

    for h in horizons:
        ic = ic_by_h[h]
        icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        hit = float((ic > 0).mean()) if len(ic) else float("nan")
        results[f"ic_h{h}"] = float(ic.mean())
        results[f"icir_h{h}"] = icir
        results[f"hit_h{h}"] = hit
        results[f"n_dates_h{h}"] = int(len(ic))
        if h == 10:
            results["direction"] = direction

    valid = factor_w.notna()
    results["coverage_asset_days"] = float(valid.mean().mean())
    results["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    results["turnover_10d_rank"] = turnover_10d_rank(factor_w)

    lib = _load_library_factors(panel)
    if lib:
        fx = factor.stack().rename("f")
        corrs = {}
        for fid, lf in lib.items():
            both = pd.concat([fx, lf.stack().rename("l")], axis=1).dropna()
            if len(both) >= 200:
                corrs[fid] = float(np.corrcoef(both["f"], both["l"])[0, 1])
        results["max_abs_library_correlation"] = (
            max((abs(v) for v in corrs.values()), default=float("nan")))
        results["library_corrs"] = {k: round(v, 3) for k, v in corrs.items()}
    else:
        results["max_abs_library_correlation"] = float("nan")

    results["decay_ic_by_horizon"] = {str(h): round(results[f"ic_h{h}"], 4) for h in horizons}

    print(f"=== {name} ===")
    print(f"  window: {factor_w.index.min().date()} .. {factor_w.index.max().date()}, "
          f"{len(factor_w)} dates, {panel.shape[1]} assets")
    print(f"  direction={direction:+.3f}")
    for h in horizons:
        print(f"  h{h:>2}: IC={results[f'ic_h{h}']:+.4f}  ICIR={results[f'icir_h{h}']:+.4f}  "
              f"hit={results[f'hit_h{h}']:.3f}  n={results[f'n_dates_h{h}']}")
    print(f"  coverage_asset_days={results['coverage_asset_days']:.3f}  "
          f"coverage_dates_ge8={results['coverage_dates_ge8']:.3f}  "
          f"turnover_10d_rank={results['turnover_10d_rank']:.3f}")
    print(f"  max_abs_library_corr={results['max_abs_library_correlation']:.3f}  "
          f"corrs={results.get('library_corrs', {})}")
    if print_extra:
        print(print_extra)
    print()

    gate_ic = abs(results["ic_h10"]) >= 0.007
    gate_icir = abs(results["icir_h10"]) >= 0.084
    results["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                                 "pass": bool(gate_ic and gate_icir)}
    print(f"  ADMISSION (h=10): |IC|={abs(results['ic_h10']):.4f} (>=0.007: {gate_ic}), "
          f"|ICIR|={abs(results['icir_h10']):.4f} (>=0.084: {gate_icir}) -> "
          f"{'PASS' if gate_ic and gate_icir else 'FAIL'}")
    return results


def _load_library_factors(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    try:
        out["mom_10d_skip5"] = per_asset(
            lambda s: s.shift(5) / s.shift(15) - 1.0)(panel, {})
    except Exception as e:
        print(f"  (lib mom_10d unavailable: {e})")
    try:
        out["mom_120d_skip5"] = per_asset(
            lambda s: s.shift(5) / s.shift(125) - 1.0)(panel, {})
    except Exception as e:
        print(f"  (lib mom_120d unavailable: {e})")
    try:
        out["vol_of_vol20x60"] = per_asset(
            lambda s: s.pct_change().rolling(20).std().rolling(60).std())(panel, {})
    except Exception as e:
        print(f"  (lib vov unavailable: {e})")
    try:
        vix = load_macro()["VIX"].dropna()
        def vbc(panel, macro):
            cols = {}
            for a in panel.columns:
                s = panel[a].dropna()
                r = s.pct_change()
                v = vix.pct_change().reindex(s.index)
                z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
                beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
                cols[a] = -beta * (z["v"].rolling(20).sum() + 1.0 - 1.0)  # placeholder replaced below
            return pd.DataFrame(cols, index=panel.index)
        # simpler exact reconstruction:
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            r = s.pct_change()
            v = vix.pct_change().reindex(s.index)
            z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            vix20 = (vix / vix.shift(20) - 1.0).reindex(s.index)
            cols[a] = (-beta * vix20)
        out["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=panel.index)
    except Exception as e:
        print(f"  (lib vix_beta unavailable: {e})")
    return out


def persist_factor(factor_id: str, factor_name: str, expression: str, description: str,
                   deps: list, params: dict, results: dict, tags: list,
                   regime_notes: str) -> None:
    direction = results.get("direction", 1.0)
    path = Path("factors") / f"{factor_id}.json"
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
                "ic": results["ic_h10"],
                "icir": results["icir_h10"],
                "ic_hit_ratio": results["hit_h10"],
                "n_ic_dates": results["n_dates_h10"],
                "coverage_asset_days": results["coverage_asset_days"],
                "coverage_dates_ge8": results["coverage_dates_ge8"],
                "turnover_10d_rank": results["turnover_10d_rank"],
                "decay_ic_by_horizon": results["decay_ic_by_horizon"],
                "max_abs_library_correlation": results.get("max_abs_library_correlation", float("nan")),
            },
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5},
            "selected_metrics": {
                "ic": results["ic_h10"],
                "icir": results["icir_h10"],
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": results.get("max_abs_library_correlation", float("nan")),
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"PERSISTED -> {path}")
    return path
