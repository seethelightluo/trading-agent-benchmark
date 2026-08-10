"""miner_2 shared validation library (per-asset calendar-aware).
Universe: 15 tradable cross-asset instruments.
Factor values computed on each asset's own trading calendar then reindexed to
the union calendar. IC = cross-sectional Spearman rank IC per date (>=8 assets).
Warm-up factor window: 2020-01-01..2026-07-15 (data visible through 2026-07-29).
Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2026-07-29"
FACTOR_LAST = "2026-07-15"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084}
LIB_IDS = ["mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60", "vix_beta_cond_60x20"]


def load_panel(assets: list[str] | None = None) -> pd.DataFrame:
    assets = assets or WATCH
    closes = {}
    for s in assets:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    return panel[~panel.index.duplicated(keep="last")].sort_index()


def load_macro(name: str | None = None) -> dict[str, pd.Series]:
    out = {}
    for m in (MACRO if name is None else [name]):
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[m] = df["close"].astype(float)
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
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    ics = {}
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        if len(r) < MIN_ASSETS:
            continue
        ics[d] = spearmanr(f.reindex(r.index), r)[0]
    return pd.Series(ics).sort_index()


def turnover_10d_rank(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def library_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute the 4 effective library factors for correlation audit."""
    out = {}
    rets = panel.pct_change()
    out["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    out["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    out["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    try:
        vix = load_macro("VIX")["VIX"]
        vixr = vix.pct_change()
        beta = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
        out["vix_beta_cond_60x20"] = -beta * (vix / vix.shift(20) - 1.0)
    except Exception:
        pass
    return out


def library_corr(factor: pd.DataFrame, panel: pd.DataFrame, libs: dict) -> tuple:
    """Mean per-date cross-sectional Spearman corr with each library factor."""
    per = {}
    common = factor.index.intersection(panel.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-700:]:
            if dt not in factor.index or dt not in lf.index:
                continue
            f = factor.loc[dt]
            g = lf.loc[dt]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[-1]
            if isinstance(g, pd.DataFrame):
                g = g.iloc[-1]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def validate_factor(name: str, factor_fn, horizons=(1, 2, 3, 5, 10, 20),
                    direction_override: float | None = None) -> dict:
    panel = load_panel()
    factor = factor_fn(panel, load_macro())
    factor_w = factor.loc[:FACTOR_LAST]
    assert len(factor_w) > 100, f"factor too short: {len(factor_w)}"

    results = {"name": name, "factor_rows": len(factor_w), "n_assets": panel.shape[1]}
    fwd = {h: fwd_returns(panel, h) for h in horizons}
    ic_by_h = {h: rank_ic_series(factor_w, fwd[h]) for h in horizons}

    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}

    for h in horizons:
        ic = ic_by_h[h]
        icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        results[f"ic_h{h}"] = float(ic.mean())
        results[f"icir_h{h}"] = icir
        results[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        results[f"n_dates_h{h}"] = int(len(ic))
        if h == 10:
            results["direction"] = direction
            results["raw_ic_h10"] = float(ic10.mean())

    valid = factor_w.notna()
    results["coverage_asset_days"] = float(valid.mean().mean())
    results["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    results["turnover_10d_rank"] = turnover_10d_rank(factor_w)

    libs = library_signals(panel)
    max_corr, per = library_corr(factor_w, panel, libs)
    results["max_abs_library_correlation"] = max_corr
    results["library_corrs"] = per

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
          f"per={per}")
    gate_ic = abs(results["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(results["icir_h10"]) >= ADMISSION["icir"]
    results["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                                 "pass": bool(gate_ic and gate_icir)}
    print(f"  ADMISSION (h=10): |IC|={abs(results['ic_h10']):.4f} (>=0.007: {gate_ic}), "
          f"|ICIR|={abs(results['icir_h10']):.4f} (>=0.084: {gate_icir}) -> "
          f"{'PASS' if gate_ic and gate_icir else 'FAIL'}")
    print()
    return results


def persist_factor(factor_id: str, factor_name: str, expression: str, description: str,
                   deps: list, params: dict, results: dict, tags: list,
                   regime_notes: str) -> Path:
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
            "contract": {"ic_threshold": ADMISSION["ic"], "icir_threshold": ADMISSION["icir"],
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
