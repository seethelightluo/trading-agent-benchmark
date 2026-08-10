"""Shared factor validation framework for miner_1.

Loads 15 tradable assets + 5 observation-only macro signals from ../persistent,
restricts data to the simulator-visible window (<= 2026-07-29), computes factor
panels, and evaluates cross-sectional rank IC / ICIR / hit ratio / coverage /
turnover / decay against forward returns.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("../persistent")
VISIBLE_THROUGH = "2026-07-29"

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

_caches = {}


def load_close(name, kind="stock"):
    key = (name, kind)
    if key in _caches:
        return _caches[key]
    path = DATA / ("stock_data" if kind == "stock" else "index_data") / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].reset_index(drop=True)
    df = df.set_index("date").sort_index()
    out = df["close"].astype(float).rename(name)
    _caches[key] = out
    return out


def load_all():
    closes = {a: load_close(a, "stock") for a in WATCH}
    macros = {m: load_close(m, "index") for m in MACRO}
    return closes, macros


def factor_panel(fn, warmup=260):
    """Compute factor values for all assets. fn(close_series) -> factor series."""
    closes, macros = load_all()
    out = {}
    for a, c in closes.items():
        try:
            f = fn(c, a, macros)
            if f is not None and len(f) > warmup:
                out[a] = f
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {a} factor failed: {exc}")
    return pd.DataFrame(out).dropna(how="all")


def forward_returns(closes, horizon=10):
    """Forward return over horizon trading days for each asset."""
    fwd = {}
    for a, c in closes.items():
        fwd[a] = c.shift(-horizon) / c - 1.0
    return pd.DataFrame(fwd)


def daily_rank_ic(factor_df, fwd_df):
    """Cross-sectional Spearman IC between factor value and forward return per date."""
    common = factor_df.index.intersection(fwd_df.index)
    ics, dates = [], []
    for dt in common:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) >= 8:  # at least 8 valid instruments per date
            ic = pair["f"].rank().corr(pair["r"].rank())
            if np.isfinite(ic):
                ics.append(ic)
                dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")


def summarize_ic(ic_series, label="", verbose=True):
    """Full IC summary with ICIR, hit ratio, coverage."""
    if len(ic_series) == 0:
        return None
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic_series > 0).mean())
    n = len(ic_series)
    if verbose:
        print(f"[{label}] n_ic_dates={n} IC={ic_mean:+.4f} ICIR={icir:+.4f} "
              f"hit={hit:.3f} std={ic_std:.4f}")
    return {"ic": ic_mean, "icir": icir, "ic_hit_ratio": hit,
            "n_ic_dates": n, "ic_std": ic_std}


def coverage_metrics(factor_df):
    n_asset_days = factor_df.notna().sum().sum()
    total = factor_df.shape[0] * factor_df.shape[1]
    cov_asset_days = n_asset_days / total if total else 0.0
    dates_ge8 = float((factor_df.notna().sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": round(cov_asset_days, 3),
            "coverage_dates_ge8": round(dates_ge8, 3)}


def turnover_metric(factor_df, horizon=10):
    """Mean absolute change in cross-sectional rank quantiles at rebalance lag."""
    ranks = factor_df.rank(axis=1, pct=True)
    diff = ranks.diff(horizon).abs()
    return float(diff.mean().mean()) if len(diff) else None


def decay_profile(factor_df, closes, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, h)
        ic = daily_rank_ic(factor_df, fwd)
        out[str(h)] = round(float(ic.mean()), 4) if len(ic) else None
    return out


def library_correlation(factor_df, library_dir="factors"):
    """Max |corr| between this factor's daily mean IC proxy and library factors.

    Uses real signal artifacts: cross-sectional mean factor value per date vs
    each library factor's reconstructed artifact (if recoverable).
    """
    best = 0.0
    factor_mean = factor_df.mean(axis=1)
    for p in Path(library_dir).glob("*.json"):
        if p.name.startswith("factor_ensemble"):
            continue
        try:
            meta = json.loads(p.read_text())
            expr = str(meta.get("calculation", {}).get("expression", ""))
            if "close.shift" in expr and meta.get("parameters", {}).get("skip") is not None:
                # momentum-style artifact
                lb = int(meta["parameters"]["lookback"])
                sk = int(meta["parameters"]["skip"])
                artifact = {}
                closes, _ = load_all()
                for a, c in closes.items():
                    artifact[a] = c.shift(sk) / c.shift(sk + lb) - 1.0
                art = pd.DataFrame(artifact).mean(axis=1)
                common = factor_mean.index.intersection(art.index)
                if len(common) > 30:
                    rho = float(factor_mean.loc[common].corr(art.loc[common]))
                    best = max(best, abs(rho))
        except Exception:  # noqa: BLE001
            continue
    return round(best, 4)


def run_full_validation(fn, factor_id, horizon=10, warmup=260, label=None,
                        regime_split=True):
    """End-to-end validation: factor -> IC summary -> coverage -> turnover -> decay."""
    label = label or factor_id
    closes, macros = load_all()
    fdf = factor_panel(fn, warmup=warmup)
    print(f"[{label}] panel shape={fdf.shape} dates={len(fdf)} assets={fdf.shape[1]}")
    fwd = forward_returns(closes, horizon)
    ic = daily_rank_ic(fdf, fwd)
    summ = summarize_ic(ic, label)
    if summ is None:
        print(f"[{label}] NO VALID IC OBSERVATIONS")
        return None
    cov = coverage_metrics(fdf)
    tov = turnover_metric(fdf, horizon)
    decay = decay_profile(fdf, closes)
    maxcorr = library_correlation(fdf)
    summ.update(cov)
    summ["turnover_10d_rank"] = round(tov, 3) if tov is not None else None
    summ["decay_ic_by_horizon"] = decay
    summ["max_abs_library_correlation"] = maxcorr
    if regime_split:
        split = pd.Timestamp("2023-06-01")
        for name, sub in [("early", ic[ic.index < split]), ("late", ic[ic.index >= split])]:
            if len(sub) > 30:
                m = float(sub.mean()); s = float(sub.std(ddof=1))
                print(f"    regime[{name}] n={len(sub)} IC={m:+.4f} ICIR={m/s if s>0 else 0:+.4f}")
    return summ
