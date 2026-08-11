"""Shared validation harness for miner_3 (2026-10-08 cycle, fixed 2026-11-13).

Calendar fix: BTC/ETH trade on weekends, so a naive outer join of the 15
series contains weekend dates where weekday-only assets are NaN, which poisons
rolling windows. We therefore:
  1. drop weekend rows from the panel (simulator marks positions on trading
     days only; decisions use previous completed trading day),
  2. require >= 8 valid assets per row (early period has fewer because
     BTC/ETH data starts 2021-04),
  3. forward-fill per-asset closes inside the remaining calendar so holiday
     gaps carry the last price (standard mixed-calendar convention) -- this
     makes rolling windows well-defined with full coverage.

- Computes daily cross-sectional rank IC for given factor panels vs forward
  returns at multiple horizons.
- Reports IC, ICIR, hit ratio, coverage, turnover, decay, and
  max_abs_library_correlation vs reconstructable library factors.
- Dumps signal artifact (base64:zlib:csv) for gate re-computation.
"""
import base64
import json
import zlib

import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_close_panel(days=2400, ffill=True, min_assets=8):
    """Common-calendar close panel for the 15 tradable instruments."""
    frames = {}
    for sym in TRADABLE:
        df = get_stock_daily_data(symbol=sym, days=days)
        if df is None or len(df) == 0:
            continue
        s = df.set_index("date")["close"].rename(sym)
        frames[sym] = s
    panel = pd.concat(frames, axis=1).sort_index()
    # drop weekends (BTC/ETH trade them; weekday assets do not)
    panel = panel[panel.index.dayofweek < 5]
    # keep rows where the cross-section is usable
    panel = panel[panel.notna().sum(axis=1) >= min_assets]
    if ffill:
        panel = panel.ffill()
    return panel


def build_library_factors(close):
    """Reconstruct simple library factors for correlation audit."""
    ret = close.pct_change()
    lib = {
        "rev_1d": -ret.shift(0),            # reversal 1d (direction -1 on past ret)
        "rev_2d": -(close / close.shift(2) - 1),
        "rev_3d": -(close / close.shift(3) - 1),
        "mom_10d_skip5": close.shift(5) / close.shift(15) - 1,
        "mom_120d_skip5": close.shift(5) / close.shift(125) - 1,
        "trend_20d": close / close.rolling(20).mean() - 1,
        "trend_60d": close / close.rolling(60).mean() - 1,
    }
    return lib


def daily_rank_ic(factor, fwd_ret, min_obs=8):
    """Daily cross-sectional Spearman IC between factor and forward return."""
    dates, ics = [], []
    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_obs:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize(ic_series, name):
    ics = ic_series.dropna()
    n = len(ics)
    if n == 0:
        return None
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())
    return {
        "factor": name,
        "ic": round(float(mean_ic), 5),
        "icir": round(float(icir), 5),
        "ic_hit_ratio": round(hit, 4),
        "n_ic_dates": int(n),
        "ic_std": round(float(std_ic), 5),
    }


def turnover_10d(factor, horizon_days=10):
    """Mean absolute change in cross-sectional rank over 10 trading days."""
    rank = factor.rank(axis=1)
    chg = rank.diff(horizon_days).abs()
    return float(chg.mean().mean())


def library_corr(factor, lib, min_obs=8):
    """Mean daily cross-sectional Spearman corr of candidate vs library factors."""
    out = {}
    for name, lf in lib.items():
        corrs = []
        for dt in factor.index:
            if dt not in lf.index:
                continue
            f, g = factor.loc[dt], lf.loc[dt]
            m = f.notna() & g.notna()
            if m.sum() < min_obs:
                continue
            c = f[m].rank().corr(g[m].rank())
            if np.isfinite(c):
                corrs.append(c)
        out[name] = round(float(np.mean(corrs)), 4) if corrs else None
    return out


def make_artifact(factor_panel):
    df = factor_panel.copy()
    df.index = df.index.strftime("%Y-%m-%d")
    csv = df.to_csv()
    comp = zlib.compress(csv.encode("utf-8"))
    b64 = base64.b64encode(comp).decode("ascii")
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates, cols = assets",
        "columns": list(df.columns),
        "shape": list(df.shape),
        "n_valid_values": int(df.notna().sum().sum()),
        "sha256": str(abs(hash(csv)) % 10**16),
        "data": b64,
    }


def run_validation(factor_panel, close, horizons=(1, 2, 3, 5, 10, 20),
                   factor_id="", regime_notes="", return_summary=False):
    """Full validation routine; prints metrics and returns summary dict."""
    results = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        ic_series = daily_rank_ic(factor_panel, fwd)
        results[h] = summarize(ic_series, f"h={h}")

    print(f"=== {factor_id} ===")
    for h, r in results.items():
        if r:
            print(f"  h={h:2d}  IC={r['ic']:+.5f}  ICIR={r['icir']:+.5f}  "
                  f"hit={r['ic_hit_ratio']:.3f}  n={r['n_ic_dates']}")

    # pick admission horizon: largest abs(ICIR) among h<=10 with abs(IC)>=gate
    gate_ic, gate_icir = 0.0070, 0.0840
    admitted = None
    for h in (1, 2, 3, 5, 10):
        r = results[h]
        if r and abs(r["ic"]) >= gate_ic and abs(r["icir"]) >= gate_icir:
            if admitted is None or abs(r["icir"]) > abs(results[admitted]["icir"]):
                admitted = h
    if admitted is not None:
        r = results[admitted]
        print(f"  >> ADMISSION horizon h={admitted}: IC={r['ic']:+.5f} "
              f"ICIR={r['icir']:+.5f} (gates {gate_ic}/{gate_icir})")
    else:
        print(f"  >> NO admission (gates {gate_ic}/{gate_icir})")

    cov_assets = float(factor_panel.notna().mean().mean())
    cov_dates_ge8 = float(factor_panel.notna().sum(axis=1).ge(8).mean())
    to10 = turnover_10d(factor_panel)
    lib = build_library_factors(close)
    lc = library_corr(factor_panel, lib)
    max_abs_lc = max((abs(v) for v in lc.values() if v is not None), default=0.0)
    print(f"  coverage_asset_days={cov_assets:.3f} coverage_dates_ge8={cov_dates_ge8:.3f} "
          f"turnover_10d_rank={to10:.3f}")
    print(f"  library_corr={lc}  max_abs={max_abs_lc:.4f}")

    summary = {
        "factor_id": factor_id,
        "metrics_by_horizon": {str(h): r for h, r in results.items() if r},
        "admission_horizon": admitted,
        "admission": results[admitted] if admitted is not None else None,
        "coverage_asset_days": round(cov_assets, 4),
        "coverage_dates_ge8": round(cov_dates_ge8, 4),
        "turnover_10d_rank": round(to10, 4),
        "library_corr": lc,
        "max_abs_library_correlation": round(max_abs_lc, 4),
        "n_assets": int(factor_panel.shape[1]),
        "n_dates": int(factor_panel.shape[0]),
        "regime_notes": regime_notes,
    }
    if return_summary:
        return summary
    return None
