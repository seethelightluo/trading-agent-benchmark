"""miner_1 shared validation harness (2027-03-19 cycle).
Loads panel from scripts/panel_cache.pkl (built through 2027-03-18).
- weekday-only calendar (BTC/ETH weekends dropped to align cross-section)
- require >= 8 valid assets per row
- daily rank IC vs forward returns at multiple horizons
- full-sample + recent-window metrics, coverage, turnover, library correlation
"""
import base64
import json
import zlib

import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_panel():
    with open("scripts/panel_cache.pkl", "rb") as fh:
        p = pd.read_pickle(fh)
    C = p["close"].copy()
    C = C[C.index.dayofweek < 5]
    C = C[C.notna().sum(axis=1) >= 8]
    C = C.ffill()
    return C


def load_macro():
    with open("scripts/panel_cache.pkl", "rb") as fh:
        p = pd.read_pickle(fh)
    M = p["macro"].copy()
    M = M[M.index.dayofweek < 5]
    M = M.ffill()
    return M


def build_library_factors(close):
    """Reconstruct simple library factors for correlation audit."""
    lib = {
        "rev_1d": -(close.pct_change()),
        "rev_2d": -(close / close.shift(2) - 1),
        "rev_3d": -(close / close.shift(3) - 1),
        "rev_5d": -(close / close.shift(5) - 1),
        "mom_10d_skip5": close.shift(5) / close.shift(15) - 1,
        "mom_120d_skip5": close.shift(5) / close.shift(125) - 1,
        "trend_20d": close / close.rolling(20).mean() - 1,
        "trend_60d": close / close.rolling(60).mean() - 1,
        "vol_of_vol20x60": close.pct_change().rolling(20).std().rolling(60).std(),
    }
    return lib


def daily_rank_ic(factor, fwd_ret, min_obs=8):
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
    return {"ic": round(float(mean_ic), 5), "icir": round(float(icir), 5),
            "ic_hit_ratio": round(hit, 4), "n_ic_dates": int(n), "ic_std": round(float(std_ic), 5)}


def turnover_10d(factor, horizon_days=10):
    rank = factor.rank(axis=1)
    chg = rank.diff(horizon_days).abs()
    return float(chg.mean().mean())


def library_corr(factor, lib, min_obs=8):
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
    return {"format": "base64:zlib:csv", "description": "Factor signal panel: rows = dates, cols = assets",
            "columns": list(df.columns), "shape": list(df.shape),
            "n_valid_values": int(df.notna().sum().sum()),
            "sha256": str(abs(hash(csv)) % 10 ** 16), "data": b64}


def run_validation(factor_panel, close, horizons=(1, 2, 3, 5, 10, 20),
                   factor_id="", regime_notes="", return_summary=False, recent_split=250):
    results = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        ic_series = daily_rank_ic(factor_panel, fwd)
        results[h] = summarize(ic_series, f"h={h}")
        if recent_split and results[h] and len(ic_series) > recent_split:
            results[h]["recent_" + str(recent_split)] = summarize(ic_series.iloc[-recent_split:], f"h={h} recent")

    print(f"=== {factor_id} ===")
    for h, r in results.items():
        if r:
            extra = ""
            if "recent_" + str(recent_split) in r:
                rr = r["recent_" + str(recent_split)]
                extra = f"  RECENT{recent_split}: IC={rr['ic']:+.5f} ICIR={rr['icir']:+.5f} hit={rr['ic_hit_ratio']:.3f}"
            print(f"  h={h:2d}  IC={r['ic']:+.5f}  ICIR={r['icir']:+.5f}  hit={r['ic_hit_ratio']:.3f}  n={r['n_ic_dates']}{extra}")

    admitted = None
    for h in (1, 2, 3, 5, 10):
        r = results[h]
        if r and abs(r["ic"]) >= GATE_IC and abs(r["icir"]) >= GATE_ICIR:
            if admitted is None or abs(r["icir"]) > abs(results[admitted]["icir"]):
                admitted = h
    if admitted is not None:
        r = results[admitted]
        print(f"  >> ADMISSION horizon h={admitted}: IC={r['ic']:+.5f} ICIR={r['icir']:+.5f} (gates {GATE_IC}/{GATE_ICIR})")
    else:
        print(f"  >> NO admission (gates {GATE_IC}/{GATE_ICIR})")

    cov_assets = float(factor_panel.notna().mean().mean())
    cov_dates_ge8 = float(factor_panel.notna().sum(axis=1).ge(8).mean())
    to10 = turnover_10d(factor_panel)
    lib = build_library_factors(close)
    lc = library_corr(factor_panel, lib)
    max_abs_lc = max((abs(v) for v in lc.values() if v is not None), default=0.0)
    print(f"  coverage_asset_days={cov_assets:.3f} coverage_dates_ge8={cov_dates_ge8:.3f} turnover_10d_rank={to10:.3f}")
    print(f"  library_corr={lc}  max_abs={max_abs_lc:.4f}")

    summary = {"factor_id": factor_id,
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
               "regime_notes": regime_notes}
    if return_summary:
        return summary
    return None
