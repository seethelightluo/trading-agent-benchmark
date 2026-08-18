"""Cross-asset factor ensemble trader (online, 15-asset benchmark).

Ensemble (from factor_ensemble.json, screener 2026-07-30):
  - mom_10d_skip5          w=0.6107 dir=+1  (10d momentum, 5d skip)
  - vix_beta_cond_60x20    w=0.2682 dir=-1  (conditional VIX beta * 20d VIX move)
  - yield_beta_cond_60x20  w=0.1211 dir=+1  (conditional yield beta * 20d yield move)

Full-investment, long-only, non-negative 15-asset target weights summing to 1.
Factor values are recomputed live from price data (no stale panels).
Regime overlay tilts toward defensive assets in bear/high-vol conditions.

Data-feed robustness: a few series can be frozen/flat (e.g. no prints for
weeks). Such assets get a cross-sectional fallback volatility instead of a
near-zero realized vol, which would otherwise blow up vol-scaled weights.
Regime statistics (market return, dispersion) are computed on live assets only.
"""
from math import isfinite
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
    rebalance_to_weights,
    register_hook,
)

DEFAULT_ENSEMBLE = [
    {"factor_id": "mom_10d_skip5", "weight": 0.6107, "direction": 1},
    {"factor_id": "vix_beta_cond_60x20", "weight": 0.2682, "direction": -1},
    {"factor_id": "yield_beta_cond_60x20", "weight": 0.1211, "direction": 1},
]

DEF = {"XAU", "US10Y", "CN10Y"}
CAP = 0.22
FLOOR = 0.004
N_DAYS = 200
STALE_VOL_EPS = 1e-6


def series(symbol, days=N_DAYS, index=False):
    try:
        df = (get_index_daily_data(symbol, days=days) if index
              else get_stock_daily_data(symbol, days=days))
    except Exception:
        return None
    if df is None or "close" not in df.columns or len(df) < 70:
        return None
    df = df.sort_values("date")
    return pd.Series(df["close"].astype(float).values,
                     index=pd.DatetimeIndex(df["date"]))


def rolbeta(y, x, win=60, minp=30):
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    beta = pd.Series(np.nan, index=y.index)
    if len(d) >= minp + 5:
        cov = d["y"].rolling(win, min_periods=minp).cov(d["x"])
        var = d["x"].rolling(win, min_periods=minp).var()
        beta = (cov / var).reindex(y.index)
    return beta


def realized_vol(c, win=20):
    r = c.pct_change()
    if len(r) < win + 2:
        return None
    v = float(r.tail(win).std())
    if not isfinite(v):
        return None
    # flat / stale series -> no reliable volatility
    if v < STALE_VOL_EPS:
        return None
    return max(v, 0.004)


def load_ensemble():
    try:
        doc = json.loads((Path(__file__).parent / "factor_ensemble.json").read_text())
        items = doc.get("selected_factors", [])
        if items:
            return [{"factor_id": str(it["factor_id"]),
                     "weight": float(it["weight"]),
                     "direction": int(it.get("direction", 1))} for it in items][:10]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return DEFAULT_ENSEMBLE


def ranks_latest(values, assets):
    valid = [(float(values[a]), a) for a in assets
             if a in values and values[a] is not None and isfinite(float(values[a]))]
    valid.sort()
    out = {a: 0.5 for a in assets}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, n - 1)
    return out


def capped_normalize(w, pref):
    for _ in range(80):
        excess = sum(max(0.0, x - CAP) for x in w.values())
        w = {a: min(CAP, max(0.0, x)) for a, x in w.items()}
        room = [a for a, x in w.items() if x < CAP - 1e-12]
        if excess < 1e-12 or not room:
            break
        den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
        for a in room:
            w[a] += excess * (max(0.0, pref.get(a, 0.0)) / den if den > 0 else 1.0 / len(room))
    total = sum(w.values())
    return {a: x / total if total > 0 else 1.0 / len(w) for a, x in w.items()}


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    closes = {a: series(a) for a in assets}
    usable = [a for a, c in closes.items() if c is not None]
    if len(usable) < 8:
        rebalance_to_weights(
            {a: 1.0 / len(assets) for a in assets},
            forecast_returns={a: 0.0 for a in assets},
            horizon_days=10,
        )
        return

    vix_s = series("VIX", index=True)
    ensemble = load_ensemble()

    # ---- per-asset realized vol with fallback for stale/flat series ----
    vols = {}
    for a, c in closes.items():
        if c is None or len(c) < 25:
            vols[a] = None
            continue
        v = realized_vol(c)
        vols[a] = v
    live_vols = [v for v in vols.values() if v is not None]
    med_vol = float(np.median(live_vols)) if live_vols else 0.015
    for a in assets:
        if vols.get(a) is None:
            vols[a] = med_vol  # stale asset: treat as average-vol

    # ---- factor panels (per asset dense calendar) ----
    mom = {}
    vixb = {}
    yldb = {}
    for a, c in closes.items():
        if c is None or len(c) < 70:
            continue
        r = c.pct_change()
        try:
            mom[a] = float(c.shift(5).iloc[-1] / c.shift(15).iloc[-1] - 1.0)
        except Exception:
            mom[a] = None
        vix_a = vix_s.reindex(c.index).ffill() if vix_s is not None else None
        if vix_a is not None and len(vix_a) >= 70:
            try:
                b = rolbeta(r, vix_a.pct_change()).iloc[-1]
                dv = float(vix_a.iloc[-1] / vix_a.iloc[-21] - 1.0) if len(vix_a) >= 21 else None
                vixb[a] = -float(b) * dv if b is not None and isfinite(b) and dv is not None else None
            except Exception:
                vixb[a] = None
        y_c = closes.get("US10Y")
        if y_c is not None and len(y_c) >= 70:
            try:
                y_a = y_c.reindex(c.index).ffill()
                b = rolbeta(r, y_a.pct_change()).iloc[-1]
                dy = float(y_a.iloc[-1] / y_a.iloc[-21] - 1.0) if len(y_a) >= 21 else None
                yldb[a] = float(b) * dy if b is not None and isfinite(b) and dy is not None else None
            except Exception:
                yldb[a] = None

    # ---- weighted rank composite ----
    panels = {"mom_10d_skip5": mom, "vix_beta_cond_60x20": vixb, "yield_beta_cond_60x20": yldb}
    score = {a: 0.0 for a in assets}
    for f in ensemble:
        fid = f["factor_id"]
        panel = panels.get(fid)
        if panel is None:
            continue
        rk = ranks_latest(panel, assets)
        for a in assets:
            score[a] += f["weight"] * f["direction"] * (rk[a] - 0.5)

    valid_sc = [float(score[a]) for a in usable if isfinite(float(score[a]))]
    if not valid_sc:
        rebalance_to_weights(
            {a: 1.0 / len(assets) for a in assets},
            forecast_returns={a: 0.0 for a in assets},
            horizon_days=10,
        )
        return

    # ---- regime on LIVE assets only (ignore stale/flat) ----
    live = [a for a in usable if realized_vol(closes[a]) is not None]
    if len(live) < 5:
        live = usable
    ret20 = {}
    for a in live:
        c = closes[a]
        if c is not None and len(c) >= 21:
            ret20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0)
    r20v = [x for x in ret20.values() if isfinite(x)]
    mkt20 = float(np.mean(r20v)) if r20v else 0.0
    disp20 = float(np.std(r20v)) if len(r20v) >= 3 else 0.0
    vix_lvl = float(vix_s.iloc[-1]) if vix_s is not None and len(vix_s) else 20.0
    bear = mkt20 < -0.02 or vix_lvl > 28.0
    high_disp = disp20 >= 0.010

    # ---- volatility-scaled preferences + defensive regime tilt ----
    smin, smax = min(valid_sc), max(valid_sc)
    span = (smax - smin) or 1e-9
    pref = {}
    for a in assets:
        sc = (float(score[a]) - smin) / span
        v = max(float(vols.get(a) or med_vol), 0.004)
        pref[a] = (0.55 + 1.35 * sc) / v
        if a in DEF and (bear or high_disp):
            pref[a] *= 1.6

    w = {a: max(FLOOR, pref.get(a, FLOOR)) for a in assets}
    w = capped_normalize(w, pref)
    total = sum(w.values())
    w = {a: x / total for a, x in w.items()}

    # ---- forecast returns (score z-score scaled by vol) ----
    mean = float(np.mean(valid_sc))
    std = float(np.std(valid_sc)) if len(valid_sc) > 1 else 1e-9
    vols_ok = [vols[a] for a in usable if vols.get(a)]
    scale = float(np.median(vols_ok)) if vols_ok else 0.01
    forecast = {}
    for a in assets:
        s = (float(score[a]) - mean) / max(std, 1e-9)
        forecast[a] = s * scale * 3.16  # ~10d horizon scaling

    factor_ids = [f["factor_id"] for f in ensemble]
    rebalance_to_weights(
        w,
        forecast_returns=forecast,
        factor_ids=factor_ids,
        horizon_days=10,
    )