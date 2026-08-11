"""Trader strategy v3: Screener 6-factor ensemble (quality-IC tilt, 2026-08-13).

Ensemble from factors/factor_ensemble.json:
  eurusd_beta_60d      w=0.214 dir=-1  low EURUSD-beta tilt (risk-appetite hedge)
  rate_beta_cn10y_60d  w=0.201 dir=-1  low CN10Y-beta tilt (rate-hedge)
  dn_mkt_beta_60d      w=0.195 dir=+1  low downside-market-beta (safe-haven)
  mom_120d_skip5       w=0.180 dir=+1  120d momentum, skip 5d
  mom_10d_skip5        w=0.121 dir=+1  10d momentum, skip 5d
  vix_beta_cond_60x20  w=0.089 dir=-1  conditional VIX-beta * 20d VIX move

Full-investment long-only 15-asset cross-sectional strategy; non-negative
weights sum to 1 (cash=0). Rebalance cadence 10 trading days (handled by
rebalance_to_weights horizon_days). Bearish views expressed by defensive tilt
(XAU/US10Y/CN10Y + low-beta assets), never by cash or shorts.
"""
from math import isfinite
import json
from pathlib import Path

import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
    rebalance_to_weights,
    register_hook,
)

N_ASSETS = 15
CAP = 0.18          # per-asset weight cap
FLOOR = 0.5 / N_ASSETS
SPREAD = 0.14       # max score-driven spread above floor before vol tilt
MIN_OBS = 40        # min obs for 60d beta factors


def stock(a, n=170):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None


def index(a, n=170):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None


def ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out


def rolling_beta(y, x, win=60, min_obs=MIN_OBS):
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(z) < min_obs:
        return None
    var = float(z.x.var())
    if var <= 1e-14:
        return None
    return float(z.y.cov(z.x) / var)


def load_ensemble():
    try:
        raw = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        return [(str(it["factor_id"]), float(it["weight"]), int(it.get("direction", 1)))
                for it in raw.get("selected_factors", [])
                if isinstance(it, dict) and it.get("factor_id")]
    except (OSError, ValueError, TypeError):
        return []


def compute_factor_values(assets, closes, panel, eurusd_ret, vix_close):
    """Raw cross-sectional factor values for the 6-factor ensemble."""
    mkt = panel.mean(axis=1)
    dn_x = mkt.clip(upper=0.0)
    vix_ret = vix_close.pct_change() if vix_close is not None else None
    vix_move = (float(vix_close.iloc[-1] / vix_close.iloc[-21] - 1.0)
                if vix_close is not None and len(vix_close) >= 22 else None)
    cn10y_ret = closes["CN10Y"].pct_change() if closes.get("CN10Y") is not None else None

    f = {fid: {} for fid, _, _ in []}  # placeholder
    vals = {
        "mom_120d_skip5": {},
        "mom_10d_skip5": {},
        "dn_mkt_beta_60d": {},
        "eurusd_beta_60d": {},
        "rate_beta_cn10y_60d": {},
        "vix_beta_cond_60x20": {},
    }
    for a in assets:
        c = closes.get(a)
        if c is not None and len(c) >= 126:
            vals["mom_120d_skip5"][a] = float(c.iloc[-6] / c.iloc[-126] - 1.0)
        if c is not None and len(c) >= 16:
            vals["mom_10d_skip5"][a] = float(c.iloc[-6] / c.iloc[-16] - 1.0)
        y = panel[a]
        vals["dn_mkt_beta_60d"][a] = rolling_beta(y, dn_x)
        if eurusd_ret is not None:
            vals["eurusd_beta_60d"][a] = rolling_beta(y, eurusd_ret)
        if cn10y_ret is not None:
            vals["rate_beta_cn10y_60d"][a] = rolling_beta(y, cn10y_ret)
        if vix_ret is not None and vix_move is not None:
            b = rolling_beta(y, vix_ret)
            vals["vix_beta_cond_60x20"][a] = -b * vix_move if b is not None else None
    return vals


def capped_normalize(w, cap=CAP):
    """Normalize weights to sum 1, then water-fill cap at `cap`.

    Normalize first so the cap comparison is meaningful, then iteratively clip
    over-cap assets and redistribute the excess proportionally among the
    under-cap assets until convergence.
    """
    w = {a: max(0.0, float(x)) for a, x in w.items()}
    total = sum(w.values())
    if total <= 0:
        return {a: 1.0 / len(w) for a in w}
    w = {a: x / total for a, x in w.items()}
    for _ in range(200):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess < 1e-12:
            break
        clipped = {a: min(cap, x) for a, x in w.items()}
        room = [a for a, x in clipped.items() if x < cap - 1e-12]
        if not room:
            w = clipped
            break
        room_total = sum(w[a] for a in room)
        if room_total <= 0:
            w = clipped
            break
        for a in room:
            clipped[a] += excess * (w[a] / room_total)
        w = clipped
    total = sum(w.values())
    if abs(total - 1.0) > 1e-9 and total > 0:
        w = {a: x / total for a, x in w.items()}
    return w


def compute_target(assets):
    """Return (weights, forecast_returns, factor_ids, info)."""
    frames = {a: stock(a) for a in assets}
    closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
              for a, f in frames.items()}
    usable = [c.pct_change().rename(a) for a, c in closes.items() if c is not None and len(c) >= 30]
    panel = (pd.concat(usable, axis=1, join="inner").dropna().tail(130)
             if len(usable) >= 8 else pd.DataFrame())
    if len(panel) < 50:
        return ({a: 1.0 / len(assets) for a in assets},
                {a: 0.0 for a in assets}, [], {"fallback": "short_panel"})

    ef = index("EURUSD")
    eurusd_ret = (ef.close.astype(float).pct_change()
                  if ef is not None and "close" in ef else None)
    vf = index("VIX")
    vix_close = (vf.close.astype(float) if vf is not None and "close" in vf else None)

    ens = load_ensemble()
    factor_ids = [fid for fid, _, _ in ens]
    if not factor_ids:
        # defensive fallback: slight safe-haven tilt, zero forecast
        w = {a: 1.0 / len(assets) for a in assets}
        for a in ("XAU", "US10Y", "CN10Y"):
            if a in w:
                w[a] += 0.02
        w = capped_normalize(w, cap=0.16)
        return (w, {a: 0.0 for a in assets}, [], {"fallback": "no_ensemble"})

    vals = compute_factor_values(assets, closes, panel, eurusd_ret, vix_close)

    # composite score = sum(weight * direction * rank)
    score = {a: 0.0 for a in assets}
    for fid, wgt, drc in ens:
        r = ranks(vals.get(fid, {}), assets)
        for a in assets:
            score[a] += wgt * drc * r[a]

    s_vals = [score[a] for a in assets]
    lo, hi = min(s_vals), max(s_vals)
    vol20 = {a: max(float(panel[a].tail(20).std()), 0.004) for a in assets}

    # base weight: floor + score-driven spread; then inverse-vol tilt
    base = {a: FLOOR + SPREAD * ((score[a] - lo) / (hi - lo + 1e-12)) for a in assets}
    tilted = {a: base[a] / (vol20[a] ** 0.5) for a in assets}
    weights = capped_normalize(tilted)

    # forecast returns (10-day proxy): z-scored score * typical 10d cross-sectional vol
    mean_s = sum(s_vals) / len(s_vals)
    std_s = (sum((v - mean_s) ** 2 for v in s_vals) / len(s_vals)) ** 0.5 or 1e-12
    scale = float(panel.tail(60).std(axis=1, ddof=0).median()) * (10.0 ** 0.5) or 0.01
    forecast_returns = {}
    for a in assets:
        z = max(-2.5, min(2.5, (score[a] - mean_s) / std_s))
        forecast_returns[a] = z * scale
    return weights, forecast_returns, factor_ids[:10], {
        "scores": {a: round(float(score[a]), 4) for a in assets},
        "scale": round(float(scale), 5),
        "vol20": {a: round(float(v), 4) for a, v in vol20.items()},
    }


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    weights, forecast_returns, factor_ids, info = compute_target(assets)
    rebalance_to_weights(
        weights,
        forecast_returns=forecast_returns,
        factor_ids=factor_ids,
        horizon_days=10,
    )


if __name__ == "__main__":
    import sys
    from alphacrafter.sim.utils import get_account_dict
    _assets = list(get_account_dict()["watch_list"])
    _w, _f, _ids, _info = compute_target(_assets)
    print("factor_ids:", _ids)
    print("info:", json.dumps(_info, indent=1)[:1500])
    print("weights sum:", round(sum(_w.values()), 6))
    for _a in _assets:
        print(f"  {_a:10s} w={_w[_a]:.4f} f={_f[_a]:+.5f}")
