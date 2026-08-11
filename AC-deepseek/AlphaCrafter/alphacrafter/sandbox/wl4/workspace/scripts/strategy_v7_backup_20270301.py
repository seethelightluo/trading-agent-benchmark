"""Trader strategy v7: defensive re-weight on top of v4 ensemble.

Ensemble from factors/factor_ensemble.json (unchanged, quality-IC tilt):
  vol_price_corr_20     w=0.3851 dir=+1  volume-confirmed price moves
  dn_mkt_beta_60d       w=0.2445 dir=+1  low downside-market-beta (safe-haven)
  eurusd_beta_60d       w=0.2062 dir=-1  low EURUSD-beta tilt (risk-appetite hedge)
  rate_beta_cn10y_60d   w=0.1642 dir=-1  low CN10Y-beta tilt (rate-hedge)

v5 changes (triggered 2026-11-09 after 3 consecutive negative live blocks):
  1. Inverse-vol exponent 0.5 -> 0.6 (stronger vol dampening)
  2. Per-asset cap 0.18 -> 0.15 (lower concentration)
  3. Stale-quote guard: assets with STALE_N consecutive identical closes are
     excluded from factor computation (neutral rank 0.5), dropped from the
     market panel, and assigned cross-sectional median vol20. Prevents frozen
     quotes (HSI/ETH since ~2026-10-14) from distorting betas / inflating
     low-vol weights.

v6 changes (triggered 2027-01-18 after a negative block with risk-on stall:
SOX -6.2%, BTC -5.1%, COPPER -4.8%, WTI -3.1% in the 01-04..01-18 block):
  1. SPREAD 0.14 -> 0.10 (flatter score-driven dispersion; lowers single-name
     conviction and reduces migration turnover between similar-score assets).

v7 changes (triggered 2027-03-01 after the 02-15..03-01 negative block: the
pre-agreed N225 trigger fired - factor score stayed high while price fell
another -6.7%; 000688.SH also dropped -13.8% on a high score; COPPER +13.9%
despite a negative forecast => high-score concentration is the main risk):
  1. SPREAD 0.10 -> 0.08 (further flatten score-driven spread; reduces weight
     on top-scored names so a single air pocket damages the book less).
  Kept: vol exp 0.6, cap 0.15, stale-quote guard, full-investment 15-asset
  cross-section, 10-trading-day cadence.

Full-investment long-only 15-asset cross-sectional strategy; non-negative
weights sum to 1 (cash=0). Rebalance cadence 10 trading days (handled by
rebalance_to_weights horizon_days). Bearish views expressed by defensive tilt,
never by cash or shorts.
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
CAP = 0.15          # per-asset weight cap (v5: 0.18 -> 0.15)
FLOOR = 0.5 / N_ASSETS
SPREAD = 0.08       # max score-driven spread above floor before vol tilt (v7: 0.10 -> 0.08)
MIN_OBS = 40        # min obs for 60d beta factors
CORR_WIN = 20       # vol-price corr window
CORR_MIN = 10       # min obs for corr
VOL_EXP = 0.6       # inverse-vol exponent (v5: 0.5 -> 0.6)
STALE_N = 5         # consecutive identical closes => stale quote


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


def is_stale(closes, a, n=STALE_N):
    c = closes.get(a)
    if c is None or len(c) < n + 2:
        return False
    tail = c.astype(float).iloc[-n - 1:]
    return bool((tail.diff().dropna().abs() < 1e-12).all())


def compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol):
    """Raw cross-sectional factor values for the 4-factor ensemble.
    Stale assets get no entries -> neutral rank 0.5 downstream."""
    mkt = panel.mean(axis=1)
    dn_x = mkt.clip(upper=0.0)
    vals = {
        "vol_price_corr_20": {},
        "eurusd_beta_60d": {},
        "rate_beta_cn10y_60d": {},
        "dn_mkt_beta_60d": {},
    }
    for a in assets:
        if a in stale:
            continue
        c = closes.get(a)
        v = vol.get(a)
        if c is not None and v is not None:
            z = pd.concat([c.pct_change().rename("r"), v.astype(float).rename("v")],
                          axis=1).dropna().tail(CORR_WIN)
            if len(z) >= CORR_MIN:
                _c = z.r.corr(z.v)
                if _c is not None and isfinite(float(_c)):
                    vals["vol_price_corr_20"][a] = float(_c)
        y = panel[a]
        vals["dn_mkt_beta_60d"][a] = rolling_beta(y, dn_x)
        if eurusd_ret is not None:
            vals["eurusd_beta_60d"][a] = rolling_beta(y, eurusd_ret)
        if cn10y_ret is not None:
            vals["rate_beta_cn10y_60d"][a] = rolling_beta(y, cn10y_ret)
    return vals


def capped_normalize(w, cap=CAP):
    """Normalize weights to sum 1, then water-fill cap at `cap`."""
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
    vol = {a: (f.volume.astype(float) if f is not None and "volume" in f else None)
           for a, f in frames.items()}

    # v5 stale-quote guard
    stale = {a for a in assets if is_stale(closes, a)}
    live = [a for a in assets if a not in stale]

    usable = [c.pct_change().rename(a) for a, c in closes.items()
              if a in live and c is not None and len(c) >= 30]
    panel = (pd.concat(usable, axis=1, join="inner").dropna().tail(130)
             if len(usable) >= 8 else pd.DataFrame())
    if len(panel) < 50:
        return ({a: 1.0 / len(assets) for a in assets},
                {a: 0.0 for a in assets}, [], {"fallback": "short_panel",
                                               "stale": sorted(stale)})

    ef = index("EURUSD")
    eurusd_ret = (ef.close.astype(float).pct_change()
                  if ef is not None and "close" in ef else None)
    cn10y_ret = (closes["CN10Y"].pct_change() if closes.get("CN10Y") is not None else None)

    ens = load_ensemble()
    factor_ids = [fid for fid, _, _ in ens]
    if not factor_ids:
        # defensive fallback: slight safe-haven tilt, zero forecast
        w = {a: 1.0 / len(assets) for a in assets}
        for a in ("XAU", "US10Y", "CN10Y"):
            if a in w:
                w[a] += 0.02
        w = capped_normalize(w, cap=0.16)
        return (w, {a: 0.0 for a in assets}, [], {"fallback": "no_ensemble",
                                                  "stale": sorted(stale)})

    vals = compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol)

    # composite score = sum(weight * direction * rank)
    score = {a: 0.0 for a in assets}
    for fid, wgt, drc in ens:
        r = ranks(vals.get(fid, {}), assets)
        for a in assets:
            score[a] += wgt * drc * r[a]

    s_vals = [score[a] for a in assets]
    lo, hi = min(s_vals), max(s_vals)
    vol20 = {a: max(float(panel[a].tail(20).std()), 0.004) for a in live}
    if vol20:
        med_vol = sorted(vol20.values())[len(vol20) // 2]
    else:
        med_vol = 0.01
    for a in stale:
        vol20[a] = med_vol  # frozen quotes: neutral average risk

    # base weight: floor + score-driven spread; then inverse-vol tilt (v5 exp 0.6)
    base = {a: FLOOR + SPREAD * ((score[a] - lo) / (hi - lo + 1e-12)) for a in assets}
    tilted = {a: base[a] / (vol20[a] ** VOL_EXP) for a in assets}
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
        "stale": sorted(stale),
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
    from alphacrafter.sim.utils import get_account_dict
    _assets = list(get_account_dict()["watch_list"])
    _w, _f, _ids, _info = compute_target(_assets)
    print("factor_ids:", _ids)
    print("info:", json.dumps(_info, indent=1)[:1500])
    print("weights sum:", round(sum(_w.values()), 6))
    for _a in _assets:
        print(f"  {_a:10s} w={_w[_a]:.4f} f={_f[_a]:+.5f}")
