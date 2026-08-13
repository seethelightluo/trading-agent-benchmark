"""Trader strategy v39 (2031-06-30):
v38 + five risk-adjustment fires from the 06-16..06-30 block (proposal on
06-16 EXECUTED, cost = 06-13 closes; account 1005704.93 -> 1008222.37,
+0.25% block):
  - WTI x0.65 -> x0.50  LARGE AIR-POCKET -10.45% (06-16 close -> 06-27 close)
    immediately after the v38 re-boost x0.50->x0.65 (mirror v36 chain:
    x0.65->x0.50 on -14.55%; re-boost failed on first block).
  - SX5E x0.85 -> x0.70  2nd consecutive negative block (-0.47%, -5.35%).
  - XAU x1.00 -> x0.85   2nd consecutive negative block (-4.34%, -1.01%).
  - NDX x0.65 -> x0.75   2 consecutive positive blocks (+3.14%, +2.52%).
  - US10Y x0.60 -> x0.70 2 consecutive positive blocks (+0.80%, +4.74%)
    (re-boost after v37-lite x0.60 cut).
  Kept: SOX x0.35 (single pos +5.97% after neg), SPX x0.65 (single neg -1.70%),
  000688.SH x0.70 (single neg -1.66%), N225 x0.85 (+5.73% pos), CN10Y x0.70
  frozen, COPPER cap 0.12 (single pos +4.64% after -5.08%).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y.

v38 (2031-06-16):
v37-lite + single risk adjustment WTI x0.50 -> x0.65 (RE-BOOST fired 2031-06-16:
2 consecutive positive blocks after the v36 large-air-pocket cut x0.65->x0.50
(2030-12-02): 05-05..05-19 +1.4k, 06-02..06-16 +8.70% WTI; mirror v27 precedent
WTI re-boost after 2 stable blocks). All other v37-lite multipliers kept.

v37-lite (2031-02-24 fired, restored 2031-03-10):
v36 base + single risk adjustment US10Y x0.70 -> x0.60 only (3 consecutive
negative blocks -1.39%/-1.46%/-5.86% incl large print + r20 -7.24% bond
selloff; SPX/000688 evidence mixed after the 12-30..01-13 recovery, kept at
v36 levels). Ensemble read live from factors/factor_ensemble.json.
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
CAP = 0.12          # per-asset weight cap (v10: 0.13 -> 0.12; v11 kept)
FLOOR = 0.5 / N_ASSETS
SPREAD = 0.06       # max score-driven spread above floor before vol tilt (v8: 0.08 -> 0.06)
MIN_OBS = 40        # min obs for 60d beta factors
CORR_WIN = 20       # vol-price corr window
CORR_MIN = 10       # min obs for corr
VOL_EXP = 0.6       # inverse-vol exponent (v5: 0.5 -> 0.6)
STALE_N = 5         # consecutive identical closes => stale quote
# v27 defensive multipliers (fired 2029-04-09 after the 03-26..04-09 block +1.35%):
#   BTC x0.30 -> x0.25 (7th cons neg incl -6.50%), US10Y x1.25 -> x1.00 (2 cons neg
#   -3.86%/-2.01%), CN10Y x0.60 -> x0.70 (2 cons pos +2.46%/+17.90% mean-reversion)
# Kept: XAU x1.00 (+5.29% pos), SOX x0.45 (+10.80% pos), NDX x0.75 (single -4.54%),
#   WTI x0.80 (mild -0.48% after +28.26%), SPX x0.85 (single -2.01%), 000688.SH x0.70
#   (single +0.81%), SX5E x1.00 (single -2.86%), ETH x0.75 (frozen)
# Prior v26 (2029-03-26): BTC x0.35->x0.30, SOX x0.55->x0.45, SPX x0.65->x0.85,
#   WTI x0.65->x0.80, 000688 x0.85->x0.70
# Prior v25 (2029-02-12): XAU x0.85->x1.00, SX5E x0.85->x1.00, NDX x0.65->x0.75,
#   BTC x0.40->x0.35, CN10Y x0.70->x0.60
# Prior v24 (2028-11-06). Prior v23 (2028-10-23):
#   XAU x1.00 -> x0.85 (6th consecutive neg), BTC x0.55 -> x0.45 (2nd cons neg),
#   SPX x0.85 -> x0.65 (3rd consecutive neg)
# Prior v22 (2028-10-09): SPX x1.00 -> x0.85 (2nd cons neg), WTI x0.65 -> x0.80
#   (re-boost after 2 stable blocks)
# Prior v20 (2028-09-11): XAU x1.25 -> x1.00 (3 cons neg), WTI x0.80 -> x0.65
#   (2 consecutive large air-pockets)
# Prior v19 (2028-08-28): BTC x0.65 -> x0.55, CN10Y x1.00 -> x0.85,
#   US10Y x1.00 -> x1.25
# Prior v17 (2028-06-05): XAU x1.00 -> x1.25, US10Y x1.25 -> x1.00,
#   BTC x0.75 -> x0.65, SX5E x1.00 -> x0.85
DEFENSIVE_MULT = {
    "XAU": 0.85, "US10Y": 0.70, "CN10Y": 0.70,   # safe havens (v39 fired 2031-06-30: XAU x1.00->x0.85 2nd cons neg -4.34%/-1.01%; US10Y x0.60->x0.70 RE-BOOST 2 cons pos +0.80%/+4.74%; CN10Y kept x0.70 frozen stale)
    "SOX": 0.35, "NDX": 0.75, "ETH": 0.75, "WTI": 0.50, "BTC": 0.15, "N225": 0.85,  # high-beta (v39: WTI x0.65->x0.50 LARGE AIR-POCKET -10.45% after failed v38 re-boost, mirror v36; NDX x0.65->x0.75 2 cons pos +3.14%/+2.52%; SOX kept x0.35 single pos +5.97%; BTC/ETH kept frozen stale; N225 kept x0.85 pos +5.73%)
    "SX5E": 0.70, "SPX": 0.65, "000688.SH": 0.70,  # v39: SX5E x0.85->x0.70 2nd cons neg -0.47%/-5.35%; SPX kept x0.65 single neg -1.70% after pos; 000688 kept x0.70 single neg -1.66% after pos
}


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


def vol_adj_mom_accel(c, fast=20, slow=60, vol_win=20):
    """(close/close.shift(20)-1 - (close/close.shift(60)-1)) / rolling_std(ret,20).
    Positive = recently accelerating per unit risk; last valid value."""
    r = c.pct_change()
    mom_fast = c / c.shift(fast) - 1.0
    mom_slow = c / c.shift(slow) - 1.0
    vol = r.rolling(vol_win).std()
    z = (mom_fast - mom_slow) / vol
    z = z.dropna()
    if len(z) < 1:
        return None
    val = float(z.iloc[-1])
    return val if isfinite(val) else None



def compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol):
    """Raw cross-sectional factor values for the 3-factor ensemble.
    Stale assets get no entries -> neutral rank 0.5 downstream."""
    mkt = panel.mean(axis=1)
    dn_x = mkt.clip(upper=0.0)
    vals = {
        "vol_price_corr_20": {},
        "eurusd_beta_60d": {},
        "rate_beta_cn10y_60d": {},
        "dn_mkt_beta_60d": {},
        "vol_adj_mom_accel_20x60": {},
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
        c2 = closes.get(a)
        if c2 is not None:
            ma = vol_adj_mom_accel(c2)
            if ma is not None:
                vals["vol_adj_mom_accel_20x60"][a] = ma
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

    # base weight: floor + score-driven spread (v10: defensive multiplier applied
    # here, before the inverse-vol tilt; weights re-normalized downstream)
    base = {a: FLOOR + SPREAD * ((score[a] - lo) / (hi - lo + 1e-12)) for a in assets}
    for a, m in DEFENSIVE_MULT.items():
        if a in base:
            base[a] *= m
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
