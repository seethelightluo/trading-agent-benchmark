"""Screener ensemble strategy, trader 2026-11-05 refresh.

Cross-sectional factor ensemble (quality_ic_tilt) drives a fully-invested
15-asset long-only target. One proposal per 10-trading-day block (first day
only); the rebalance helper applies the 3bp gate.

Ensemble (2026-11-05): down_beta_60(+1) cn10y_beta_60(-1) spx_beta_60(+1)
vol_adj_mom_20_60(+1) dxy_beta_cond_60x20(+1) hs300_beta_60(-1)
hilo_vol_ratio_20(+1) intraday_ret_skew_20(+1) comm_basket_beta_60(+1)
vol_of_vol20x60(+1). vix_beta_cond_60x20 and dd_duration_120_resid dropped.

Weighting: rank-linear tilt * inverse-vol (sqrt dampened), defensive floor,
water-fill cap at 0.18. Sum-to-1, cash 0, fractional quantities.
"""
import json
import math
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

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
DEF = {"XAU", "US10Y", "CN10Y"}
CAP = 0.18
ONLINE_START = "2026-07-16"
HORIZON = 10


def get_df(symbol, days=260):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None


def series(df, col="close"):
    if df is None or col not in df or len(df) < 40:
        return None
    s = df[col].astype(float)
    try:
        s.index = pd.to_datetime(df["date"])
    except Exception:
        s.index = pd.RangeIndex(len(s))
    return s


def beta_last(y, x, win=60, min_obs=20):
    """Rolling-window beta of y on x; last window value."""
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(q) < min_obs:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)


def down_beta(y, x, win=60):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    q = q[q.x < 0].tail(win)
    if len(q) < 20:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)


def cs_rank(values, assets):
    """Cross-sectional rank in [0,1]; missing -> 0.5."""
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and np.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = max(1, len(valid) - 1)
    for i, (_, a) in enumerate(valid):
        out[a] = i / n
    return out


def is_block_start():
    try:
        d = json.load(open("../persistent/date.json"))
        tds = d.get("trading_days", [])
        cur = d.get("current_date")
        if ONLINE_START in tds and cur in tds:
            return (tds.index(cur) - tds.index(ONLINE_START)) % HORIZON == 0
    except Exception:
        pass
    return True


def build_weights(score, assets, panel, def_floor, spread, cap=CAP):
    """Rank-linear tilt * inverse-vol (sqrt), defensive floor, water-fill cap."""
    order = sorted(assets, key=lambda a: (-score[a], a))
    lin = {a: 1.0 - i / max(1, len(order) - 1) for i, a in enumerate(order)}
    vols = {a: max(float(panel[a].tail(20).std()), 0.003) for a in assets}
    vmed = float(np.median([vols[a] for a in assets]))
    pref = {a: (1.0 + spread * lin[a]) * math.sqrt(vmed / vols[a]) for a in assets}

    total = sum(max(0.0, float(x)) for x in pref.values())
    w = {a: max(0.0, float(pref[a])) / total for a in assets}

    # defensive floor (risk posture), then renormalize
    for a in DEF:
        w[a] = max(w[a], def_floor)
    tot = sum(w.values())
    if tot > 0:
        w = {a: x / tot for a, x in w.items()}

    # water-fill cap: cap at `cap`, redistribute excess proportionally to pref
    for _ in range(200):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(cap, x) for a, x in w.items()}
        room = [a for a, x in w.items() if x < cap - 1e-9]
        if not room:
            break
        p = {a: max(0.0, pref.get(a, 0.0)) for a in room}
        den = sum(p.values())
        if den <= 0:
            p = {a: 1.0 for a in room}
            den = len(room)
        for a in room:
            w[a] += excess * p[a] / den

    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


def load_ensemble():
    for path in ("factors/factor_ensemble.json", "factor_ensemble.json"):
        p = Path(__file__).parent / path
        try:
            ens = json.loads(p.read_text())
            sel = ens.get("selected_factors", [])
            if sel:
                return [(str(s["factor_id"]), float(s["weight"]), int(s["direction"]))
                        for s in sel if isinstance(s, dict) and s.get("factor_id")]
        except (OSError, ValueError, TypeError):
            continue
    return []


@register_hook
def strategy_hook():
    if not is_block_start():
        return  # mid-block: no new target; sim marks/processes orders

    assets = list(get_account_dict()["watch_list"])
    frames = {a: get_df(a) for a in assets}
    close = {a: series(frames[a]) for a in assets}
    open_ = {a: series(frames[a], "open") for a in assets}
    if any(c is None for c in close.values()):
        return

    ret = {a: close[a].pct_change() for a in assets}
    panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
    if len(panel) < 70:
        return

    ens = load_ensemble()
    ens_ids = {fid for fid, _, _ in ens}

    r_spx = ret["SPX"]
    r_300 = ret["000300.SH"]
    d_cn = close["CN10Y"].pct_change()
    r_comm = pd.concat(
        [ret["XAU"].rename("XAU"), ret["COPPER"].rename("COPPER"), ret["WTI"].rename("WTI")],
        axis=1, join="inner").mean(axis=1)
    dxy = series(get_df("DXY"))
    vix = series(get_df("VIX"))
    r_dxy = dxy.pct_change() if dxy is not None else None
    r_vix = vix.pct_change() if vix is not None else None

    # ---- factor signals (only those in the active ensemble) --------------
    sig = {fid: {} for fid in ens_ids}
    for a in assets:
        c, o, r = close[a], open_[a], ret[a]
        if "down_beta_60" in ens_ids:
            sig["down_beta_60"][a] = down_beta(r, r_spx)
        if "spx_beta_60" in ens_ids:
            sig["spx_beta_60"][a] = beta_last(r, r_spx)
        if "hs300_beta_60" in ens_ids:
            sig["hs300_beta_60"][a] = beta_last(r, r_300)
        if "cn10y_beta_60" in ens_ids:
            sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
        if "comm_basket_beta_60" in ens_ids:
            sig["comm_basket_beta_60"][a] = beta_last(r, r_comm)
        if "vol_adj_mom_20_60" in ens_ids:
            sig["vol_adj_mom_20_60"][a] = (
                (c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6)
                if len(c) >= 30 else None)
        if "dxy_beta_cond_60x20" in ens_ids:
            if r_dxy is not None:
                b = beta_last(r, r_dxy)
                sig["dxy_beta_cond_60x20"][a] = (
                    b * (dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if b is not None else None)
            else:
                sig["dxy_beta_cond_60x20"][a] = None
        if "vix_beta_cond_60x20" in ens_ids:
            if r_vix is not None:
                b = beta_last(r, r_vix)
                sig["vix_beta_cond_60x20"][a] = (
                    -b * (vix.iloc[-1] / vix.iloc[-21] - 1.0) if b is not None else None)
            else:
                sig["vix_beta_cond_60x20"][a] = None
        if "hilo_vol_ratio_20" in ens_ids:
            hi20 = c.rolling(20).max()
            lo20 = c.rolling(20).min()
            sig["hilo_vol_ratio_20"][a] = (
                float((hi20.iloc[-1] - lo20.iloc[-1]) / max(c.iloc[-1], 1e-9)
                      / max(float(r.tail(20).std()), 1e-6))
                if len(c) >= 25 else None)
        if "intraday_ret_skew_20" in ens_ids:
            if o is not None:
                ir = (c / o - 1.0).dropna().tail(20)
                sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
            else:
                sig["intraday_ret_skew_20"][a] = None
        if "vol_of_vol20x60" in ens_ids:
            rv20 = r.rolling(20).std()
            sig["vol_of_vol20x60"][a] = (
                float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None)

    # ---- composite score (direction preserved) --------------------------
    score = {a: 0.0 for a in assets}
    for fid, w, d in ens:
        rk = cs_rank(sig.get(fid, {}), assets)
        for a in assets:
            score[a] += w * d * rk[a]

    # ---- regime posture ---------------------------------------------------
    market = panel.mean(axis=1)
    wealth = (1.0 + market).cumprod()
    mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
    mkt20 = float(market.tail(20).mean())
    vol20 = float(panel.tail(20).std().mean())
    vol_med = float(panel.tail(120).std().median(axis=0))
    risk_off = (mkt20 < 0.0 and mdd < -0.03) or (vol20 > 1.3 * max(vol_med, 1e-6))
    risk_on = mkt20 > 0.0 and mdd > -0.02
    def_floor = 0.15 if risk_off else (0.10 if risk_on else 0.12)
    spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)

    # ---- target weights: full 15-asset, sum 1, cash 0 --------------------
    weights = build_weights(score, assets, panel, def_floor, spread)

    # ---- forecast returns (z-scored composite) ---------------------------
    vals = np.array([score[a] for a in assets], dtype=float)
    mu, sd = float(vals.mean()), float(vals.std())
    scale = float(panel.tail(252).std(axis=1, ddof=0).median()) if len(panel) >= 30 else 0.01
    if not math.isfinite(scale) or scale <= 0:
        scale = 0.01
    forecast = {a: ((score[a] - mu) / sd) * scale if sd > 1e-12 else 0.0 for a in assets}

    rebalance_to_weights(
        weights,
        forecast_returns=forecast,
        factor_ids=[fid for fid, _, _ in ens][:10],
        horizon_days=HORIZON,
    )
