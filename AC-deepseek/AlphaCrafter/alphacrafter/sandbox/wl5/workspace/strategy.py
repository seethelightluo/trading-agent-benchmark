"""Cross-asset ensemble strategy, Trader (v18 ensemble, 2030-11-14).

Ensemble is loaded dynamically from factors/factor_ensemble.json (quality_ic_tilt,
10 active factors <= 10 cap) with fallback to DEFAULT_ENSEMBLE:
  trend_r2_30_signed    w=0.2208 dir=+1  signed 30d log-price trend R2
  semi_down_ratio_20    w=0.1762 dir=-1  downside/upside semi-vol ratio - 1
  cny_beta_60           w=0.1271 dir=+1  60d beta of asset returns to USDCNY
  dxy_beta_60           w=0.1182 dir=+1  60d beta of asset returns to DXY
  mom_120d_skip5        w=0.0910 dir=+1  120d momentum skipping last 5d
  vol_of_vol20x60       w=0.0739 dir=+1  60d std of 20d realized vol
  time_under_water_120  w=0.0661 dir=-1  days since last 120d rolling high
  vix_beta_cond_60x20   w=0.0576 dir=-1  -beta(asset,VIX,60) * 20d VIX move
  mom_10d_skip5         w=0.0350 dir=+1  10d momentum skipping last 5d
  tail_ratio_20         w=0.0341 dir=+1  20d q95/|q05| return-tail asymmetry

Full-investment long-only 15-asset target (cash = 0), one rebalance proposal
per decision via rebalance_to_weights; helper gates on turnover vs 3bp cost.
"""
from math import isfinite, copysign
import json
from pathlib import Path
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights, register_hook,
)

DEFAULT_ENSEMBLE = [
    ("trend_r2_30_signed",   0.1945,  1),
    ("semi_down_ratio_20",   0.1494, -1),
    ("mom_120d_skip5",       0.1489,  1),
    ("dxy_beta_60",          0.1297,  1),
    ("time_under_water_120", 0.0882, -1),
    ("mom_10d_skip5",        0.0851,  1),
    ("tail_ratio_20",        0.0748,  1),
    ("vix_beta_cond_60x20",  0.0659, -1),
    ("vol_of_vol20x60",      0.0635,  1),
]
DEF = {"XAU", "US10Y", "CN10Y"}
MAX_W, MIN_W = 0.18, 0.005
FETCH = 200


def _load_ensemble():
    try:
        p = Path(__file__).parent / "factors" / "factor_ensemble.json"
        ens = json.loads(p.read_text())
        out = []
        for it in ens.get("selected_factors", []):
            if not isinstance(it, dict):
                continue
            fid = str(it.get("factor_id", ""))
            if not fid:
                continue
            try:
                w = float(it.get("weight", 0.0))
                d = int(it.get("direction", 1))
            except (TypeError, ValueError):
                continue
            out.append((fid, w, d))
        if out:
            return out[:10]
    except (OSError, ValueError, TypeError):
        pass
    return list(DEFAULT_ENSEMBLE)


def _closes(assets):
    out = {}
    for a in assets:
        df = None
        try:
            df = get_stock_daily_data(a, days=FETCH)
        except Exception:
            df = None
        if df is None or len(df) < 140:
            try:
                df = get_index_daily_data(a, days=FETCH)
            except Exception:
                df = None
        if df is not None and len(df) >= 140 and "close" in df:
            s = df[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"])
            out[a] = s.set_index("date")["close"].astype(float)
    return out


def _macro_close(symbol):
    df = None
    try:
        df = get_index_daily_data(symbol, days=150)
    except Exception:
        df = None
    if df is None or "close" not in df or len(df) < 80:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)


def _rank_map(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out


def _trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return copysign(cov * cov / (vy * vx), cov)


def _semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0


def _mom_120(c):
    if len(c) < 126:
        return None
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _mom_10(c):
    if len(c) < 17:
        return None
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def _vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not isfinite(out) else float(out)


def _kurt_20(r):
    s = r.dropna().tail(40)
    if len(s) < 20:
        return None
    k = s.rolling(20, min_periods=8).kurt().iloc[-1]
    return None if not isfinite(k) else float(k)


def _tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return None
    return q95 / abs(q05)


def _beta_60(r, m_r):
    """Rolling 60d beta of asset returns to a macro return series (min 30 obs)."""
    z = pd.concat([r.rename("a"), m_r.rename("m")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vm = float(z["m"].var())
    if vm < 1e-14:
        return None
    return float(z["a"].cov(z["m"]) / vm)


def _vix_beta_cond(r, vix_r, vix_c):
    z = pd.concat([r.rename("a"), vix_r.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vv = float(z["v"].var())
    if vv < 1e-14:
        return None
    beta = float(z["a"].cov(z["v"]) / vv)
    if vix_c is None or len(vix_c) < 22:
        return None
    v0 = float(vix_c.iloc[-21])
    if v0 <= 0:
        return None
    vmove = float(vix_c.iloc[-1]) / v0 - 1.0
    return -beta * vmove


def _to_weights(score, assets, regime_w):
    vals = np.array([score[a] for a in assets], dtype=float)
    mu, sd = float(vals.mean()), float(vals.std())
    if sd < 1e-12:
        return {a: 1.0 / len(assets) for a in assets}
    z = (vals - mu) / sd
    w = np.exp(z / 0.85)
    raw = {a: float(w[i]) for i, a in enumerate(assets)}
    total = sum(raw.values())
    if total <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    wts = {a: raw[a] / total for a in assets}
    # regime blend: defensive tilt when market is weak
    for a in assets:
        wts[a] = 0.70 * wts[a] + 0.30 * regime_w.get(a, 1.0 / len(assets))
    # clamp
    for _ in range(60):
        excess = sum(max(0.0, x - MAX_W) for x in wts.values())
        wts = {a: min(MAX_W, max(MIN_W, x)) for a, x in wts.items()}
        room = [a for a in wts if wts[a] < MAX_W - 1e-12]
        if excess < 1e-12 or not room:
            break
        den = sum(max(0.0, regime_w.get(a, 0.0)) for a in room)
        for a in room:
            wts[a] += excess * (regime_w.get(a, 0.0) / den if den else 1.0 / len(room))
    s = sum(wts.values())
    wts = {a: x / s for a, x in wts.items()}
    wts[assets[-1]] += 1.0 - sum(wts.values())
    return wts


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    ensemble = _load_ensemble()
    factor_ids = [fid for fid, _, _ in ensemble]

    closes = _closes(assets)
    if len(closes) < 8:
        eq = {a: 1.0 / len(assets) for a in assets}
        rebalance_to_weights(eq, forecast_returns={a: 0.0 for a in assets},
                             factor_ids=factor_ids, horizon_days=10)
        return

    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change()

    dxy_c = _macro_close("DXY")
    dxy_r = dxy_c.pct_change() if dxy_c is not None else None
    vix_c = _macro_close("VIX")
    vix_r = vix_c.pct_change() if vix_c is not None else None
    cny_c = _macro_close("USDCNY")
    cny_r = cny_c.pct_change() if cny_c is not None else None

    fvals = {fid: {} for fid, _, _ in ensemble}
    for a in assets:
        c = closes.get(a)
        r = rets[a] if a in rets else None
        if c is None or r is None:
            continue
        for fid, _, _ in ensemble:
            try:
                if fid == "trend_r2_30_signed":
                    v = _trend_r2(c)
                elif fid == "semi_down_ratio_20":
                    v = _semi_down_ratio(r)
                elif fid == "mom_120d_skip5":
                    v = _mom_120(c)
                elif fid == "mom_10d_skip5":
                    v = _mom_10(c)
                elif fid == "vol_of_vol20x60":
                    v = _vol_of_vol(r)
                elif fid == "time_under_water_120":
                    v = _underwater(c)
                elif fid == "tail_ratio_20":
                    v = _tail_ratio(r)
                elif fid == "kurt_20":
                    v = _kurt_20(r)
                elif fid == "dxy_beta_60":
                    v = _beta_60(r, dxy_r) if dxy_r is not None else None
                elif fid == "cny_beta_60":
                    v = _beta_60(r, cny_r) if cny_r is not None else None
                elif fid == "vix_beta_cond_60x20":
                    v = _vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None
                else:
                    v = None
            except Exception:
                v = None
            fvals[fid][a] = v

    score = {a: 0.0 for a in assets}
    for fid, w, direction in ensemble:
        rk = _rank_map(fvals[fid], assets)
        for a in assets:
            score[a] += w * direction * rk[a]

    market = rets.mean(axis=1)
    trend20 = float(market.tail(20).mean()) if len(market) >= 20 else 0.0
    avg_px = float(panel.mean(axis=1).iloc[-1]) if len(panel) else 0.0
    ma60 = float(panel.mean(axis=1).tail(60).mean()) if len(panel) >= 60 else avg_px
    bearish = trend20 < 0.0 and avg_px < ma60
    regime_w = {}
    for a in assets:
        if bearish and a in DEF:
            regime_w[a] = 2.2
        elif bearish:
            regime_w[a] = 0.75
        else:
            regime_w[a] = 1.0

    weights = _to_weights(score, assets, regime_w)

    sv = np.array([score[a] for a in assets], dtype=float)
    smu, ssd = float(sv.mean()), float(sv.std())
    return_scale = float(panel.tail(200).pct_change().std(axis=1, ddof=0).median()) if len(panel) >= 40 else 0.01
    if not isfinite(return_scale) or return_scale <= 0:
        return_scale = 0.01
    forecast_returns = {
        a: ((float(score[a]) - smu) / max(ssd, 1e-12)) * return_scale
        for a in assets
    }

    rebalance_to_weights(
        weights,
        forecast_returns=forecast_returns,
        factor_ids=factor_ids[:10],
        horizon_days=10,
    )
