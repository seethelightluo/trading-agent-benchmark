"""Trader strategy: Screener quality-IC 4-factor ensemble (admitted 2026-08-11).

Cross-sectional composite across all 15 tradable benchmark assets; long-only,
fully invested (no cash sleeve). One target submitted daily via
rebalance_to_weights; the benchmark helper gates cadence (10d), turnover cost
(3bp one-way) and gross edge. Defensive tilt expressed through weights, never cash.
"""
from math import isfinite
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    get_index_daily_data, rebalance_to_weights,
                                    register_hook)

N_DAYS = 300            # covers mom_120d (shift 125) + buffers
VIX_BETA_WIN = 60       # vix_beta_cond_60x20 beta window
DEF = {"XAU", "US10Y", "CN10Y"}   # defensive tradable assets for bear tilt


def stock(a, n=N_DAYS):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None


def index(a, n=N_DAYS):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None


def rank_series(values, assets):
    """Cross-sectional rank in [0,1]; missing values get neutral 0.5."""
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out


def load_ensemble():
    for p in (Path(__file__).parent / "factor_ensemble.json",
              Path(__file__).parent / "factors" / "factor_ensemble.json"):
        try:
            ens = json.loads(p.read_text())
            factors = [dict(f) for f in ens.get("selected_factors", [])
                       if isinstance(f, dict) and f.get("factor_id")]
            if factors:
                return factors
        except (OSError, ValueError, TypeError):
            continue
    return []


def compute_raw_factors(closes, vix_close, assets):
    """Return {factor_id: {asset: raw value}} on the last completed bar."""
    fids = ["mom_10d_skip5", "mom_120d_skip5", "vix_beta_cond_60x20", "vol_of_vol20x60"]
    raw = {fid: {} for fid in fids}
    vix_ret = vix_close.pct_change() if vix_close is not None else None
    for a in assets:
        c = closes.get(a)
        if c is None or len(c) < 140:
            for fid in fids:
                raw[fid][a] = None
            continue
        ret = c.pct_change()
        s5, s15, s125 = c.shift(5), c.shift(15), c.shift(125)
        mom10 = (s5 / s15 - 1.0).iloc[-1]
        mom120 = (s5 / s125 - 1.0).iloc[-1]
        vov = ret.rolling(20).std().rolling(60).std().iloc[-1]
        vb = None
        if vix_ret is not None:
            z = pd.concat([ret.rename("a"), vix_ret.rename("v")], axis=1).dropna().tail(VIX_BETA_WIN)
            var = float(z["v"].var())
            beta = float(z["a"].cov(z["v"]) / var) if len(z) >= 30 and var > 1e-14 else None
            vix_move = (vix_close / vix_close.shift(20) - 1.0).iloc[-1] if len(vix_close) > 21 else None
            vb = (-beta * vix_move) if (beta is not None and vix_move is not None
                                        and isfinite(vix_move)) else None
        raw["mom_10d_skip5"][a] = float(mom10) if isfinite(mom10) else None
        raw["mom_120d_skip5"][a] = float(mom120) if isfinite(mom120) else None
        raw["vol_of_vol20x60"][a] = float(vov) if isfinite(vov) else None
        raw["vix_beta_cond_60x20"][a] = vb
    return raw


def regime_from_market(panel):
    """Bull / sideways / bear from risk-adjusted 20d cross-asset drift."""
    if len(panel) < 30:
        return "sideways"
    mkt = panel.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 if v20 and v20 > 1e-12 else 0.0
    if trend > 0.25:
        return "bull"
    if trend < -0.25:
        return "bear"
    return "sideways"


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    frames = {a: stock(a) for a in assets}
    closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
              for a, f in frames.items()}
    usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
    fallback = {a: 1.0 / len(assets) for a in assets}
    if len(usable) < 8:
        rebalance_to_weights(fallback, forecast_returns={a: 0.0 for a in assets},
                             horizon_days=10)
        return

    panel = pd.concat(usable, axis=1, join="inner")
    factors = load_ensemble()
    if not factors:
        rebalance_to_weights(fallback, forecast_returns={a: 0.0 for a in assets},
                             horizon_days=10)
        return
    factor_ids = [f["factor_id"] for f in factors][:10]

    vf = index("VIX")
    vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
    raw = compute_raw_factors(closes, vix_close, assets)

    # Composite score: sum of weight * direction * centered rank.
    score = {a: 0.0 for a in assets}
    for f in factors:
        fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
        r = rank_series(raw.get(fid, {}), assets)
        for a in assets:
            score[a] += (w * d) * (r[a] - 0.5)

    # Regime-conditional concentration + defensive overlay.
    regime = regime_from_market(panel)
    K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
    lo = min(score.values())
    span = max(max(score.values()) - lo, 1e-9)
    raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
    top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
    w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
    if sum(w.values()) < 1e-9:
        w = {a: (1.0 / K if a in top else 0.0) for a in assets}
    if regime != "bull":
        overlay = 0.25 if regime == "bear" else 0.10
        def_w = {a: (1.0 / len(DEF) if a in DEF else 0.0) for a in assets}
        w = {a: (1.0 - overlay) * w.get(a, 0.0) + overlay * def_w[a] for a in assets}
    total = sum(w.values())
    weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
               for a in assets}
    rem = 1.0 - sum(weights.values())
    weights[assets[0]] += rem  # floating-point exactness on sum-to-one

    # Deterministic forecast returns: z-scored composite * cross-sectional vol.
    score_mean = sum(score.values()) / len(assets)
    score_std = (sum((x - score_mean) ** 2 for x in score.values()) / len(assets)) ** 0.5
    ret_scale = float(panel.tail(252).std(axis=1, ddof=0).median()) if len(panel) else 0.01
    if not isfinite(ret_scale) or ret_scale <= 0:
        ret_scale = 0.01
    forecast_returns = {a: ((score[a] - score_mean) / max(score_std, 1e-12)) * ret_scale
                        for a in assets}

    rebalance_to_weights(weights, forecast_returns=forecast_returns,
                         factor_ids=factor_ids, horizon_days=10)
