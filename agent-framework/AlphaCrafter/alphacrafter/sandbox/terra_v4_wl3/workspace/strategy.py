import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble: four admitted factors, all directions positive.
FACTOR_WEIGHTS = {"clv": 0.28, "peer": 0.31, "reversal": 0.14, "momentum": 0.27}
MIN_WEIGHT, MAX_WEIGHT = 0.015, 0.16
REBALANCE_DAYS = 10
last_decision = None
previous_score = None


def rank_cross_section(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    n = len(valid)
    for i, (s, _) in enumerate(valid):
        out[s] = (i + 1.0) / n if n > 1 else 0.5
    return out


def project_weights(raw):
    # Iterative bounded-simplex projection; always returns a complete 15-asset vector.
    raw = {s: max(float(raw.get(s, 0.0)), 1e-8) for s in UNIVERSE}
    free, fixed = set(UNIVERSE), {}
    for _ in range(40):
        if not free:
            break
        scale = (1.0 - sum(fixed.values())) / max(sum(raw[s] for s in free), 1e-12)
        low = [s for s in free if raw[s] * scale < MIN_WEIGHT]
        high = [s for s in free if raw[s] * scale > MAX_WEIGHT]
        if not low and not high:
            fixed.update({s: raw[s] * scale for s in free})
            free.clear()
            break
        chosen, bound = (low, MIN_WEIGHT) if low else (high, MAX_WEIGHT)
        for s in chosen:
            fixed[s] = bound
            free.remove(s)
    if free:
        rem = max(1.0 - sum(fixed.values()), 0.0)
        denom = max(sum(raw[s] for s in free), 1e-12)
        fixed.update({s: rem * raw[s] / denom for s in free})
    total = sum(fixed.values())
    return {s: fixed.get(s, 0.0) / max(total, 1e-12) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_decision, previous_score
    series = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=100)
        if df is None or len(df) < 35:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        series[symbol] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(series) < 12:
        return
    decision_date = max(v[4] for v in series.values())
    if last_decision is not None:
        try:
            if (np.datetime64(decision_date) - np.datetime64(last_decision)) / np.timedelta64(1, "D") < REBALANCE_DAYS:
                return
        except Exception:
            return

    clv, peer, reversal, momentum, invvol = {}, {}, {}, {}, {}
    for s, (c, h, l, r, _) in series.items():
        candle = (2.0 * c - h - l) / np.maximum(h - l, 1e-12)
        vol = max(float(np.std(r[-20:])), 0.008)
        clv[s] = float(np.mean(candle[-5:]))
        reversal[s] = float(-np.mean(r[-5:]))
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01))
        peer[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
        invvol[s] = 1.0 / vol
    peer_median = float(np.median(list(peer.values())))
    peer = {s: v - peer_median for s, v in peer.items()}
    ranked = {k: rank_cross_section(v) for k, v in (("clv", clv), ("peer", peer), ("reversal", reversal), ("momentum", momentum))}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranked[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}
    # 60/40 historical/current smoothing controls CLV and reversal whipsaw.
    if previous_score is not None:
        score = {s: 0.60 * previous_score.get(s, score[s]) + 0.40 * score[s] for s in UNIVERSE}
    previous_score = dict(score)

    breadth = float(np.mean([c[-1] > c[-21] for c, *_ in series.values()]))
    bearish = False
    if "SPX" in series:
        spx = series["SPX"][0]
        bearish = spx[-1] < spx[-21] and breadth < 0.50
    if bearish:
        # High-risk bearish regime: remain fully invested but rotate to defensive tradables.
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += 0.20
        for s in ("BTC", "ETH", "WTI"):
            score[s] = max(0.05, score[s] - 0.10)

    mean_iv = float(np.mean(list(invvol.values())))
    raw = {}
    for s in UNIVERSE:
        iv_ratio = invvol.get(s, mean_iv) / max(mean_iv, 1e-12)
        raw[s] = max(score[s], 0.02) * (0.70 + 0.30 * iv_ratio)
    weights = project_weights(raw)
    avg, sd = float(np.mean(list(score.values()))), max(float(np.std(list(score.values()))), 1e-12)
    forecast = {s: 0.01 * (score[s] - avg) / sd for s in UNIVERSE}
    try:
        ensemble = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        factor_ids = [x["factor_id"] for x in ensemble.get("selected_factors", []) if isinstance(x, dict) and x.get("factor_id")][:10]
    except Exception:
        factor_ids = []
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=10)
    last_decision = decision_date

if __name__ == "__main__":
    pass
