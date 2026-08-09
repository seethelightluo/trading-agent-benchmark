import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

# Four admitted factors, with directions preserved from the screener ensemble.
UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"clv": 0.3395, "peer": 0.2612, "reversal": 0.2012, "momentum": 0.1981}
MIN_W, MAX_W = 0.015, 0.16
REBALANCE_EVERY = 10
sessions = 0


def cross_rank(values):
    out = {s: 0.5 for s in UNIVERSE}
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1.0) / len(valid)
    return out


def project_weights(raw):
    # Iterative box/simplex projection: complete, long-only, and diversified.
    w = {s: max(float(raw.get(s, 0.0)), 1e-9) for s in UNIVERSE}
    fixed = set()
    for _ in range(40):
        free = [s for s in UNIVERSE if s not in fixed]
        scale = (1.0 - sum(w[s] for s in fixed)) / max(sum(w[s] for s in free), 1e-12)
        trial = {s: w[s] * scale for s in free}
        viol = [s for s in free if trial[s] < MIN_W or trial[s] > MAX_W]
        if not viol:
            for s in free:
                w[s] = trial[s]
            break
        for s in viol:
            w[s] = MIN_W if trial[s] < MIN_W else MAX_W
            fixed.add(s)
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global sessions
    sessions += 1
    if sessions > 1 and (sessions - 1) % REBALANCE_EVERY != 0:
        return

    # The data API exposes only completed observations at each decision.
    market = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=85)
        if df is None or len(df) < 25:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
        ret = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
        if len(close) >= 22:
            market[symbol] = (close, high, low, ret)
    if len(market) < 12:
        return

    clv, peer, reversal, momentum, invvol, five_day = ({}, {}, {}, {}, {}, {})
    for symbol, (close, high, low, ret) in market.items():
        # The admitted CLV factor is negative close-location value (reversal).
        loc = (2.0 * close - high - low) / np.maximum(high - low, 1e-12)
        clv[symbol] = -float(np.mean(loc[-3:]))
        reversal[symbol] = -float(np.mean(ret[-5:]))
        vol = max(float(np.std(ret[-20:])), 0.008)
        invvol[symbol] = 1.0 / vol
        momentum[symbol] = float((close[-1] / max(close[-21], 1e-12) - 1.0) / (vol + 0.01))
        five_day[symbol] = float(close[-1] / max(close[-6], 1e-12) - 1.0)

    median_5d = float(np.median(list(five_day.values())))
    peer = {s: value - median_5d for s, value in five_day.items()}
    ranked = {name: cross_rank(values) for name, values in (
        ("clv", clv), ("peer", peer), ("reversal", reversal), ("momentum", momentum))}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranked[k][s] for k in ranked) for s in UNIVERSE}

    # Bullish/sideways stays fully invested in the scored blend. In a confirmed
    # SPX downtrend, rotate toward tradable defensives rather than cash.
    if "SPX" in market:
        spx = market["SPX"][0]
        if spx[-1] < spx[-21] and spx[-1] < spx[-6]:
            for symbol in ("XAU", "US10Y", "CN10Y"):
                score[symbol] += 0.18
            for symbol in ("BTC", "ETH", "WTI"):
                score[symbol] = max(0.05, score[symbol] - 0.10)

    mean_invvol = max(float(np.mean(list(invvol.values()))), 1e-12)
    raw = {s: max(score[s], 0.01) * (0.78 + 0.22 * invvol.get(s, mean_invvol) / mean_invvol)
           for s in UNIVERSE}
    weights = project_weights(raw)
    # Atomic full-investment rebalance; quantities remain fractional in the API.
    rebalance_to_weights(weights)
