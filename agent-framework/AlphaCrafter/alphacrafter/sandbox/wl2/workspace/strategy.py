import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = {"clv": 0.3378, "leadlag": 0.2599, "reversal": 0.2003, "momentum": 0.2020}
MIN_WEIGHT, MAX_WEIGHT = 0.015, 0.16
_day = 0


def rank_cs(values):
    valid = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    result = {s: 0.5 for s in ASSETS}
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            result[s] = (i + 1.0) / len(valid)
    return result


def bounded_weights(raw):
    # Project positive scores to a complete simplex with practical concentration caps.
    w = {s: max(float(raw.get(s, 0.01)), 1e-8) for s in ASSETS}
    total = sum(w.values())
    w = {s: v / total for s, v in w.items()}
    fixed = {}
    for _ in range(40):
        free = [s for s in ASSETS if s not in fixed]
        remain = 1.0 - sum(fixed.values())
        base = sum(w[s] for s in free)
        hit = False
        for s in free:
            x = remain * w[s] / base
            if x < MIN_WEIGHT:
                fixed[s] = MIN_WEIGHT; hit = True
            elif x > MAX_WEIGHT:
                fixed[s] = MAX_WEIGHT; hit = True
        if not hit:
            for s in free:
                w[s] = remain * w[s] / base
            break
    w.update(fixed)
    z = sum(w.values())
    return {s: w[s] / z for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    account = get_account_dict()
    positions = account.get("positions", [])
    # One proposal per ten-trading-day block; retry initial allocation if needed.
    if positions and (_day - 1) % 10 != 0:
        return

    series = {}
    for symbol in ASSETS:
        df = get_stock_daily_data(symbol=symbol, days=90)
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        if len(c) >= 22:
            series[symbol] = (c, h, l, r, max(float(np.std(r[-20:])), 0.008))
    if len(series) < 12:
        return

    clv, leadlag, reversal, momentum, invvol = {}, {}, {}, {}, {}
    for s, (c, h, l, r, vol) in series.items():
        day_range = max(float(h[-1] - l[-1]), 1e-12)
        clv[s] = (2.0 * c[-1] - h[-1] - l[-1]) / day_range
        leadlag[s] = c[-1] / max(c[-6], 1e-12) - 1.0
        reversal[s] = -float(np.mean(r[-5:]))
        momentum[s] = (c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01)
        invvol[s] = 1.0 / vol
    median_leadlag = float(np.median(list(leadlag.values())))
    leadlag = {s: v - median_leadlag for s, v in leadlag.items()}
    ranked = {k: rank_cs(v) for k, v in {
        "clv": clv, "leadlag": leadlag, "reversal": reversal, "momentum": momentum
    }.items()}
    score = {s: sum(FACTORS[k] * ranked[k][s] for k in FACTORS) for s in ASSETS}

    # Moderate bearish regime: remain fully invested but rotate toward tradable defensives.
    if "SPX" in series:
        spx = series["SPX"][0]
        bearish = spx[-1] < spx[-6] and spx[-1] < spx[-21]
        if bearish:
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += 0.16
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(0.02, score[s] - 0.08)

    mean_invvol = float(np.mean(list(invvol.values())))
    raw = {s: max(score[s], 0.01) * (0.80 + 0.20 * invvol[s] / mean_invvol) for s in ASSETS}
    rebalance_to_weights(bounded_weights(raw))
