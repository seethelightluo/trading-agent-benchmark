import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current five-factor screener ensemble; all directions are positive.
FACTOR_WEIGHTS = {"clv": 0.283, "peer": 0.218, "reversal": 0.168, "momentum": 0.165, "shock": 0.166}
MIN_W, MAX_W, REBALANCE_DAYS = 0.015, 0.16, 10
last_date = None

def ranks(values):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    if len(good) > 1:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / len(good)
    return out

def bounded_weights(raw):
    w = {s: max(0.0, float(raw.get(s, 0.0))) for s in UNIVERSE}
    z = sum(w.values()) or 1.0
    w = {s: x / z for s, x in w.items()}
    # Box-constrained simplex projection, preserving full investment.
    fixed = set()
    for _ in range(50):
        low = [s for s in UNIVERSE if s not in fixed and w[s] < MIN_W]
        high = [s for s in UNIVERSE if s not in fixed and w[s] > MAX_W]
        if not low and not high:
            break
        for s in low:
            w[s], fixed = MIN_W, fixed | {s}
        for s in high:
            w[s], fixed = MAX_W, fixed | {s}
        free = [s for s in UNIVERSE if s not in fixed]
        if not free:
            break
        rem = 1.0 - sum(w[s] for s in fixed)
        z = sum(max(w[s], 1e-12) for s in free)
        for s in free:
            w[s] = rem * w[s] / z
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global last_date
    account = get_account_dict()
    market = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 25:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        market[s] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(market) < 12:
        return
    decision = max(v[4] for v in market.values())
    if last_date is not None:
        try:
            elapsed = (np.datetime64(decision, "D") - np.datetime64(last_date, "D")) / np.timedelta64(1, "D")
            if elapsed < REBALANCE_DAYS:
                return
        except Exception:
            return

    clv, peer, reversal, momentum, shock, invvol, five = {}, {}, {}, {}, {}, {}, {}
    for s, (c, h, l, r, _) in market.items():
        if len(c) < 22:
            continue
        rng = np.maximum(h - l, 1e-12)
        clv[s] = float(np.mean((2.0 * c[-3:] - h[-3:] - l[-3:]) / rng[-3:]))
        reversal[s] = float(-np.mean(r[-5:]))
        vol = max(float(np.std(r[-20:])), 0.008)
        invvol[s] = 1.0 / vol
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01))
        five[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
        # Contrarian response to an unusually large completed-day shock.
        shock[s] = float(-r[-1] * min(abs(r[-1]) / (np.std(r[-20:]) + 0.002), 3.0))
    median5 = float(np.median(list(five.values())))
    peer = {s: v - median5 for s, v in five.items()}
    ranked = {k: ranks(v) for k, v in (("clv", clv), ("peer", peer),
                                        ("reversal", reversal), ("momentum", momentum),
                                        ("shock", shock))}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranked[k][s] for k in ranked) for s in UNIVERSE}

    # Medium-high risk, sideways/mildly bearish posture: full investment remains,
    # but defensive tradable benchmarks receive a modest tilt.
    bearish = False
    if "SPX" in market:
        c = market["SPX"][0]
        bearish = c[-1] < c[-21] and c[-1] < c[-6]
    if bearish:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += 0.18
        for s in ("BTC", "ETH", "WTI"):
            score[s] = max(0.05, score[s] - 0.10)
    mean_iv = float(np.mean(list(invvol.values()))) if invvol else 1.0
    raw = {s: max(0.01, score[s]) * (0.78 + 0.22 * invvol.get(s, mean_iv) / mean_iv) for s in UNIVERSE}
    weights = bounded_weights(raw)
    if set(weights) == set(UNIVERSE) and all(np.isfinite(v) and v >= 0 for v in weights.values()):
        rebalance_to_weights(weights)
        last_date = decision

