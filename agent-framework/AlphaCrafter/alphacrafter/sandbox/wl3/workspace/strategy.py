import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"clv": 0.3395, "peer": 0.2612, "reversal": 0.2012, "momentum": 0.1981}
REBALANCE_EVERY = 10
MIN_W, MAX_W = 0.02, 0.14
_calls = 0


def rank(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    if len(good) > 1:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / len(good)
    return out


def bounded_weights(raw):
    # Box-constrained simplex projection, preserving a full-investment target.
    w = {s: max(float(raw.get(s, 0.0)), 1e-12) for s in UNIVERSE}
    free = set(UNIVERSE)
    fixed = {}
    for _ in range(20):
        remaining = 1.0 - sum(fixed.values())
        scale = remaining / max(sum(w[s] for s in free), 1e-12)
        clipped = []
        for s in list(free):
            x = w[s] * scale
            if x < MIN_W:
                fixed[s] = MIN_W
                clipped.append(s)
            elif x > MAX_W:
                fixed[s] = MAX_W
                clipped.append(s)
            else:
                w[s] = x
        if not clipped:
            break
        free -= set(clipped)
    if free:
        left = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = left * w[s] / max(z, 1e-12)
    w.update(fixed)
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _calls
    _calls += 1
    if _calls > 1 and (_calls - 1) % REBALANCE_EVERY != 0:
        return

    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 26:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        if not np.all(np.isfinite(c[-26:])):
            continue
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        data[s] = (c, h, l, r)
    if len(data) < 12:
        return

    clv, reversal, momentum, recent, invvol = {}, {}, {}, {}, {}
    for s, (c, h, l, r) in data.items():
        # One-day close location, with a tiny range floor for stale series.
        clv[s] = float((c[-1] - l[-1]) / max(h[-1] - l[-1], abs(c[-1]) * 1e-8))
        reversal[s] = -float(np.mean(r[-5:]))
        sigma = max(float(np.std(r[-20:])), 0.008)
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / (sigma + 0.01))
        recent[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
        invvol[s] = 1.0 / sigma

    # Leave-one-out peer median lead-lag: peers' recent performance is the signal.
    peer = {}
    for s in data:
        peers = [v for t, v in recent.items() if t != s]
        peer[s] = float(np.median(peers)) if peers else 0.0
    ranks = {k: rank(v) for k, v in (("clv", clv), ("peer", peer),
                                      ("reversal", reversal), ("momentum", momentum))}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranks[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}

    # Bullish/moderate-high risk: full investment, diversified caps, volatility control.
    avg_inv = np.mean(list(invvol.values()))
    raw = {s: max(score[s], 0.02) * (0.85 + 0.15 * invvol.get(s, avg_inv) / max(avg_inv, 1e-12))
           for s in UNIVERSE}
    rebalance_to_weights(bounded_weights(raw))
