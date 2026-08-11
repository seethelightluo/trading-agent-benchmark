import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current screener ensemble (six active factors; all directions +1).
FACTORS = {"cons30": 0.30, "cons20": 0.26, "persist30": 0.12,
           "downmom": 0.12, "eff": 0.11, "resid": 0.09}
MIN_W, MAX_W = 0.03, 0.15
POSITION_SCALING = 0.85
REBALANCE_DAYS = 10
_day = 0


def rank_cs(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in ASSETS}
    if not good:
        return out
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / len(good)
    return out


def make_weights(score, invvol):
    # Rank tilt is shrunk toward equal weight; inverse volatility only
    # moderates risk, rather than changing the factor ordering aggressively.
    equal = 1.0 / len(ASSETS)
    raw = {}
    for s in ASSETS:
        tilt = max(float(score.get(s, 0.5)), 0.05)
        risk = np.clip(float(invvol.get(s, 1.0)), 0.65, 1.35)
        raw[s] = max(0.001, equal + POSITION_SCALING * (tilt - 0.5) / len(ASSETS)) * risk
    w = {s: MIN_W + (1.0 - MIN_W * len(ASSETS)) * raw[s] / sum(raw.values()) for s in ASSETS}
    # Cap and redistribute excess, preserving the full-investment invariant.
    for _ in range(30):
        over = [s for s in ASSETS if w[s] > MAX_W + 1e-12]
        if not over:
            break
        excess = sum(w[s] - MAX_W for s in over)
        for s in over:
            w[s] = MAX_W
        free = [s for s in ASSETS if s not in over]
        den = sum(raw[s] for s in free)
        if not free or den <= 0:
            break
        for s in free:
            w[s] += excess * raw[s] / den
    total = sum(w.values())
    return {s: max(0.0, w[s] / total) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % REBALANCE_DAYS != 0:
        return

    features = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=190)
        if df is None or len(df) < 65:
            continue
        df = df.sort_values("date")
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        if np.any(~np.isfinite(c[-65:])) or np.any(c[-65:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        vol20 = max(float(np.std(r[-20:])), 0.006)
        downside = max(float(np.std(np.minimum(r[-20:], 0.0))), 0.003)
        t20, t30 = c[-1] / c[-21] - 1.0, c[-1] / c[-31] - 1.0
        p20, p30 = np.mean(r[-20:] > 0), np.mean(r[-30:] > 0)
        bar_range = np.maximum(h[-30:] - l[-30:], c[-30:] * .001) / c[-30:]
        rr = max(float(np.mean(bar_range)), .002)
        features[s] = {
            "cons20": t20 / (vol20 + .01) * (.5 + p20),
            "cons30": t30 / (vol20 + .01) * (.5 + p30),
            "downmom": t20 / (downside + .01),
            "eff": t20 / (downside + .015),
            "persist30": t30 / (rr + .01) * (.5 + p30),
            "resid": t30, "t30": t30, "vol": vol20,
        }
    if len(features) < 10:
        return

    # Cross-sectional residual momentum and lagged completed-day observations.
    median_t30 = float(np.median([x["t30"] for x in features.values()]))
    for x in features.values():
        x["resid"] -= median_t30
    ranks = {f: rank_cs({s: x[f] for s, x in features.items()}) for f in FACTORS}
    score = {s: sum(FACTORS[f] * ranks[f][s] for f in FACTORS) for s in ASSETS}

    # Sideways-to-bullish, medium-risk posture: retain momentum breadth, but
    # use defensive tradable assets when breadth/volatility contradicts it.
    equities = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"]
    breadth = np.mean([features[s]["t30"] > 0 for s in equities if s in features])
    medvol = float(np.median([x["vol"] for x in features.values()]))
    if breadth < .50 or medvol > .018:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += .05
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= .02

    median_vol = float(np.median([x["vol"] for x in features.values()]))
    invvol = {s: np.clip(median_vol / max(x["vol"], .006), .65, 1.35)
              for s, x in features.items()}
    rebalance_to_weights(make_weights(score, invvol))


def strategy():
    return cross_asset_strategy()
