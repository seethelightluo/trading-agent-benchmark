import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble (8 active factors, all positive direction).
FACTORS = {"mom20": .07, "mom30": .07, "resid30": .06, "downmom": .24,
           "eff": .22, "persist": .08, "lead": .17, "rev": .09}
MIN_W, MAX_W = .025, .16
_day = 0


def rank_cs(vals):
    out = {s: .5 for s in ASSETS}
    valid = [(s, float(v)) for s, v in vals.items() if np.isfinite(v)]
    valid.sort(key=lambda z: z[1])
    n = len(valid)
    if n > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1.) / n
    return out


def bounded_weights(scores):
    # Positive score transform keeps all 15 assets invested; caps limit concentration.
    raw = {s: max(.05, float(scores.get(s, .5))) for s in ASSETS}
    base = 1. - MIN_W * len(ASSETS)
    w = {s: MIN_W + base * raw[s] / sum(raw.values()) for s in ASSETS}
    for _ in range(20):
        over = sum(max(0., w[s] - MAX_W) for s in ASSETS)
        if over < 1e-10:
            break
        capped = {s for s in ASSETS if w[s] >= MAX_W - 1e-10}
        for s in capped:
            w[s] = MAX_W
        free = [s for s in ASSETS if s not in capped]
        if not free:
            break
        den = sum(raw[s] for s in free)
        for s in free:
            w[s] += over * raw[s] / den
    z = sum(w.values())
    return {s: max(0., w[s] / z) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % 10 != 0:
        return

    data = {}
    for symbol in ASSETS:
        df = get_stock_daily_data(symbol=symbol, days=180)
        if df is None or len(df) < 65:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(close) < 65 or not np.all(np.isfinite(close[-65:])) or close[-1] <= 0:
            continue
        ret = close[1:] / np.maximum(close[:-1], 1e-12) - 1.
        vol = max(float(np.std(ret[-20:])), .008)
        down = max(float(np.std(np.minimum(ret[-20:], 0.))), .004)
        t20 = close[-1] / close[-21] - 1.
        t30 = close[-1] / close[-31] - 1.
        data[symbol] = {
            "mom20": t20 / (vol + .01) * (.5 + np.mean(ret[-20:] > 0)),
            "mom30": t30 / (vol + .01) * (.5 + np.mean(ret[-30:] > 0)),
            "downmom": t20 / (down + .01),
            "eff": t20 / (down + .015),
            "persist": t30 / (vol + .01) * np.mean(ret[-30:] > 0),
            "lead": close[-1] / close[-6] - 1.,
            "rev": -float(np.mean(ret[-5:])),
            "trend": t30, "vol": vol
        }

    # Missing observations receive neutral ranks but the target remains complete.
    if len(data) < 10:
        return
    market = float(np.median([x["trend"] for x in data.values()]))
    for x in data.values():
        x["resid30"] = x["trend"] - market
    ranks = {f: rank_cs({s: x[f] for s, x in data.items()}) for f in FACTORS}
    score = {s: sum(FACTORS[f] * ranks[f][s] for f in FACTORS) for s in ASSETS}

    # Bear/high-risk regime: full investment is retained, with defensive tradable tilts.
    eq_names = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "NDX"]
    eq = [data[s] for s in eq_names if s in data]
    breadth = sum(x["trend"] > 0 for x in eq) / max(1, len(eq))
    spx = data.get("SPX", {})
    median_vol = float(np.median([x["vol"] for x in data.values()]))
    bear = len(eq) >= 3 and (breadth <= .50 or (spx.get("trend", 0.) < 0 and spx.get("lead", 0.) < 0))
    high_risk = median_vol > .018 or breadth < .67
    if bear or high_risk:
        boost = .16 if bear else .10
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += boost
        for s in ("BTC", "ETH", "WTI"):
            score[s] = max(.05, score[s] - (.08 if bear else .04))

    # Mild inverse-volatility overlay controls concentration without changing factor ranks.
    for s in ASSETS:
        if s in data:
            score[s] *= .985 + .015 * median_vol / max(data[s]["vol"], .004)
    target = bounded_weights(score)
    if abs(sum(target.values()) - 1.) < 1e-8:
        rebalance_to_weights(target)


def strategy():
    return cross_asset_strategy()
