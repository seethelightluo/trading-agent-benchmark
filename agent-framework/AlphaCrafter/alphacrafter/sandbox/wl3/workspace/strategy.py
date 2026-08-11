import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CADENCE = 10
MIN_W, MAX_W = 0.04, 0.20
# Current Screener ensemble, directions all positive.
FACTORS = ("failure", "residual", "consistency", "short_breakout", "breakout120", "downside")
FACTOR_W = (0.26, 0.22, 0.20, 0.14, 0.10, 0.08)
_day = 0
_previous = None


def cs_rank(values):
    valid = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(valid):
        out[s] = (i + 1.0) / max(1, len(valid))
    return out


def bounded(raw):
    w = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    fixed = set()
    for _ in range(50):
        free = [s for s in UNIVERSE if s not in fixed]
        rem = 1.0 - sum(w[s] for s in fixed)
        den = sum(w[s] for s in free)
        for s in free:
            w[s] = rem * w[s] / max(den, 1e-12)
        hit = [s for s in free if w[s] < MIN_W or w[s] > MAX_W]
        if not hit:
            break
        for s in hit:
            w[s] = MIN_W if w[s] < MIN_W else MAX_W
        fixed.update(hit)
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return

    feats, returns = {}, {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=195)
        if df is None or len(df) < 130:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)
        # Explicit one-session lag: the final visible bar is not used as a
        # fresh signal observation. This also avoids accidental look-ahead.
        close = close[:-1]
        if len(close) < 125 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        vol = max(float(np.std(r[-20:])), 0.006)
        r5 = float(np.prod(1 + r[-5:]) - 1)
        r20 = float(np.prod(1 + r[-20:]) - 1)
        r30 = float(np.prod(1 + r[-30:]) - 1)
        h20 = max(float(np.max(close[-21:])), 1e-12)
        h120 = max(float(np.max(close[-121:])), 1e-12)
        downvol = max(float(np.sqrt(np.mean(np.minimum(r[-20:], 0) ** 2))), 0.006)
        feats[symbol] = {
            "vol": vol,
            "failure": (-r5) * max(0.0, close[-1] / h20 - 0.985) / vol,
            "short_breakout": (close[-1] / h20 - 1.0) / max(vol * np.sqrt(20), .01),
            "breakout120": (close[-1] / h120 - 1.0) / max(vol * np.sqrt(120), .01),
            "downside": r20 / downvol,
            "consistency": r30 / max(vol * np.sqrt(30), .01) * (.5 + np.mean(r[-30:] > 0)),
            "residual": r20 / max(vol * np.sqrt(20), .01),
        }
        returns[symbol] = r
    if len(feats) < 10:
        return

    market = returns.get("SPX")
    if market is not None:
        for symbol, r in returns.items():
            n = min(60, len(r), len(market))
            if n >= 30:
                a, b = r[-n:], market[-n:]
                beta = np.cov(a, b, ddof=1)[0, 1] / max(np.var(b, ddof=1), 1e-8)
                mkt20 = np.prod(1 + b[-20:]) - 1
                feats[symbol]["residual"] -= beta * mkt20 / max(np.std(b[-20:]) * np.sqrt(20), .01)

    ranks = [cs_rank({s: feats[s][f] for s in feats}) for f in FACTORS]
    score = {s: sum(FACTOR_W[i] * ranks[i][s] for i in range(6)) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .75 * score[s] + .25 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()

    median_vol = float(np.median([x["vol"] for x in feats.values()]))
    breadth = float(np.mean([x["breakout120"] > 0 for x in feats.values()]))
    stressed = median_vol > .018 or breadth < .45
    mean_invvol = np.mean([1.0 / x["vol"] for x in feats.values()])
    raw = {}
    for symbol in UNIVERSE:
        x = feats.get(symbol, {"vol": median_vol, "breakout120": 0.0})
        damp = np.clip((1.0 / x["vol"]) / max(mean_invvol, 1e-12), .65, 1.25)
        raw[symbol] = max(score[symbol], .05) * (.82 + .18 * damp)
        if stressed:
            raw[symbol] *= 5.0 if symbol in DEFENSIVE else (.55 if x["breakout120"] < -.10 else .78)
    target = bounded(raw)
    if all(np.isfinite(target[s]) for s in UNIVERSE) and abs(sum(target.values()) - 1.0) < 1e-8:
        rebalance_to_weights(target)
