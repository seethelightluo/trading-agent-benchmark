import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
# Six active factors supplied by the screener; all directions are positive.
FACTORS = ("consistency", "failure", "residual", "relative", "volstate", "lowvol")
FACTOR_W = {"consistency": .24, "failure": .20, "residual": .18,
            "relative": .14, "volstate": .14, "lowvol": .10}
CADENCE = 10
MIN_W, MAX_W = .04, .18
_day = 0
_previous = None

def rank_cross_section(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(n, 1)
    return out

def project_weights(raw):
    # Clamp and renormalize iteratively; preserves a complete fractional target.
    w = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    fixed = {}
    for _ in range(80):
        free = [s for s in UNIVERSE if s not in fixed]
        remaining = 1.0 - sum(fixed.values())
        total = sum(w[s] for s in free)
        for s in free:
            w[s] = remaining * w[s] / max(total, 1e-12)
        hit = [s for s in free if w[s] < MIN_W or w[s] > MAX_W]
        if not hit:
            break
        for s in hit:
            fixed[s] = MIN_W if w[s] < MIN_W else MAX_W
            w[s] = fixed[s]
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return

    data = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=220)
        if df is None or len(df) < 100:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 80 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        ret = close[1:] / close[:-1] - 1.0
        v20 = max(float(np.std(ret[-20:])), .006)
        v60 = max(float(np.std(ret[-60:])), .006)
        data[symbol] = {"ret": ret, "v20": v20, "v60": v60,
            "consistency": (np.mean(ret[-60:] > 0) - .5) / v60,
            "failure": -(np.prod(1 + ret[-5:]) - 1) / v20,
            "relative": (np.prod(1 + ret[-20:]) - 1) / v60,
            "volstate": -(np.prod(1 + ret[-3:]) - 1) / v20,
            "lowvol": -v20}

    if len(data) < 10:
        return
    market = data.get("000300.SH", data.get("SPX"))
    if market is not None:
        mr = market["ret"][-20:]
        vm = max(float(np.var(mr)), 1e-8)
        for symbol, x in data.items():
            ar = x["ret"][-20:]
            beta = float(np.cov(ar, mr, ddof=0)[0, 1] / vm)
            residual = ar - beta * mr
            x["residual"] = float(np.mean(residual)) / x["v60"]
    ranks = {f: rank_cross_section({s: x[f] for s, x in data.items()}) for f in FACTORS}
    score = {s: sum(FACTOR_W[f] * ranks[f][s] for f in FACTORS) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()

    vols = [x["v20"] for x in data.values()]
    median_vol = float(np.median(vols))
    breadth = float(np.mean([x["consistency"] > 0 for x in data.values()]))
    market30 = np.prod(1 + market["ret"][-30:]) - 1 if market is not None else 0.0
    stressed = median_vol > .015 or market30 < -.06 or breadth < .40
    inv_mean = float(np.mean([1.0 / v for v in vols]))
    raw = {}
    for symbol in UNIVERSE:
        x = data.get(symbol)
        vol = x["v20"] if x is not None else median_vol
        damp = np.clip((1.0 / max(vol, .006)) / inv_mean, .78, 1.10)
        raw[symbol] = max(score[symbol], .03) * (.88 + .12 * damp)
        if stressed:
            raw[symbol] *= 1.70 if symbol in DEFENSIVE else (.48 if symbol in RISKY else .86)
        elif symbol in RISKY:
            raw[symbol] *= .90

    target = project_weights(raw)
    if set(target) == set(UNIVERSE) and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)
