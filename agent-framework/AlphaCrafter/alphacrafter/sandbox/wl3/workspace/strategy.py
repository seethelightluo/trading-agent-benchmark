import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
CADENCE = 10
MIN_W, MAX_W = 0.04, 0.18
# Screener ensemble: reversal-led, volatility managed, with only a small momentum sleeve.
FACTORS = ("cluster", "volstate", "failure", "consistency", "lowvol", "stress", "residual_mom")
FACTOR_W = (0.24, 0.20, 0.16, 0.15, 0.12, 0.08, 0.05)
_day = 0
_previous = None

def rank(vals):
    good = sorted((s, float(v)) for s, v in vals.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(n, 1)
    return out

def bounded(raw):
    # Iterative water-filling enforces complete, long-only, full-investment weights.
    w = {s: max(float(raw.get(s, 1.0)), 1e-9) for s in UNIVERSE}
    fixed = {}
    for _ in range(40):
        free = [s for s in UNIVERSE if s not in fixed]
        remain = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = remain * w[s] / max(z, 1e-12)
        hit = [s for s in free if w[s] < MIN_W or w[s] > MAX_W]
        if not hit:
            break
        for s in hit:
            fixed[s] = MIN_W if w[s] < MIN_W else MAX_W
            w[s] = fixed[s]
    z = sum(w.values())
    return {s: w[s] / max(z, 1e-12) for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=220)
        if df is None or len(df) < 100:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(c) < 80 or np.any(~np.isfinite(c)) or np.any(c <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), 0.006)
        v60 = max(float(np.std(r[-60:])), 0.006)
        r3 = np.prod(1 + r[-3:]) - 1
        r5 = np.prod(1 + r[-5:]) - 1
        r20 = np.prod(1 + r[-20:]) - 1
        r60 = np.prod(1 + r[-60:]) - 1
        # Price-only proxies preserve factor direction while avoiding look-ahead.
        rev = np.clip(-r20 / (v60 * np.sqrt(20)), -3, 3)
        data[s] = {
            "cluster": rev,
            "volstate": np.clip(-r3 / v20, -3, 3),
            "failure": np.clip(-r5 / v20, -3, 3),
            "consistency": np.clip((np.mean(r[-60:] > 0) - .5) / v60, -3, 3),
            "lowvol": np.clip(-r5 / v20, -3, 3) / (1 + 12 * v20),
            "stress": np.clip(-r5 / v20, -3, 3) / (1 + 8 * v20),
            "residual_mom": np.clip(r20 / (v60 * np.sqrt(20)), -3, 3),
            "vol": v20,
            "r": r,
        }
    if len(data) < 10:
        return
    rr = {f: rank({s: data[s][f] for s in data}) for f in FACTORS}
    score = {s: sum(FACTOR_W[i] * rr[FACTORS[i]][s] for i in range(len(FACTORS))) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    vols = [x["vol"] for x in data.values()]
    market = data.get("000300.SH", data.get("SPX"))
    market30 = np.prod(1 + market["r"][-30:]) - 1 if market is not None else 0.0
    breadth = np.mean([data[s]["consistency"] > 0 for s in data])
    stressed = float(np.median(vols)) > .015 or market30 < -.06 or breadth < .40
    mean_inv = np.mean([1 / x["vol"] for x in data.values()])
    raw = {}
    for s in UNIVERSE:
        x = data.get(s, {"vol": np.median(vols)})
        damp = np.clip((1 / max(x["vol"], .006)) / mean_inv, .78, 1.10)
        raw[s] = max(score[s], .05) * (.88 + .12 * damp)
        if stressed:
            raw[s] *= 1.70 if s in DEFENSIVE else (.48 if s in RISKY else .86)
        elif s in RISKY:
            raw[s] *= .90
    target = bounded(raw)
    if set(target) == set(UNIVERSE) and abs(sum(target.values()) - 1) < 1e-8:
        rebalance_to_weights(target)
