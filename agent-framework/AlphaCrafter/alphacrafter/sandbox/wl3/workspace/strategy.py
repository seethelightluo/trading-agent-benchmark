import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
CADENCE = 10
MIN_W, MAX_W = 0.04, 0.18
# Current six-factor screener ensemble, mapped to lagged price/volatility proxies.
FACTORS = ("cluster_reversal", "volstate", "failure", "lowvol_reversal", "consistency", "dispersion")
FACTOR_W = (0.22, 0.20, 0.18, 0.16, 0.14, 0.10)
_day = 0
_previous = None


def ranks(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(valid):
        out[s] = (i + 1.0) / max(1, len(valid))
    return out


def bounded_weights(raw):
    w = {s: max(float(raw.get(s, 0.05)), 1e-12) for s in UNIVERSE}
    fixed = {}
    for _ in range(60):
        free = [s for s in UNIVERSE if s not in fixed]
        left = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = left * w[s] / max(z, 1e-12)
        hits = [s for s in free if w[s] < MIN_W or w[s] > MAX_W]
        if not hits:
            break
        for s in hits:
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

    feat, returns = {}, {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=210)
        if df is None or len(df) < 140:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 125 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), 0.006)
        v60 = max(float(np.std(r[-60:])), 0.006)
        r3 = np.prod(1 + r[-3:]) - 1
        r5 = np.prod(1 + r[-5:]) - 1
        r20 = np.prod(1 + r[-20:]) - 1
        # Reversal signals are deliberately smoothed and volatility scaled.
        feat[symbol] = {
            "cluster_reversal": np.clip(-r20 / (v60 * np.sqrt(20)), -3, 3),
            "volstate": np.clip(-r3 / v20, -3, 3),
            "failure": np.clip(-r5 / v20, -3, 3),
            "lowvol_reversal": np.clip(-r5 / v20, -3, 3) / (1.0 + 12.0 * v20),
            "consistency": np.clip(np.mean(r[-60:] > 0) * (.5 + np.mean(r[-60:])) / v60, -3, 3),
            "dispersion": np.clip(-r20 / (v60 * np.sqrt(20)), -3, 3),
            "vol": v20,
        }
        returns[symbol] = r
    if len(feat) < 10:
        return

    ranked = {f: ranks({s: feat[s][f] for s in feat}) for f in FACTORS}
    score = {s: sum(FACTOR_W[i] * ranked[FACTORS[i]][s] for i in range(len(FACTORS))) for s in UNIVERSE}
    if _previous is not None:
        score = {s: 0.70 * score[s] + 0.30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()

    median_vol = float(np.median([x["vol"] for x in feat.values()]))
    market = returns.get("000300.SH", returns.get("SPX"))
    market30 = np.prod(1 + market[-30:]) - 1 if market is not None and len(market) >= 30 else 0.0
    breadth = float(np.mean([feat[s]["consistency"] > 0 for s in feat]))
    stressed = median_vol > 0.015 or market30 < -0.06 or breadth < 0.40
    invvol_mean = float(np.mean([1.0 / x["vol"] for x in feat.values()]))
    raw = {}
    for symbol in UNIVERSE:
        x = feat.get(symbol, {"vol": median_vol})
        damp = np.clip((1.0 / max(x["vol"], .006)) / max(invvol_mean, 1e-12), .78, 1.10)
        raw[symbol] = max(score[symbol], .05) * (.88 + .12 * damp)
        if stressed:
            raw[symbol] *= 1.75 if symbol in DEFENSIVE else (.50 if symbol in RISKY else .86)
        elif symbol in RISKY:
            raw[symbol] *= .90
    target = bounded_weights(raw)
    if len(target) == 15 and abs(sum(target.values()) - 1.0) < 1e-8 and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)
