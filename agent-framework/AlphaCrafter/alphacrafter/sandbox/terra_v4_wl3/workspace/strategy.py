import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = {
    "miner_3_clv_1d": (0.22, "clv"),
    "peer_median_leadlag_5d": (0.24, "peer"),
    "miner_2_risk_adjusted_momentum_20d": (0.20, "mom"),
    "miner_3_trend_consistency_20d_1d": (0.22, "cons"),
    "miner_2_relative_reversal_3d_vol": (0.12, "rev3"),
}
MIN_W, MAX_W = 0.025, 0.15
last_date = None
prior_score = None


def xrank(values):
    good = [(s, float(v)) for s, v in values.items() if np.isfinite(v)]
    out = {s: 0.5 for s in UNIVERSE}
    if not good:
        return out
    vals = np.array([v for _, v in good])
    lo, hi = (np.quantile(vals, [0.05, 0.95]) if len(vals) > 3 else (vals.min(), vals.max()))
    clipped = {s: min(max(v, lo), hi) for s, v in good}
    for i, s in enumerate(sorted(clipped, key=clipped.get)):
        out[s] = (i + 1) / len(clipped)
    return out


def bounded(raw):
    free, result = set(UNIVERSE), {}
    while free:
        remain = 1.0 - sum(result.values())
        denom = sum(max(raw[s], 1e-9) for s in free)
        trial = {s: remain * max(raw[s], 1e-9) / denom for s in free}
        low = [s for s in free if trial[s] < MIN_W]
        high = [s for s in free if trial[s] > MAX_W]
        if not low and not high:
            result.update(trial)
            break
        chosen, fixed = (low, MIN_W) if low else (high, MAX_W)
        for s in chosen:
            result[s] = fixed
            free.remove(s)
    z = sum(result.values())
    return {s: result.get(s, 0.0) / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_date, prior_score
    data = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=110)
        if df is None or len(df) < 35:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c, o, h, low = [np.asarray(df[k], float) for k in ("close", "open", "high", "low")]
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        data[symbol] = (c, o, h, low, r, str(df.iloc[-1]["date"]))
    if len(data) < 12:
        return
    decision_date = max(v[-1] for v in data.values())
    if last_date is not None:
        try:
            if (np.datetime64(decision_date) - np.datetime64(last_date)) / np.timedelta64(1, "D") < 13:
                return
        except Exception:
            return

    values = {key: {} for _, key in FACTORS.values()}
    invvol = {}
    for s, (c, o, h, low, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), 0.008)
        invvol[s] = 1.0 / vol
        values["clv"][s] = (2*c[-1] - h[-1] - low[-1]) / max(h[-1] - low[-1], 1e-12)
        values["peer"][s] = c[-1] / max(c[-6], 1e-12) - 1.0
        values["mom"][s] = (c[-1] / max(c[-21], 1e-12) - 1.0) / vol
        rr = r[-20:]
        values["cons"][s] = np.sign(np.sum(rr)) * np.mean(np.sign(rr)) * abs(c[-1] / max(c[-21], 1e-12) - 1.0) / vol
        values["rev3"][s] = -(c[-1] / max(c[-4], 1e-12) - 1.0) / vol

    med = float(np.median(list(values["peer"].values())))
    values["peer"] = {s: v - med for s, v in values["peer"].items()}
    ranks = {key: xrank(v) for key, v in values.items()}
    score = {s: sum(w * ranks[key].get(s, 0.5) for w, key in FACTORS.values()) for s in UNIVERSE}
    breadth = float(np.mean([v[0][-1] > v[0][-21] for v in data.values()]))
    market_vol = float(np.mean([np.std(v[4][-20:]) for v in data.values()]))
    if breadth < 0.55 or market_vol > 0.025:
        # High-risk bear/sideways regime: use tradable defensive assets, never cash.
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += 0.25
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= 0.10
    if prior_score is not None:
        score = {s: 0.75 * prior_score.get(s, score[s]) + 0.25 * score[s] for s in UNIVERSE}
    prior_score = dict(score)
    mean_iv = max(float(np.mean(list(invvol.values()))), 1e-12)
    raw = {s: max(score[s], 0.05) * (0.65 + 0.35 * invvol.get(s, mean_iv) / mean_iv) for s in UNIVERSE}
    weights = bounded(raw)
    mu, sd = float(np.mean(list(score.values()))), max(float(np.std(list(score.values()))), 1e-12)
    forecast = {s: float(0.01 * (score[s] - mu) / sd) for s in UNIVERSE}
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=list(FACTORS), horizon_days=10)
    last_date = decision_date

if __name__ == "__main__":
    pass
