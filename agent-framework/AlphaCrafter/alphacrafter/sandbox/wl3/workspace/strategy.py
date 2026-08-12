import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
FACTOR_IDS = [
    "miner_2_20320624_relative_momentum20",
    "miner_3_20280601_beta_residual_momentum20",
    "miner_2_20280615_volmanaged_consistency30",
    "miner_2_20280629_breakout_distance60",
    "miner_2_20280907_breakout_failure_reversal",
    "miner_3_20270211_volstate_reversal_3d",
    "miner_2_20270520_dispersion_conditioned_reversal",
]
FACTOR_WEIGHTS = [0.32, 0.14, 0.13, 0.11, 0.10, 0.10, 0.10]
FACTOR_DIRECTIONS = [1, 1, 1, 1, 1, 1, 1]
CADENCE = 10
_day = 0
_previous_score = None


def rank(values):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / len(good)
    return out


def capped_weights(raw):
    w = np.array([max(float(raw.get(s, 0.0)), 0.02) for s in UNIVERSE], dtype=float)
    w /= w.sum()
    for _ in range(20):
        over = w > 0.25
        if not np.any(over):
            break
        excess = float((w[over] - 0.25).sum())
        w[over] = 0.25
        under = ~over
        w[under] += excess * w[under] / max(float(w[under].sum()), 1e-12)
    w /= w.sum()
    return dict(zip(UNIVERSE, (float(x) for x in w)))


@register_hook
def cross_asset_strategy():
    global _day, _previous_score
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE:
        return

    features = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=320)
        if df is None or len(df) < 140:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 130 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        ret = close[1:] / close[:-1] - 1.0
        v10 = max(float(np.std(ret[-10:])), 0.006)
        v20 = max(float(np.std(ret[-20:])), 0.006)
        v30 = max(float(np.std(ret[-30:])), 0.006)
        v60 = max(float(np.std(ret[-60:])), 0.006)
        m3 = float(np.prod(1 + ret[-3:]) - 1)
        m5 = float(np.prod(1 + ret[-5:]) - 1)
        m20 = float(np.prod(1 + ret[-20:]) - 1)
        m30 = float(np.prod(1 + ret[-30:]) - 1)
        h20 = max(float(np.max(close[-20:])), 1e-12)
        h60 = max(float(np.max(close[-60:])), 1e-12)
        features[symbol] = {"ret": ret, "v": v20, "relative": m20,
            "rev3": -m3 / v10, "failrev": -((close[-1] / h20) - 1) / v20,
            "consistency": m30 / v30 * np.mean(ret[-30:] > 0),
            "break60": close[-1] / h60, "disprev": -m5 / v60}
    if len(features) < 10:
        return

    benchmark = features.get("000300.SH", features.get("SPX"))["ret"]
    bench_m20 = float(np.prod(1 + benchmark[-20:]) - 1)
    for x in features.values():
        n = min(60, len(x["ret"]), len(benchmark))
        a, b = x["ret"][-n:], benchmark[-n:]
        beta = float(np.cov(a, b, ddof=0)[0, 1]) / max(float(np.var(b)), 1e-8)
        x["residual"] = float(np.sum(a[-20:] - beta * b[-20:])) / max(x["v"] * 4, 0.01)

    raw = {
        "relative": {s: x["relative"] - bench_m20 for s, x in features.items()},
        "residual": {s: x["residual"] for s, x in features.items()},
        "consistency": {s: x["consistency"] for s, x in features.items()},
        "break60": {s: x["break60"] for s, x in features.items()},
        "failrev": {s: x["failrev"] for s, x in features.items()},
        "rev3": {s: x["rev3"] for s, x in features.items()},
        "disprev": {s: x["disprev"] for s, x in features.items()},
    }
    ranks = {k: rank(v) for k, v in raw.items()}
    score = {s: sum(w * ranks[k].get(s, 0.5) for w, k in zip(FACTOR_WEIGHTS, ranks)) for s in UNIVERSE}
    if _previous_score is not None:
        score = {s: 0.8 * score[s] + 0.2 * _previous_score[s] for s in UNIVERSE}
    _previous_score = dict(score)

    stressed = (float(np.median([x["v"] for x in features.values()])) > 0.015 or
                np.mean([x["relative"] > 0 for x in features.values()]) < 0.40)
    mean_invvol = float(np.mean([1.0 / x["v"] for x in features.values()]))
    raw_weight = {}
    for s in UNIVERSE:
        x = features.get(s)
        invvol = 1.0 if x is None else np.clip((1.0 / x["v"]) / mean_invvol, 0.75, 1.15)
        tilt = (1.45 if s in DEFENSIVE else 0.55 if s in RISKY else 0.85) if stressed else (1.15 if s in DEFENSIVE else 0.85 if s in RISKY else 1.0)
        raw_weight[s] = max(score[s], 0.04) * (0.85 + 0.15 * invvol) * tilt
    target = capped_weights(raw_weight)
    forecast = {s: float(np.clip((score[s] - 0.5) * 0.06, -0.03, 0.03)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS)


assert len(UNIVERSE) == 15 and len(FACTOR_IDS) == 7
assert FACTOR_IDS == ["miner_2_20320624_relative_momentum20", "miner_3_20280601_beta_residual_momentum20", "miner_2_20280615_volmanaged_consistency30", "miner_2_20280629_breakout_distance60", "miner_2_20280907_breakout_failure_reversal", "miner_3_20270211_volstate_reversal_3d", "miner_2_20270520_dispersion_conditioned_reversal"]
assert abs(sum(FACTOR_WEIGHTS) - 1.0) < 1e-9
assert all(x == 1 for x in FACTOR_DIRECTIONS)
assert len(FACTOR_IDS) <= 10
מס = None
if מס is not None:
    pass
