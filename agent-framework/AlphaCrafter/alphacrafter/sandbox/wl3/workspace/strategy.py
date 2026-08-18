import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_2_20280907_breakout_failure_reversal",
    "miner_2_20270603_medium_dispersion_reversal",
    "miner_3_20280601_beta_residual_momentum20",
    "miner_2_20280727_breakout_distance120",
    "miner_2_20341208_smooth_trend_acceleration20_60",
]
FACTOR_WEIGHTS = np.array([0.32, 0.22, 0.18, 0.15, 0.13])
CADENCE = 10
_day = 0
_previous = None


def _rank(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / len(good)
    return out


def _capped_weights(raw, cap=0.20):
    w = np.array([max(float(raw.get(s, 0.01)), 1e-8) for s in UNIVERSE])
    w /= w.sum()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        rest = ~over
        w[rest] += excess * w[rest] / max(float(w[rest].sum()), 1e-12)
    w /= w.sum()
    return dict(zip(UNIVERSE, map(float, w)))


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return

    prices, returns, vols = {}, {}, {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=285)
        if df is None or len(df) < 145:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 125 or np.any(~np.isfinite(close[-125:])) or np.any(close[-125:] <= 0):
            continue
        prices[symbol] = close
        returns[symbol] = np.diff(close) / close[:-1]
        vols[symbol] = max(float(np.std(returns[symbol][-30:])), 0.004)
    if len(prices) < 8:
        return

    n = min(len(x) for x in returns.values())
    matrix = np.array([returns[s][-n:] for s in prices])
    benchmark = matrix.mean(axis=0)
    signals = [dict() for _ in FACTOR_IDS]
    for symbol, close in prices.items():
        r = returns[symbol][-n:]
        v20 = max(float(np.std(r[-20:])), 0.004)
        v60 = max(float(np.std(r[-60:])), 0.004)
        beta = float(np.cov(r[-60:], benchmark[-60:], ddof=1)[0, 1] /
                     max(np.var(benchmark[-60:], ddof=1), 1e-8))
        ret20 = close[-1] / close[-21] - 1.0
        ret60 = close[-1] / close[-61] - 1.0
        ret120 = close[-1] / close[-121] - 1.0
        # Reversal signals: failed/extended breakouts and medium-horizon dispersion.
        failure = -(close[-1] / max(close[-21:-1]) - 1.0) / v20
        dispersion = -(ret60 - float(np.median([prices[x][-1] / prices[x][-61] - 1.0 for x in prices]))) / v60
        residual = np.sum(r[-20:] - beta * benchmark[-20:]) / v20
        breakout = (close[-1] / max(close[-121:-1]) - 1.0) / max(float(np.std(r[-30:])), 0.004)
        acceleration = ret20 / v20 - ret60 / v60
        for i, value in enumerate([failure, dispersion, residual, breakout, acceleration]):
            signals[i][symbol] = float(value)

    ranks = [_rank(x) for x in signals]
    score = {s: sum(float(FACTOR_WEIGHTS[i]) * ranks[i][s] for i in range(5)) for s in UNIVERSE}
    if _previous is not None:
        score = {s: 0.8 * score[s] + 0.2 * _previous[s] for s in UNIVERSE}
    _previous = dict(score)

    breadth = float(np.mean([prices[s][-1] > prices[s][-21] for s in prices]))
    median_vol = float(np.median(list(vols.values())))
    stressed = breadth < 0.40 or median_vol > 0.015
    defensive = {"XAU", "US10Y", "CN10Y"}
    risky = {"BTC", "ETH", "WTI", "COPPER"}
    inv_mean = np.mean([1.0 / vols.get(s, 0.02) for s in UNIVERSE])
    raw = {}
    for s in UNIVERSE:
        vol_adj = np.clip((1.0 / vols.get(s, 0.02)) / inv_mean, 0.70, 1.20)
        raw[s] = max(score[s], 0.03) * (0.85 + 0.15 * vol_adj)
        if stressed:
            raw[s] *= 1.35 if s in defensive else (0.60 if s in risky else 0.90)
    target = _capped_weights(raw)
    # Deterministic, bounded 10-day forecast required by the migration-cost gate.
    forecast = {s: float(np.clip((score[s] - 0.5) * 0.06, -0.03, 0.03)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS)


assert len(UNIVERSE) == 15 and abs(float(FACTOR_WEIGHTS.sum()) - 1.0) < 1e-9
assert not set(UNIVERSE) & {"DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"}
