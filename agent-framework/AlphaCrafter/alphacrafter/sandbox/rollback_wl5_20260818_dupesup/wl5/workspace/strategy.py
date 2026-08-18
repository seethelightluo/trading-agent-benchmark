import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
                                    get_index_daily_data, rebalance_to_weights)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Exact current screener ensemble; all directions are +1.
FACTOR_IDS = [
    "miner_2_20280128_contrarian_risk_adjusted_trend_20d",
    "miner_2_downside_efficiency_20d",
    "miner_2_20280215_volneutral_recovery_10d",
    "miner_1_20260716_peer_median_leadlag_5d",
    "miner_3_20280114_risk_momentum_20d",
    "miner_2_20280509_volshock10_reversal_10d",
    "miner_2_20271217_vix_shock_resilience_10d",
    "miner_2_channel_trend_20d",
    "miner_2_20280829_smoothed_volnorm_reversal_10d",
]
FACTOR_WEIGHTS = np.array([.29, .14, .14, .12, .10, .07, .08, .04, .02])
_last_decision = None


def cs_rank(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1) / n if n > 1 else .5
    return out


def make_weights(score):
    # Full investment and modest concentration for a 15-asset cross-section.
    x = np.array([max(float(score.get(s, .5)), 1e-9) for s in UNIVERSE])
    x /= x.sum()
    for _ in range(100):
        y = np.clip(x, .03, .14)
        fixed = (y <= .03000001) | (y >= .13999999)
        if (~fixed).any():
            y[~fixed] = x[~fixed] / x[~fixed].sum() * (1 - y[fixed].sum())
        if np.max(np.abs(y - x)) < 1e-12:
            break
        x = y
    y = np.maximum(y, 0.0)
    y /= y.sum()
    return {s: float(w) for s, w in zip(UNIVERSE, y)}


@register_hook
def cross_asset_strategy():
    global _last_decision
    prices, returns = {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=190)
        if df is None or len(df) < 70:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if np.all(np.isfinite(c[-70:])) and np.all(c[-70:] > 0):
            prices[s] = c
            returns[s] = c[1:] / c[:-1] - 1.0
    if len(prices) < 12:
        return
    dates = [get_stock_daily_data(symbol=s, days=2).sort_values("date").iloc[-1]["date"] for s in prices]
    decision = str(max(dates))
    if _last_decision is not None:
        try:
            if (np.datetime64(decision) - np.datetime64(_last_decision)) / np.timedelta64(1, "D") < 10:
                return
        except Exception:
            return

    r5 = {s: prices[s][-1] / prices[s][-6] - 1 for s in prices}
    r10 = {s: prices[s][-1] / prices[s][-11] - 1 for s in prices}
    r20 = {s: prices[s][-1] / prices[s][-21] - 1 for s in prices}
    vol20 = {s: max(float(np.std(returns[s][-20:])), .005) for s in prices}
    vol10 = {s: max(float(np.std(returns[s][-10:])), .005) for s in prices}
    peer = float(np.median(list(r5.values())))
    raw = [dict() for _ in FACTOR_IDS]
    for s in prices:
        downside = float(np.std(np.minimum(returns[s][-20:], 0.0)))
        raw[0][s] = -r20[s] / vol20[s]
        raw[1][s] = -downside / vol20[s]
        raw[2][s] = (prices[s][-1] / max(np.min(prices[s][-11:]), 1e-12) - 1) / vol10[s]
        raw[3][s] = (r5[s] - peer) / vol10[s]
        raw[4][s] = r20[s] / vol20[s]
        raw[5][s] = -r10[s] / vol10[s]
        raw[6][s] = 1.0 / vol20[s]
        raw[7][s] = r20[s] / vol20[s]
        # Smoothed volatility-normalized reversal: lower noise via two windows.
        raw[8][s] = -r10[s] / max((vol10[s] + vol20[s]) * .5, .005)
    ranks = [cs_rank(v) for v in raw]
    score = {s: sum(FACTOR_WEIGHTS[i] * ranks[i][s] for i in range(len(FACTOR_IDS))) for s in UNIVERSE}

    vix = get_index_daily_data(symbol="VIX", days=25)
    shock = False
    if vix is not None and len(vix) >= 6:
        vc = np.asarray(vix.sort_values("date")["close"], dtype=float)
        shock = bool(np.isfinite(vc[-1]) and vc[-1] / max(vc[-6], 1e-9) - 1 > .08)
    if shock or r20.get("SPX", 0) < 0 or vol20.get("SPX", 0) > .018:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += .07
        for s in ("BTC", "ETH", "WTI"):
            score[s] *= .82

    target = make_weights(score)
    z = np.array([score[s] for s in UNIVERSE])
    z = (z - z.mean()) / max(float(z.std()), 1e-12)
    forecast_returns = {s: float(.006 * z[i]) for i, s in enumerate(UNIVERSE)}
    rebalance_to_weights(target, forecast_returns=forecast_returns,
                         factor_ids=FACTOR_IDS, horizon_days=10)
    _last_decision = decision

assert len(UNIVERSE) == 15 and len(FACTOR_IDS) == 9
assert abs(float(FACTOR_WEIGHTS.sum()) - 1.0) < 1e-9
assert set(UNIVERSE).isdisjoint({"DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"})
