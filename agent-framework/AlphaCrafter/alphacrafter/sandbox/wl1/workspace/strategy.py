import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20260730_vix_state_reversal_5d",
    "miner_3_20261105_downside_risk_adjusted_momentum",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_3_20260730_vix_conditioned_clv_1d",
    "miner_3_20270128_downside_asymmetry_30d",
    "miner_2_20261203_drawdown_recovery_20d",
]
FACTOR_WEIGHTS = np.array([.193, .177, .156, .155, .133, .096, .090])
_gate = 0


def cs_rank(values, symbols):
    valid = sorted((float(values[s]), s) for s in symbols if np.isfinite(values.get(s, np.nan)))
    out = {s: .5 for s in symbols}
    n = len(valid)
    for i, (_, s) in enumerate(valid):
        out[s] = (i + 1.0) / n if n else .5
    return out


def capped_weights(score, vol, defensive):
    # In stress, keep every asset represented but put the incremental risk budget
    # in gold and the two yield benchmarks; crypto and oil are explicitly capped.
    raw = {}
    for s in UNIVERSE:
        tilt = 1.0
        if defensive and s in ("XAU", "US10Y", "CN10Y"):
            tilt = 1.45
        if defensive and s in ("BTC", "ETH", "WTI"):
            tilt = .55
        raw[s] = max(.001, (.25 + score.get(s, .5)) * tilt / max(vol.get(s, .02), .009))
    total = sum(raw.values())
    w = {s: raw[s] / total for s in UNIVERSE}
    cap = .12 if defensive else .14
    # Iterative cap-and-redistribute preserves nonnegative, fully invested targets.
    for _ in range(50):
        over = sum(max(0.0, w[s] - cap) for s in UNIVERSE)
        if over < 1e-10:
            break
        fixed = {s for s in UNIVERSE if w[s] >= cap - 1e-10}
        for s in fixed:
            w[s] = cap
        free = [s for s in UNIVERSE if s not in fixed]
        base = sum(w[s] for s in free)
        if not free or base <= 0:
            break
        for s in free:
            w[s] += over * w[s] / base
    z = sum(w.values())
    return {s: max(0.0, w[s] / z) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return

    prices, returns, vols = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 80:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            prices[s] = c
            returns[s] = c[1:] / c[:-1] - 1.0
            vols[s] = max(float(np.std(returns[s][-30:])), .008)
    syms = [s for s in UNIVERSE if s in prices]
    if len(syms) < 12:
        return

    r5 = {s: prices[s][-1] / prices[s][-6] - 1.0 for s in syms}
    r20 = {s: prices[s][-1] / prices[s][-21] - 1.0 for s in syms}
    r60 = {s: prices[s][-1] / prices[s][-61] - 1.0 for s in syms}
    breadth = float(np.mean([r20[s] > 0 for s in syms]))

    vix_level, vix_jump = 0.0, 0.0
    vix = get_index_daily_data(symbol="VIX", days=35)
    if vix is not None and len(vix) >= 22:
        vc = np.asarray(vix.sort_values("date")["close"], dtype=float)
        vix_level = vc[-1] / max(np.median(vc[-21:]), 1e-9) - 1.0
        vix_jump = vc[-1] / max(vc[-6], 1e-9) - 1.0
    defensive = (r20.get("SPX", 0.0) < 0 and r5.get("SPX", 0.0) < 0) or breadth < .45 or np.mean(list(vols.values())) > .022

    factors = [dict() for _ in FACTOR_WEIGHTS]
    for s in syms:
        d, v = returns[s], vols[s]
        downside = max(float(np.std(np.minimum(d[-40:], 0.0))), .006)
        peer = np.mean([returns[x][-20:] for x in syms if x != s], axis=0)
        beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        residual = r20[s] - beta * np.median([r20[x] for x in syms if x != s])
        # The seven live factors, each ranked before combination, avoid scale dominance.
        factors[0][s] = -r5[s] * (1.0 + max(vix_level, 0.0)) / v
        factors[1][s] = r20[s] / downside
        factors[2][s] = residual / v
        factors[3][s] = residual / max(v, .009)
        clv = (prices[s][-1] - np.min(prices[s][-20:])) / max(np.ptp(prices[s][-20:]), 1e-9) - .5
        factors[4][s] = clv * (1.0 + max(vix_jump, 0.0)) / v
        factors[5][s] = (np.mean(np.maximum(d[-30:], 0.0)) - np.mean(np.minimum(d[-30:], 0.0))) / downside
        factors[6][s] = (r20[s] - r60[s]) / v

    score = {s: sum(w * cs_rank(f, syms)[s] for w, f in zip(FACTOR_WEIGHTS, factors)) for s in syms}
    target = capped_weights(score, vols, defensive)
    forecast = {s: max(.0001, .04 * (score.get(s, .5) - .5)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _gate = 9
