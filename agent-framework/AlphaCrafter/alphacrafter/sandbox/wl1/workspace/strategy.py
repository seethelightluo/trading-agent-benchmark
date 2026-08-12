import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    get_account_dict, rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
    "miner_1_20320624_path_stability_lead_10d",
    "breadth_vol_quality_40d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "miner_2_20310626_trend_acceleration_quality",
    "miner_3_20310724_relative_momentum_acceleration_20d",
    "macro_stress_resilience_20d",
    "miner_3_20310904_recovery_pullback_20d",
    "miner_1_20311211_exhaustion_contrarian_20d",
]
WEIGHTS = np.array([.15, .12, .14, .11, .09, .09, .05, .10, .08, .07])
WAIT = 0


def rank_cs(x):
    ordered = sorted(x, key=lambda s: (x[s], s))
    n = float(len(ordered))
    return {s: (i + 1.0) / n for i, s in enumerate(ordered)}


def capped(raw, cap=.15):
    out = {s: 0.0 for s in UNIVERSE}
    active = set(UNIVERSE)
    left = 1.0
    while active:
        denom = sum(max(raw[s], 1e-8) for s in active)
        hit = [s for s in active if left * max(raw[s], 1e-8) / denom > cap]
        if not hit:
            for s in active:
                out[s] = left * max(raw[s], 1e-8) / denom
            break
        for s in hit:
            out[s] = cap
            left -= cap
            active.remove(s)
    total = sum(out.values())
    return {s: out[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global WAIT
    if WAIT:
        WAIT -= 1
        return
    prices = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) >= 125 and np.all(np.isfinite(close)) and np.all(close > 0):
            prices[s] = close
    if len(prices) < 12:
        WAIT = 9
        return
    ret = {s: prices[s][1:] / prices[s][:-1] - 1.0 for s in prices}
    r5 = {s: prices[s][-1] / prices[s][-6] - 1 for s in prices}
    r10 = {s: prices[s][-1] / prices[s][-11] - 1 for s in prices}
    r20 = {s: prices[s][-1] / prices[s][-21] - 1 for s in prices}
    r40 = {s: prices[s][-1] / prices[s][-41] - 1 for s in prices}
    r60 = {s: prices[s][-1] / prices[s][-61] - 1 for s in prices}
    v10 = {s: max(float(np.std(ret[s][-10:])), .008) for s in prices}
    v20 = {s: max(float(np.std(ret[s][-20:])), .008) for s in prices}
    v40 = {s: max(float(np.std(ret[s][-40:])), .008) for s in prices}
    down = {s: max(float(np.std(np.minimum(ret[s][-40:], 0))), .003) for s in prices}
    breadth = {s: float(np.mean(ret[s][-40:] > 0)) for s in prices}
    defensive = [s for s in ("XAU", "US10Y", "CN10Y") if s in prices]
    dlead = float(np.mean([r20[s] for s in defensive])) if defensive else 0.0
    stress = sum(r20[s] > 0 for s in prices) / len(prices) < .40
    vix = get_index_daily_data(symbol="VIX", days=65)
    if vix is not None:
        vc = np.asarray(vix.sort_values("date")["close"], dtype=float)[:-1]
        if len(vc) >= 22 and np.all(np.isfinite(vc)):
            stress = stress or vc[-1] > max(22., 1.25 * float(np.median(vc[-60:])))

    # Proxies preserve the ensemble's directions while using only lagged prices.
    sig = {}
    for s in prices:
        sig[s] = [
            (r20[s] - dlead) / v40[s],
            (r20[s] - .5 * max(-r40[s], 0)) / v40[s],
            (breadth[s] - .50) / v20[s] - .25 * v40[s] / v20[s],
            (breadth[s] - .50) / v20[s] - .25 * v40[s] / v20[s],
            (r20[s] - .5 * max(-r40[s], 0)) / down[s],
            (r10[s] + .5 * r20[s] - .25 * r60[s]) / v20[s],
            (r10[s] + .5 * r20[s] - .25 * r60[s]) / v20[s],
            (r20[s] - dlead) / v40[s] + breadth[s] - .5,
            (-r5[s] + .25 * r10[s]) / v10[s],
            (-r5[s] + .10 * r20[s]) / v10[s],
        ]
    ranks = [rank_cs({s: sig[s][j] for s in prices}) for j in range(10)]
    score = {s: sum(WEIGHTS[j] * (ranks[j][s] - .5) for j in range(10)) for s in prices}
    score.update({s: 0.0 for s in UNIVERSE if s not in score})
    tilt = {s: 1.0 for s in UNIVERSE}
    if stress:
        tilt.update({"XAU": 1.35, "US10Y": 1.25, "CN10Y": 1.18, "BTC": .72, "ETH": .70, "WTI": .84, "COPPER": .88})
    raw = {s: tilt[s] * max(.51 + score[s], .03) / max(v20.get(s, .02), .008) ** .20 for s in UNIVERSE}
    target = capped(raw)

    account = get_account_dict()
    assets = max(float(account.get("total_assets", 0.0)), 1.0)
    old = {s: 0.0 for s in UNIVERSE}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0:
            old[s] = max(float(p.get("market_value", 0)), 0.0) / assets
    if sum(old.values()) > .001:
        target = {s: (.35 * target[s] + .65 * old[s]) for s in UNIVERSE}
        z = sum(target.values())
        target = {s: target[s] / z for s in UNIVERSE}
    forecast = {s: float(.012 * score[s] * (.85 if stress else 1.0)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    WAIT = 9
