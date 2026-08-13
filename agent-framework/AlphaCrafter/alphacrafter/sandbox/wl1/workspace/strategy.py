import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_account_dict, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20320624_path_stability_lead_10d",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "breadth_vol_quality_40d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "macro_stress_resilience_20d",
    "miner_2_20321125_quiet_reversal_40d_q60",
    "miner_3_20310904_recovery_pullback_20d",
    "miner_2_20310626_trend_acceleration_quality",
]
FACTOR_WEIGHTS = np.array([.16, .13, .12, .10, .10, .14, .04, .12, .09])
WAIT = 0

def rank(values):
    ordered = sorted(values, key=lambda s: (values[s], s))
    n = float(len(ordered))
    return {s: (i + 1.0) / n for i, s in enumerate(ordered)}

def capped(raw, cap=.15):
    out = {s: 0.0 for s in UNIVERSE}; active = set(UNIVERSE); left = 1.0
    while active:
        denom = sum(max(raw[s], 1e-8) for s in active)
        hit = [s for s in active if left * max(raw[s], 1e-8) / denom > cap]
        if not hit:
            for s in active: out[s] = left * max(raw[s], 1e-8) / denom
            break
        for s in hit:
            out[s] = cap; left -= cap; active.remove(s)
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
        if df is None: continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(close) > 1: close = close[:-1]
        if len(close) >= 125 and np.all(np.isfinite(close)) and np.all(close > 0): prices[s] = close
    if len(prices) < 12:
        WAIT = 9
        return
    returns = {s: prices[s][1:] / prices[s][:-1] - 1.0 for s in prices}
    def momentum(s, n): return prices[s][-1] / prices[s][-n-1] - 1.0
    v20 = {s: max(float(np.std(returns[s][-20:])), .008) for s in prices}
    v40 = {s: max(float(np.std(returns[s][-40:])), .008) for s in prices}
    breadth = {s: float(np.mean(returns[s][-40:] > 0)) for s in prices}
    defensive = [s for s in ("XAU", "US10Y", "CN10Y") if s in prices]
    dlead = float(np.mean([momentum(s, 20) for s in defensive])) if defensive else 0.0
    signals = {}
    for s in prices:
        r10, r20, r40, r60 = [momentum(s, n) for n in (10, 20, 40, 60)]
        path = (breadth[s] - .5) / v20[s] - .25 * v40[s] / v20[s]
        relative = (r20 - dlead) / v40[s]
        pullback = (r10 - .5 * r20 - .25 * r40) / v20[s]
        signals[s] = [path, (r20 - .25 * max(-r40, 0)) / v40[s],
                      (r20 - .25 * max(-r40, 0)) / v40[s], path,
                      relative, relative, pullback, pullback,
                      (r10 + .5 * r20 - .25 * r60) / v20[s]]
    ranks = [rank({s: signals[s][j] for s in prices}) for j in range(9)]
    score = {s: sum(FACTOR_WEIGHTS[j] * (ranks[j][s] - .5) for j in range(9)) for s in prices}
    score.update({s: 0.0 for s in UNIVERSE if s not in score})
    stress = sum(momentum(s, 20) > 0 for s in prices) / len(prices) < .40
    tilt = {s: 1.0 for s in UNIVERSE}
    if stress:
        tilt.update({"XAU": 1.35, "US10Y": 1.25, "CN10Y": 1.18,
                     "BTC": .72, "ETH": .70, "WTI": .84, "COPPER": .88})
    raw = {s: tilt[s] * max(.51 + score[s], .03) / max(v20.get(s, .02), .008) ** .20 for s in UNIVERSE}
    target = capped(raw)
    account = get_account_dict()
    assets = max(float(account.get("total_assets", 0)), 1.0)
    old = {s: 0.0 for s in UNIVERSE}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0:
            old[s] = max(float(p.get("market_value", 0)), 0.0) / assets
    if sum(old.values()) > .001:
        target = {s: .35 * target[s] + .65 * old[s] for s in UNIVERSE}
        z = sum(target.values())
        target = {s: target[s] / z for s in UNIVERSE}
    forecast = {s: float(.012 * score[s] * (.85 if stress else 1.0)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    WAIT = 9
