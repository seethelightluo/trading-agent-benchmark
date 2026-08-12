import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, get_account_dict, rebalance_to_weights)

U = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
     "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
F = [
    "miner_3_20290628_residual_momentum_volscaled_20",
    "miner_3_20310724_relative_momentum_acceleration_20d",
    "recovery_downside_trend_20_60",
    "miner_1_20280323_vix_conditioned_efficiency_trend",
    "breadth_vol_quality_40d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "drift_vol_compression_20_10_40",
    "miner_1_20311113_residual_momentum_20d",
    "macro_stress_resilience_20d",
]
W = [0.18, 0.17, 0.14, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04]
WAIT = 0


def rank(x):
    a = sorted(x, key=lambda s: (x[s], s))
    n = float(len(a))
    return {s: (i + 1.0) / n for i, s in enumerate(a)}


def capped(raw, cap=0.15):
    out, free, left = {}, set(U), 1.0
    while free:
        z = sum(max(raw[s], 1e-8) for s in free)
        hit = [s for s in free if left * max(raw[s], 1e-8) / z > cap]
        if not hit:
            for s in free:
                out[s] = left * max(raw[s], 1e-8) / z
            break
        for s in hit:
            out[s] = cap
            left -= cap
            free.remove(s)
    z = sum(out.values())
    return {s: out.get(s, 0.0) / z for s in U}


@register_hook
def cross_asset_strategy():
    global WAIT
    if WAIT:
        WAIT -= 1
        return

    px = {}
    for s in U:
        d = get_stock_daily_data(symbol=s, days=280)
        if d is None:
            continue
        c = np.asarray(d.sort_values("date")["close"], dtype=float)[:-1]
        if len(c) >= 125 and np.all(np.isfinite(c)) and np.all(c > 0):
            px[s] = c
    names = [s for s in U if s in px]
    if len(names) < 12:
        WAIT = 9
        return

    ret = {s: px[s][1:] / px[s][:-1] - 1.0 for s in names}
    r10 = {s: px[s][-1] / px[s][-11] - 1.0 for s in names}
    r20 = {s: px[s][-1] / px[s][-21] - 1.0 for s in names}
    r60 = {s: px[s][-1] / px[s][-61] - 1.0 for s in names}
    v20 = {s: max(float(np.std(ret[s][-20:])), .008) for s in names}
    v40 = {s: max(float(np.std(ret[s][-40:])), .008) for s in names}
    med20 = float(np.median(list(r20.values())))
    breadth = float(np.mean([r20[s] > 0 for s in names]))

    vix, vix_med = 20.0, 20.0
    vd = get_index_daily_data(symbol="VIX", days=65)
    if vd is not None:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)[:-1]
        if len(vc) >= 22 and np.all(np.isfinite(vc)):
            vix, vix_med = float(vc[-1]), float(np.median(vc[-60:]))
    stressed = breadth < .40 or vix > max(22.0, 1.25 * vix_med)
    defensive = {"XAU", "US10Y", "CN10Y"}
    dlead = float(np.mean([r20[s] for s in defensive if s in r20]))

    vals = {s: [] for s in names}
    for s in names:
        x20, x40 = ret[s][-20:], ret[s][-40:]
        downside = max(float(np.mean(np.maximum(-x40, 0.0))), .002)
        path_eff = abs(r20[s]) / max(float(np.sum(np.abs(x20))), .02)
        breadth40 = float(np.mean(x40 > 0.0))
        # These are lagged, transparent proxies for the persisted factor IDs.
        vals[s] = [
            (r20[s] - .35 * r10[s]) / v20[s],
            (r20[s] - r10[s]) / v20[s],
            (r20[s] - .35 * r60[s]) / downside,
            (1.0 if not stressed else -1.0) * (r20[s] / v20[s]) * path_eff,
            ((breadth40 - .5) * 2.0) / max(v40[s] * np.sqrt(252), .01),
            (r20[s] - dlead) / max(v40[s], .008),
            (r20[s] - r10[s]) / max(v40[s], .008),
            (r20[s] - med20) / v20[s],
            ((r20[s] - dlead) / max(v40[s], .008)) if stressed else breadth40,
        ]
    rr = [rank({s: vals[s][j] for s in names}) for j in range(len(F))]
    score = {s: sum(W[j] * (rr[j][s] - .5) for j in range(len(F))) for s in names}
    score.update({s: 0.0 for s in U if s not in score})

    tilt = {s: 1.0 for s in U}
    if stressed:
        tilt.update({"XAU": 1.28, "US10Y": 1.20, "CN10Y": 1.12,
                     "BTC": .82, "ETH": .80, "WTI": .90, "COPPER": .92})
    raw = {s: tilt[s] * max(.51 + score[s], .03) / max(v20.get(s, .02), .008) ** .20 for s in U}
    target = capped(raw)

    account = get_account_dict()
    assets = max(float(account.get("total_assets", 0.0)), 1.0)
    old = {s: 0.0 for s in U}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0.0)) > 0:
            old[s] = max(float(p.get("market_value", 0.0)), 0.0) / assets
    if sum(old.values()) > .001:
        target = {s: (.35 * target[s] + .65 * old[s]) for s in U}
        z = sum(target.values())
        target = {s: target[s] / z for s in U}

    # Deterministic 10-day forecast is required by the migration-cost gate.
    forecast = {s: float(.012 * score[s] * (.80 if stressed else 1.0)) for s in U}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=F, horizon_days=10)
    WAIT = 9
