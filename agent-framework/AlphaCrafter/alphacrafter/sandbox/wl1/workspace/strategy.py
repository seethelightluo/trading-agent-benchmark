import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"vix": .16, "clv": .11, "rvolmom": .14, "path": .12, "resid": .10, "downmom": .09, "shock": .07, "disp": .09, "highvol": .05, "persist": .07}
FACTOR_IDS = [
    "miner_1_20260730_vix_state_reversal_5d", "miner_3_20260730_vix_conditioned_clv_1d",
    "miner_1_20260730_residual_vol_scaled_momentum_10d", "miner_1_20260730_path_efficiency_10d",
    "miner_2_20260730_residual_momentum_10d", "miner_1_20260730_downside_quality_momentum_10d",
    "miner_1_20260730_residual_shock_blend_10d", "miner_1_20260730_dispersion_conditioned_reversal_1d",
    "miner_2_20260730_highvol_reversal_5d", "miner_2_20260730_slow_rank_persistence_20d"
]
_last_decision_date = None

def rank(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    for i, (s, _) in enumerate(valid): out[s] = (i + 1.) / max(len(valid), 1)
    return out

def bounded_weights(raw):
    w = {s: max(float(raw.get(s, 1. / 15.)), 1e-9) for s in UNIVERSE}
    for _ in range(40):
        low = {s for s in UNIVERSE if w[s] < .015}; high = {s for s in UNIVERSE if w[s] > .16}
        for s in low: w[s] = .015
        for s in high: w[s] = .16
        fixed = low | high; free = [s for s in UNIVERSE if s not in fixed]
        remain = 1. - sum(w[s] for s in fixed)
        if not free or remain <= 0: break
        scale = remain / sum(w[s] for s in free)
        for s in free: w[s] *= scale
        if all(.015 - 1e-9 <= w[s] <= .16 + 1e-9 for s in UNIVERSE): break
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _last_decision_date
    prices = {}
    dates = []
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=140)
        if df is not None and len(df) >= 45:
            df = df.sort_values("date"); prices[s] = np.asarray(df["close"], dtype=float)
            dates.append(str(df.iloc[-1]["date"]))
    if len(prices) < 12 or not dates: return
    decision_date = max(dates)
    if _last_decision_date is not None:
        try:
            if (np.datetime64(decision_date) - np.datetime64(_last_decision_date)) / np.timedelta64(1, "D") < 10: return
        except Exception: return
    def ret(c, n): return c[-1] / max(c[-n-1], 1e-12) - 1.
    r10 = {s: ret(c, 10) for s, c in prices.items()}; r20 = {s: ret(c, 20) for s, c in prices.items()}
    med10 = float(np.median(list(r10.values()))); fac = {k: {} for k in FACTOR_WEIGHTS}; vols = {}
    cross = np.array(list(r10.values()))
    dispersion = max(float(np.std(cross)), .001)
    for s, c in prices.items():
        d = c[1:] / np.maximum(c[:-1], 1e-12) - 1.; vol = max(float(np.std(d[-20:])), .008)
        downvol = max(float(np.std(np.minimum(d[-30:], 0.))), .006); vols[s] = vol
        rev = -float(np.mean(d[-5:]))
        fac["vix"][s] = rev * np.clip(vol / .018, .5, 2.)
        fac["clv"][s] = -.65 * float(d[-1]) * np.clip(vol / .018, .5, 2.) - .35 * rev
        fac["rvolmom"][s] = (r10[s] - med10) / vol
        fac["path"][s] = np.sign(r10[s]) * abs(r10[s]) / max(float(np.sum(abs(d[-10:]))), .01)
        fac["resid"][s] = r10[s] - med10; fac["downmom"][s] = ret(c, 20) / downvol
        shock = -float(d[-1]) * np.clip(abs(float(d[-1])) / (vol + .002), 0., 3.)
        fac["shock"][s] = .6 * (r10[s] - med10) + .4 * shock
        fac["disp"][s] = -float(d[-1]) * dispersion / max(vol, .008)
        fac["highvol"][s] = rev * np.clip(vol / .020, .5, 2.5)
        sub = [ret(c, min(5*j, 20)) for j in range(1, 5)]
        fac["persist"][s] = sum(x > 0 for x in sub) + .25 * np.sign(r20[s])
    scores = {s: sum(FACTOR_WEIGHTS[k] * rank(fac[k])[s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}
    breadth = float(np.mean([c[-1] > c[-21] for c in prices.values()])); spx = prices.get("SPX")
    bear = spx is not None and spx[-1] < spx[-21] and spx[-1] < spx[-6]
    if bear or breadth < .40:
        for s in ("XAU", "US10Y", "CN10Y"): scores[s] += .12
        for s in ("BTC", "ETH", "WTI"): scores[s] -= .05
    elif breadth > .67:
        for s in ("SPX", "NDX", "SOX", "000300.SH"): scores[s] += .035
    mean_vol = float(np.mean(list(vols.values())))
    raw = {s: max(scores[s], .03) * (.85 + .15 * mean_vol / max(vols[s], .008)) for s in UNIVERSE}
    targets = bounded_weights(raw); forecasts = {s: .06 * (scores[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(targets, forecast_returns=forecasts, factor_ids=FACTOR_IDS, horizon_days=10)
    _last_decision_date = decision_date
