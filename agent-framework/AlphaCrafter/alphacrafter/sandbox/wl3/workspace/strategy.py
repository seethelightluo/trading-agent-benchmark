import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
FACTOR_IDS = [
    "miner_2_20280907_breakout_failure_reversal",
    "miner_2_20270520_dispersion_conditioned_reversal",
    "miner_3_20270211_volstate_reversal_3d",
    "miner_2_20280615_volmanaged_consistency30",
    "miner_2_20280727_breakout_distance120",
    "miner_3_20280601_beta_residual_momentum20",
]
FACTOR_WEIGHTS = [0.32, 0.25, 0.20, 0.08, 0.08, 0.07]
FACTOR_DIRECTIONS = [1, 1, 1, 1, 1, 1]
CADENCE = 10
_day = 0
_previous = None

def ranks(vals):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in vals.items() if np.isfinite(v))
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(1, len(good))
    return out

def capped(raw, cap=0.22):
    w = np.array([max(float(raw.get(s, 0.01)), 0.005) for s in UNIVERSE])
    w /= w.sum()
    for _ in range(30):
        over = w > cap
        if not np.any(over): break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        if np.any(~over): w[~over] += excess * w[~over] / max(float(w[~over].sum()), 1e-12)
    w /= w.sum()
    return dict(zip(UNIVERSE, w.tolist()))

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    stats = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=330)
        if df is None or len(df) < 140: continue
        df = df.sort_values("date")
        c = np.asarray(df["close"], float)[:-1]
        h = np.asarray(df["high"], float)[:-1]
        l = np.asarray(df["low"], float)[:-1]
        if len(c) < 125 or np.any(~np.isfinite(c)) or np.any(c <= 0): continue
        r = c[1:] / c[:-1] - 1.0
        v3, v20, v60 = max(float(np.std(r[-3:])), .004), max(float(np.std(r[-20:])), .006), max(float(np.std(r[-60:])), .006)
        m3, m20, m30 = float(np.prod(1+r[-3:])-1), float(np.prod(1+r[-20:])-1), float(np.prod(1+r[-30:])-1)
        atr = max(float(np.mean((h[-10:]-l[-10:])/c[-10:])), .003)
        hi20, hi60, hi120 = [max(float(np.max(c[-n:])), 1e-12) for n in (20,60,120)]
        stats[s] = {"r":r, "vol":v20,
          "failure":((c[-1]/hi60-1)-(c[-1]/hi20-1))/atr,
          "dispersion":-m3/v3*(1+abs(m20)/max(v20,.01)),
          "volstate":-m3/v3*(1+min(v20/v60,2)),
          "consistency":(m30/v20)*float(np.mean(r[-30:]>0)),
          "breakout120":(c[-1]/hi120-1)/max(atr,.005), "momentum":m20}
    if len(stats) < 10: return
    bench = stats.get("000300.SH", stats.get("SPX"))["r"]
    for x in stats.values():
        n = min(60, len(x["r"]), len(bench))
        beta = float(np.cov(x["r"][-n:], bench[-n:], ddof=0)[0,1]) / max(float(np.var(bench[-n:])),1e-8)
        x["residual"] = float(np.sum(x["r"][-20:] - beta*bench[-20:]))
    keys = ["failure","dispersion","volstate","consistency","breakout120","residual"]
    rr = {k:ranks({s:x[k] for s,x in stats.items()}) for k in keys}
    score = {s:sum(w*rr[k][s] for w,k in zip(FACTOR_WEIGHTS,keys)) for s in UNIVERSE}
    if _previous is not None: score = {s:.8*score[s]+.2*_previous[s] for s in UNIVERSE}
    _previous = dict(score)
    median_vol = float(np.median([x["vol"] for x in stats.values()]))
    breadth = float(np.mean([x["momentum"] > 0 for x in stats.values()]))
    stressed = median_vol > .015 or breadth < .40
    inv_mean = float(np.mean([1/x["vol"] for x in stats.values()]))
    raw = {}
    for s in UNIVERSE:
        x = stats.get(s); iv = 1 if x is None else np.clip((1/x["vol"])/inv_mean,.75,1.15)
        tilt = (1.55 if s in DEFENSIVE else .50 if s in RISKY else .85) if stressed else (1.15 if s in DEFENSIVE else .85 if s in RISKY else 1)
        raw[s] = max(score[s],.04)*(.85+.15*iv)*tilt
    target = capped(raw)
    forecast = {s:float(np.clip((score[s]-.5)*.06,-.03,.03)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS)

assert len(UNIVERSE)==15 and len(FACTOR_IDS)<=10
assert abs(sum(FACTOR_WEIGHTS)-1)<1e-9
assert set(UNIVERSE).isdisjoint({"DXY","USDCNY","USDJPY","EURUSD","VIX"})
