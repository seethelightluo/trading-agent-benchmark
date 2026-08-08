import json
import math
import numpy as np
from alphacrafter.sim.utils import register_hook, add_order, get_stock_daily_data, get_account_dict

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
MIN_W, MAX_W, CADENCE = 0.015, 0.16, 10
FACTORS = {"miner_3_clv_1d": .3395, "peer_median_leadlag_5d": .2612, "short_term_reversal_5d": .2012, "miner_2_risk_adjusted_momentum_20d": .1981}
try:
    with open("factors/factor_ensemble.json", encoding="utf-8") as f:
        selected = json.load(f).get("selected_factors", [])[:10]
    if selected:
        FACTORS = {x["factor_id"]: float(x.get("weight", 0)) * float(x.get("direction", 1)) for x in selected}
except Exception:
    pass
_day = 0

def xrank(values):
    valid = [v for v in values.values() if np.isfinite(v)]
    return {s: (sum(v <= values[s] for v in valid) / len(valid) if valid and np.isfinite(values[s]) else .5) for s in UNIVERSE}

def allocate(score):
    w = {s: MIN_W for s in UNIVERSE}
    rem, free = 1 - len(UNIVERSE) * MIN_W, set(UNIVERSE)
    while free and rem > 1e-10:
        z = sum(max(score[s], 1e-8) for s in free)
        capped = [s for s in free if rem * max(score[s], 1e-8) / z > MAX_W - MIN_W]
        if not capped:
            for s in free: w[s] += rem * max(score[s], 1e-8) / z
            break
        for s in capped:
            w[s] = MAX_W; rem -= MAX_W - MIN_W; free.remove(s)
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_allocator():
    global _day
    _day += 1
    if (_day - 1) % CADENCE != 0: return
    account = get_account_dict()
    positions = {p.get("symbol"): p for p in account.get("positions", [])}
    closes, vols, highs, lows = {}, {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=125)
        if df is None or len(df) < 45: continue
        df = df.sort_values("date")
        p = np.asarray(df["close"], dtype=float)
        if len(p) < 42 or np.any(~np.isfinite(p)) or np.any(p <= 0): continue
        closes[s] = p; highs[s] = float(df.iloc[-1]["high"]); lows[s] = float(df.iloc[-1]["low"])
        rets = p[1:] / p[:-1] - 1
        vols[s] = max(float(np.std(rets[-20:]) * math.sqrt(252)), .05)
    prices = {s: float(closes[s][-1]) if s in closes else float((positions.get(s) or {}).get("current_price", 0) or 0) for s in UNIVERSE}
    r5, reversal, momentum, clv = {}, {}, {}, {}
    for s in UNIVERSE:
        p = closes.get(s)
        if p is None: r5[s] = reversal[s] = momentum[s] = clv[s] = np.nan; continue
        r5[s] = p[-1] / p[-6] - 1; reversal[s] = -r5[s]
        momentum[s] = .5 * (p[-1] / p[-21] - 1) + .5 * (p[-1] / p[-41] - 1)
        d = highs[s] - lows[s]; clv[s] = (2 * p[-1] - highs[s] - lows[s]) / d if d > 0 else 0
    median = float(np.nanmedian(list(r5.values()))) if any(np.isfinite(v) for v in r5.values()) else 0
    raw = {"miner_3_clv_1d": clv, "peer_median_leadlag_5d": {s: r5[s] - median for s in UNIVERSE}, "short_term_reversal_5d": reversal, "miner_2_risk_adjusted_momentum_20d": {s: momentum[s] / vols.get(s, .5) if np.isfinite(momentum[s]) else np.nan for s in UNIVERSE}}
    ranked = {k: xrank(v) for k, v in raw.items()}; invvol = xrank({s: 1 / vols.get(s, .5) for s in UNIVERSE})
    spx = closes.get("SPX"); bearish = spx is not None and spx[-1] / spx[-31] < .995
    score = {}
    for s in UNIVERSE:
        value = .78 * sum(FACTORS.get(k, 0) * ranked[k][s] for k in raw) + .22 * invvol[s]
        if vols.get(s, 0) > .75: value *= .88
        if bearish and s in DEFENSIVE: value += .12
        score[s] = max(value, 1e-8)
    weights = allocate(score); total = float(account.get("total_assets", 0) or 0)
    targets = {s: total * weights[s] / prices[s] for s in UNIVERSE if prices[s] > 0}
    for s, target in targets.items():
        pos = positions.get(s, {}); current = float(pos.get("quantity", 0) or 0); available = float(pos.get("available_quantity", current) or 0)
        qty = min(max(current - target, 0), max(available, 0))
        if qty > 1: add_order(symbol=s, order_type="SELL", price=prices[s], quantity=int(qty))
    for s, target in targets.items():
        current = float((positions.get(s) or {}).get("quantity", 0) or 0); qty = max(target - current, 0)
        if qty > 1: add_order(symbol=s, order_type="BUY", price=prices[s], quantity=int(qty))
