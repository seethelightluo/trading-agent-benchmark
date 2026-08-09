import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"clv": 0.30, "peer": 0.23, "mom20": 0.20, "rev5": 0.17, "shortrev3": 0.10}
MIN_W, MAX_W, CADENCE = 0.02, 0.14, 10
last_decision = None

def ranks(vals):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, v) for s, v in vals.items() if np.isfinite(v))
    n = len(good)
    if n > 1:
        for i, (s, _) in enumerate(good): out[s] = (i + 1.0) / n
    return out

def bounded(raw):
    w = {s: min(MAX_W, max(MIN_W, float(raw.get(s, MIN_W)))) for s in UNIVERSE}
    for _ in range(100):
        gap = 1.0 - sum(w.values())
        free = [s for s in UNIVERSE if MIN_W < w[s] < MAX_W]
        if abs(gap) < 1e-10 or not free: break
        for s in free: w[s] = min(MAX_W, max(MIN_W, w[s] + gap / len(free)))
    # Numerical cleanup while preserving the full-investment contract.
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global last_decision
    account = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=110)
        if df is None or len(df) < 30: continue
        df = df.sort_values("date").reset_index(drop=True)
        c, h, lo = [np.asarray(df[x], float) for x in ("close", "high", "low")]
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1
        data[s] = (c, h, lo, r, str(df.iloc[-1]["date"]))
    if len(data) < 12: return
    date = max(x[4] for x in data.values())
    if last_decision is not None:
        try:
            if (np.datetime64(date, "D") - np.datetime64(last_decision, "D")) / np.timedelta64(1, "D") < CADENCE: return
        except Exception: return

    factors = {k: {} for k in FACTOR_WEIGHTS}; invvol = {}; five = {}
    for s, (c, h, lo, r, _) in data.items():
        if len(r) < 22: continue
        vol = max(float(np.std(r[-20:])), 0.008); invvol[s] = 1.0 / vol
        five[s] = c[-1] / max(c[-6], 1e-12) - 1
        clv = (2*c[-3:] - h[-3:] - lo[-3:]) / np.maximum(h[-3:] - lo[-3:], 1e-12)
        factors["clv"][s] = float(np.mean(clv))
        factors["peer"][s] = five[s]
        factors["mom20"][s] = (c[-1] / max(c[-21], 1e-12) - 1) / vol
        factors["rev5"][s] = -float(np.mean(r[-5:]))
        factors["shortrev3"][s] = -float(np.mean(r[-3:]))
    med = float(np.median(list(five.values())))
    factors["peer"] = {s: v - med for s, v in five.items()}
    rr = {k: ranks(v) for k, v in factors.items()}
    score = {s: sum(FACTOR_WEIGHTS[k] * rr[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}

    # Medium-high, sideways/mildly bearish regime: defensive tilt only on confirmed trend break.
    bearish = False
    if "SPX" in data:
        c = data["SPX"][0]; bearish = c[-1] < c[-6] and c[-1] < c[-21]
    if bearish:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += 0.14
        for s in ("BTC", "ETH", "WTI"): score[s] = max(0.05, score[s] - 0.08)
    avg_iv = float(np.mean(list(invvol.values()))) if invvol else 1.0
    raw = {s: max(0.01, score[s]) * (0.75 + 0.25 * invvol.get(s, avg_iv) / avg_iv) for s in UNIVERSE}
    weights = bounded(raw)
    a = np.asarray([score[s] for s in UNIVERSE]); sd = max(float(a.std()), 1e-9)
    forecast = {s: float(0.01 * (score[s] - a.mean()) / sd) for s in UNIVERSE}
    try:
        obj = json.loads((Path(__file__).parent / "factors/factor_ensemble.json").read_text())
        ids = [x["factor_id"] for x in obj.get("selected_factors", []) if isinstance(x, dict) and x.get("factor_id")]
    except Exception: ids = []
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=ids[:10], horizon_days=10)
    last_decision = date
