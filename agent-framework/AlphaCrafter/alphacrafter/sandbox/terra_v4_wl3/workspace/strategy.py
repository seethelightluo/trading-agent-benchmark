import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_account_dict, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current admitted ensemble.  Short-horizon signals are moderated after the weak
# prior block; medium-horizon momentum is the stabilizing component.
FACTOR_WEIGHTS = {"clv": 0.30, "peer": 0.25, "reversal": 0.15, "momentum": 0.30}
MIN_W, MAX_W, REBALANCE_DAYS = 0.02, 0.15, 10
last_date = None
previous_score = None


def rank(v):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(x)) for s, x in v.items() if np.isfinite(x))
    if len(good) >= 2:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / len(good)
    return out


def cap_weights(raw):
    # Iterative projection onto simplex with conservative concentration bounds.
    w = {s: max(float(raw.get(s, 0.0)), 1e-9) for s in UNIVERSE}
    z = sum(w.values()); w = {s: x / z for s, x in w.items()}
    fixed = set()
    for _ in range(30):
        changed = False
        for s in UNIVERSE:
            if s not in fixed and w[s] < MIN_W:
                w[s] = MIN_W; fixed.add(s); changed = True
            elif s not in fixed and w[s] > MAX_W:
                w[s] = MAX_W; fixed.add(s); changed = True
        free = [s for s in UNIVERSE if s not in fixed]
        if not changed or not free: break
        rem = 1.0 - sum(w[s] for s in fixed)
        base = sum(w[s] for s in free)
        for s in free: w[s] = rem * w[s] / max(base, 1e-12)
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_date, previous_score
    market = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 25: continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df.close, dtype=float)
        h = np.asarray(df.high, dtype=float); l = np.asarray(df.low, dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        market[s] = (c, h, l, r, str(df.iloc[-1].date))
    if len(market) < 12: return
    decision = max(x[4] for x in market.values())
    if last_date is not None:
        try:
            if (np.datetime64(decision, "D") - np.datetime64(last_date, "D")) / np.timedelta64(1, "D") < REBALANCE_DAYS:
                return
        except Exception: return

    clv = {}; reversal = {}; momentum = {}; short = {}; invvol = {}
    for s, (c, h, l, r, _) in market.items():
        candle = (2*c - h - l) / np.maximum(h-l, 1e-12)
        vol = max(float(np.std(r[-20:])), 0.008)
        clv[s] = float(np.mean(candle[-5:]))
        reversal[s] = float(-np.mean(r[-5:]))
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1) / (vol + 0.01))
        short[s] = float(c[-1] / max(c[-6], 1e-12) - 1)
        invvol[s] = 1.0 / vol
    med = float(np.median(list(short.values())))
    peer = {s: x - med for s, x in short.items()}
    rr = {k: rank(v) for k, v in (("clv", clv), ("peer", peer), ("reversal", reversal), ("momentum", momentum))}
    score = {s: sum(FACTOR_WEIGHTS[k] * rr[k][s] for k in rr) for s in UNIVERSE}
    if previous_score is not None:
        score = {s: 0.70 * previous_score.get(s, score[s]) + 0.30 * score[s] for s in UNIVERSE}
    previous_score = dict(score)

    # Bullish regime remains broad. In a confirmed broad drawdown, rotate rather
    # than de-risk to cash; these three instruments are tradable defensives.
    if "SPX" in market:
        spx = market["SPX"][0]
        breadth = np.mean([market[s][0][-1] > market[s][0][-21] for s in market])
        if (spx[-1] < spx[-21] and spx[-1] < spx[-6]) or breadth < 0.40:
            for s in ("XAU", "US10Y", "CN10Y"): score[s] += 0.16
            for s in ("BTC", "ETH", "WTI"): score[s] = max(0.05, score[s] - 0.08)

    mean_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(score[s], 0.02) * (0.70 + 0.30 * invvol[s] / max(mean_iv, 1e-12)) for s in UNIVERSE}
    weights = cap_weights(raw)
    avg = float(np.mean(list(score.values()))); sd = max(float(np.std(list(score.values()))), 1e-12)
    forecast = {s: 0.01 * (score[s] - avg) / sd for s in UNIVERSE}
    factor_ids = []
    try:
        data = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        factor_ids = [x["factor_id"] for x in data.get("selected_factors", []) if isinstance(x, dict) and x.get("factor_id")][:10]
    except Exception: pass
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=10)
    last_date = decision

_ = get_account_dict if False else None
