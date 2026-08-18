import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
REBALANCE_DAYS = 10
MIN_W, MAX_W = 0.015, 0.16
last_date = None

# The persisted ensemble is the source of truth for IDs, weights and directions.
def ensemble():
    p = Path(__file__).parent / "factors" / "factor_ensemble.json"
    try:
        x = json.loads(p.read_text())
        rows = x.get("selected_factors", [])[:10]
        return rows if rows else []
    except (OSError, ValueError, TypeError):
        return []


def rank(xs):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in xs.items() if np.isfinite(v))
    if len(good) >= 2:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / len(good)
    return out


def box_weights(raw):
    # Iterative capped simplex projection; all 15 assets remain represented.
    w = {s: max(float(raw.get(s, 0.0)), 1e-12) for s in UNIVERSE}
    w = {s: v / sum(w.values()) for s, v in w.items()}
    for _ in range(50):
        low = [s for s in UNIVERSE if w[s] < MIN_W]
        high = [s for s in UNIVERSE if w[s] > MAX_W]
        fixed = set(low + high)
        if not fixed:
            break
        for s in low: w[s] = MIN_W
        for s in high: w[s] = MAX_W
        free = [s for s in UNIVERSE if s not in fixed]
        rem = 1.0 - sum(w[s] for s in fixed)
        if not free or rem <= 0: break
        z = sum(w[s] for s in free)
        for s in free: w[s] = rem * w[s] / z
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_date
    rows = ensemble()
    if not rows:
        return
    acc = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 25:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df.close, dtype=float)
        h = np.asarray(df.high, dtype=float)
        l = np.asarray(df.low, dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1
        data[s] = (c, h, l, r, str(df.iloc[-1].date))
    if len(data) < 12:
        return
    decision = max(v[4] for v in data.values())
    if last_date is not None:
        gap = (np.datetime64(decision, "D") - np.datetime64(last_date, "D")) / np.timedelta64(1, "D")
        if gap < REBALANCE_DAYS:
            return

    rawf = {}
    invvol = {}
    five = {}
    for s, (c, h, l, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), 0.008)
        invvol[s] = 1.0 / vol
        five[s] = c[-1] / max(c[-6], 1e-12) - 1.0
        rawf.setdefault("miner_3_clv_1d", {})[s] = np.mean((2*c[-3:] - h[-3:] - l[-3:]) / np.maximum(h[-3:]-l[-3:], 1e-12))
        rawf.setdefault("short_term_reversal_5d", {})[s] = -np.mean(r[-5:])
        rawf.setdefault("miner_2_risk_adjusted_momentum_20d", {})[s] = (c[-1]/max(c[-21],1e-12)-1)/ (vol+0.01)
    med = np.median(list(five.values()))
    rawf["peer_median_leadlag_5d"] = {s: v-med for s,v in five.items()}
    ranked = {f: rank(v) for f,v in rawf.items()}
    score = {s: 0.5 for s in UNIVERSE}
    for row in rows:
        fid = str(row.get("factor_id"))
        w = float(row.get("weight", 0.0)); d = float(row.get("direction", 1.0))
        if fid in ranked:
            for s in UNIVERSE: score[s] += w * d * (ranked[fid].get(s, 0.5)-0.5)
    # Moderate bullish regime: retain risk assets but use inverse volatility.
    mean_iv = np.mean(list(invvol.values()))
    raw = {s: max(0.01, score[s]) * (0.78 + 0.22*invvol.get(s, mean_iv)/mean_iv) for s in UNIVERSE}
    weights = box_weights(raw)
    vals = np.array([score[s] for s in UNIVERSE])
    z = max(float(vals.std()), 1e-12)
    forecast = {s: float(0.01*(score[s]-vals.mean())/z) for s in UNIVERSE}
    factor_ids = [str(x.get("factor_id")) for x in rows]
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=10)
    last_date = decision
