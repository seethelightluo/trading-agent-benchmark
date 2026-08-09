import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"clv": 0.32, "peer": 0.25, "rev5": 0.20, "mom20": 0.15, "rev3": 0.08}
MIN_W, MAX_W, CADENCE = 0.015, 0.16, 10
last_decision = None


def ranks(values):
    out = {s: 0.5 for s in UNIVERSE}
    valid = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    n = len(valid)
    if n > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1.0) / n
    return out


def box_weights(raw):
    # Iterative capped/floored simplex projection for the complete portfolio.
    w = {s: max(MIN_W, min(MAX_W, float(raw.get(s, MIN_W)))) for s in UNIVERSE}
    for _ in range(100):
        err = 1.0 - sum(w.values())
        if abs(err) < 1e-10:
            break
        free = [s for s in UNIVERSE if MIN_W + 1e-9 < w[s] < MAX_W - 1e-9]
        if not free:
            break
        add = err / len(free)
        for s in free:
            w[s] = max(MIN_W, min(MAX_W, w[s] + add))
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_decision
    account = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=100)
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        data[s] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(data) < 12:
        return
    date = max(v[4] for v in data.values())
    if last_decision is not None:
        try:
            elapsed = (np.datetime64(date, "D") - np.datetime64(last_decision, "D")) / np.timedelta64(1, "D")
            if elapsed < CADENCE:
                return
        except Exception:
            return

    clv, ret5, rev5, mom20, rev3, invvol = {}, {}, {}, {}, {}, {}
    for s, (c, h, l, r, _) in data.items():
        if len(r) < 22:
            continue
        rng = np.maximum(h[-1] - l[-1], 1e-12)
        clv[s] = float((2.0 * c[-1] - h[-1] - l[-1]) / rng)
        ret5[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
        rev5[s] = -float(np.mean(r[-5:]))
        rev3[s] = -float(np.mean(r[-3:]))
        vol = max(float(np.std(r[-20:])), 0.008)
        mom20[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / vol)
        invvol[s] = 1.0 / vol
    median5 = float(np.median(list(ret5.values())))
    peer = {s: v - median5 for s, v in ret5.items()}
    rr = {k: ranks(v) for k, v in (("clv", clv), ("peer", peer), ("rev5", rev5), ("mom20", mom20), ("rev3", rev3))}
    score = {s: sum(FACTOR_WEIGHTS[k] * rr[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}

    # Medium-high, sideways/bearish regime: retain full investment but favor defensives.
    if "SPX" in data:
        spx = data["SPX"][0]
        if spx[-1] < spx[-6] or spx[-1] < spx[-21]:
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += 0.12
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(0.05, score[s] - 0.07)
    avg_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(0.01, score[s]) * (0.78 + 0.22 * invvol.get(s, avg_iv) / avg_iv) for s in UNIVERSE}
    weights = box_weights(raw)
    a = np.array([score[s] for s in UNIVERSE])
    sd = max(float(a.std()), 1e-9)
    forecast = {s: 0.01 * (score[s] - float(a.mean())) / sd for s in UNIVERSE}

    factor_ids = []
    try:
        obj = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text(encoding="utf-8"))
        factor_ids = [x["factor_id"] for x in obj.get("selected_factors", []) if isinstance(x, dict) and x.get("factor_id")]
    except Exception:
        pass
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids[:10], horizon_days=10)
    last_decision = date
