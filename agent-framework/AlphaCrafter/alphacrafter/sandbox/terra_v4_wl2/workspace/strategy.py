import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble (7 active factors; all are price/cross-asset signals).
FACTOR_WEIGHTS = {"clv": .27, "peer": .21, "mom20": .17, "rev5": .14,
                  "session": .10, "rev3": .06, "shortrev3": .05}
MIN_W, MAX_W, CADENCE_DAYS = .015, .16, 10
last_decision = None


def rank(values):
    out = {s: .5 for s in UNIVERSE}
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1.) / len(valid)
    return out


def bounded_simplex(raw):
    # Apply bounds, then redistribute residual only among non-bound names.
    w = {s: min(MAX_W, max(MIN_W, float(raw.get(s, MIN_W)))) for s in UNIVERSE}
    for _ in range(100):
        gap = 1. - sum(w.values())
        if abs(gap) < 1e-10:
            break
        free = [s for s in UNIVERSE if MIN_W < w[s] < MAX_W]
        if not free:
            break
        delta = gap / len(free)
        for s in free:
            w[s] = min(MAX_W, max(MIN_W, w[s] + delta))
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_decision
    account = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=110)
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c, o = np.asarray(df["close"], float), np.asarray(df["open"], float)
        h, lo = np.asarray(df["high"], float), np.asarray(df["low"], float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.
        data[s] = (c, o, h, lo, r, str(df.iloc[-1]["date"]))
    if len(data) < 12:
        return
    decision_date = max(v[5] for v in data.values())
    if last_decision is not None:
        try:
            elapsed = (np.datetime64(decision_date, "D") - np.datetime64(last_decision, "D")) / np.timedelta64(1, "D")
            if elapsed < CADENCE_DAYS:
                return
        except Exception:
            return

    clv, peer, mom20, rev5, session, rev3, shortrev3, invvol = ({}, {}, {}, {}, {}, {}, {}, {})
    five_day = {}
    for s, (c, o, h, lo, r, _) in data.items():
        if len(r) < 22:
            continue
        vol = max(float(np.std(r[-20:])), .008)
        invvol[s] = 1. / vol
        five_day[s] = c[-1] / max(c[-6], 1e-12) - 1.
        loc = (2*c[-1] - h[-1] - lo[-1]) / max(h[-1] - lo[-1], 1e-12)
        prevloc = (2*c[-2] - h[-2] - lo[-2]) / max(h[-2] - lo[-2], 1e-12)
        clv[s] = .65 * loc + .35 * prevloc
        rev5[s] = -float(np.mean(r[-5:]))
        rev3[s] = -float(np.mean(r[-3:]))
        shortrev3[s] = rev3[s]
        mom20[s] = (c[-1] / max(c[-21], 1e-12) - 1.) / vol
        overnight = o[-1] / max(c[-2], 1e-12) - 1.
        intraday = c[-1] / max(o[-1], 1e-12) - 1.
        session[s] = -(overnight - intraday)
    med = float(np.median(list(five_day.values())))
    peer = {s: v - med for s, v in five_day.items()}
    raw_factors = {"clv": clv, "peer": peer, "mom20": mom20, "rev5": rev5,
                   "session": session, "rev3": rev3, "shortrev3": shortrev3}
    ranked = {k: rank(v) for k, v in raw_factors.items()}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranked[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}

    # Strong trend: retain broad participation. If trend breaks, rotate toward
    # defensive tradable assets rather than cash.
    if "SPX" in data:
        spx = data["SPX"][0]
        bearish = spx[-1] < spx[-6] or spx[-1] < spx[-21]
        if bearish:
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += .12
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(.05, score[s] - .07)

    avg_inv = float(np.mean(list(invvol.values())))
    raw = {s: max(.01, score[s]) * (.78 + .22 * invvol.get(s, avg_inv) / avg_inv) for s in UNIVERSE}
    weights = bounded_simplex(raw)
    vals = np.asarray([score[s] for s in UNIVERSE])
    sd = max(float(vals.std()), 1e-9)
    forecast = {s: float(.01 * (score[s] - vals.mean()) / sd) for s in UNIVERSE}

    factor_ids = []
    try:
        ensemble = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text(encoding="utf-8"))
        factor_ids = [x["factor_id"] for x in ensemble.get("selected_factors", [])
                      if isinstance(x, dict) and x.get("factor_id")]
    except Exception:
        pass
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids[:10], horizon_days=10)
    last_decision = decision_date
