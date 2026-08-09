import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_account_dict, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_W = {"clv": 0.30, "peer": 0.25, "rev5": 0.18, "mom": 0.21, "rev3": 0.06}
MIN_W, MAX_W = 0.015, 0.16
_last_decision = None


def ranks(x):
    good = sorted((k, v) for k, v in x.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    n = len(good)
    if n > 1:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / n
    return out


def bounded_weights(raw):
    # Iterative capped simplex projection; preserves full investment.
    w = {s: max(float(raw.get(s, 0.01)), 1e-8) for s in UNIVERSE}
    fixed, free = {}, set(UNIVERSE)
    for _ in range(50):
        rem = 1.0 - sum(fixed.values())
        scale = rem / max(sum(w[s] for s in free), 1e-12)
        clipped = False
        for s in list(free):
            v = w[s] * scale
            if v < MIN_W:
                fixed[s] = MIN_W
                free.remove(s)
                clipped = True
            elif v > MAX_W:
                fixed[s] = MAX_W
                free.remove(s)
                clipped = True
        if not clipped:
            fixed.update({s: w[s] * scale for s in free})
            break
    total = sum(fixed.values())
    return {s: fixed[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _last_decision
    account = get_account_dict()
    # The benchmark watchlist is exactly the 15 tradable instruments. Keep a
    # fixed approved universe so observation-only macro series cannot be traded.
    available = set(account.get("watch_list", UNIVERSE))
    if not set(UNIVERSE).issubset(available):
        return
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
    decision_date = max(v[4] for v in data.values())
    if _last_decision is not None:
        try:
            # Ten trading days is approximately fourteen calendar days; this
            # deterministic gate avoids repeated proposals between blocks.
            if (np.datetime64(decision_date, "D") - np.datetime64(_last_decision, "D")) / np.timedelta64(1, "D") < 14:
                return
        except Exception:
            return

    clv, peer, rev5, rev3, mom, invvol, ret5 = ({}, {}, {}, {}, {}, {}, {})
    for s, (c, h, l, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), 0.008)
        candle = (2.0 * c - h - l) / np.maximum(h - l, 1e-12)
        clv[s] = float(np.mean(candle[-3:]))
        ret5[s] = c[-1] / max(c[-6], 1e-12) - 1.0
        rev5[s] = -float(np.mean(r[-5:]))
        rev3[s] = -float(np.mean(r[-3:]))
        mom[s] = (c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01)
        invvol[s] = 1.0 / vol
    med = float(np.median(list(ret5.values())))
    peer = {s: ret5[s] - med for s in ret5}
    rr = {k: ranks(v) for k, v in (("clv", clv), ("peer", peer), ("rev5", rev5), ("mom", mom), ("rev3", rev3))}
    score = {s: sum(FACTOR_W[k] * rr[k].get(s, 0.5) for k in FACTOR_W) for s in UNIVERSE}

    # Bear/high-volatility posture: full investment, defensive tradable tilt.
    breadth = sum(c[-1] < c[-6] for c, *_ in data.values()) / len(data)
    if breadth >= 0.55:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += 0.16
        for s in ("BTC", "ETH", "WTI"):
            score[s] = max(0.05, score[s] - 0.09)
    avg_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(score[s], 0.02) * (0.78 + 0.22 * invvol.get(s, avg_iv) / max(avg_iv, 1e-12)) for s in UNIVERSE}
    weights = bounded_weights(raw)
    z = {s: (score[s] - np.mean(list(score.values()))) / max(np.std(list(score.values())), 1e-12) for s in UNIVERSE}
    forecasts = {s: float(0.01 * z[s]) for s in UNIVERSE}
    rebalance_to_weights(
        weights, forecast_returns=forecasts,
        factor_ids=["miner_3_20260716_clv_1d", "miner_1_20260716_peer_median_leadlag_5d", "miner_1_20260716_short_term_reversal_5d", "miner_2_20260716_risk_adjusted_momentum_20d", "miner_2_20260716_short_reversal_3d"],
        horizon_days=10)
    _last_decision = decision_date
