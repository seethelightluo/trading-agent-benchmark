import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current screener ensemble; seven active factors, within the ten-factor cap.
FACTOR_WEIGHTS = {"clv": .20, "peer": .18, "vix_rev": .16, "momentum": .16,
                  "vix_clv": .12, "shock": .10, "reversal": .08}
MIN_W, MAX_W, REBALANCE_DAYS = .015, .16, 10
last_decision = None

def ranks(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    if len(good) > 1:
        for i, (s, _) in enumerate(good): out[s] = (i + 1.) / len(good)
    return out

def bounded_simplex(raw):
    # Project positive scores to a complete, bounded, unit-sum long-only vector.
    w = {s: max(float(raw.get(s, 0.0)), 1e-12) for s in UNIVERSE}
    for _ in range(50):
        fixed = {s for s in UNIVERSE if w[s] <= MIN_W or w[s] >= MAX_W}
        for s in fixed: w[s] = min(MAX_W, max(MIN_W, w[s]))
        free = [s for s in UNIVERSE if s not in fixed]
        gap = 1.0 - sum(w[s] for s in fixed)
        if not free: break
        total = sum(w[s] for s in free)
        for s in free: w[s] = gap * w[s] / max(total, 1e-12)
        if all(MIN_W - 1e-10 <= w[s] <= MAX_W + 1e-10 for s in UNIVERSE): break
    # A final normalization only corrects floating-point residue; bounds have ample slack.
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global last_decision
    account = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 35: continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.
        data[s] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(data) < 12: return
    decision = max(v[4] for v in data.values())
    if last_decision is not None:
        try:
            age = (np.datetime64(decision, "D") - np.datetime64(last_decision, "D")) / np.timedelta64(1, "D")
            if age < REBALANCE_DAYS: return
        except Exception: return

    factors = {k: {} for k in FACTOR_WEIGHTS}
    invvol = {}
    ret5 = {}
    for s, (c, h, l, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), .008)
        ret5[s] = float(c[-1] / max(c[-6], 1e-12) - 1.)
        clv1 = float((2*c[-1] - h[-1] - l[-1]) / max(h[-1] - l[-1], 1e-12))
        clv5 = float(np.mean((2*c[-5:] - h[-5:] - l[-5:]) / np.maximum(h[-5:] - l[-5:], 1e-12)))
        reversal5 = -float(np.mean(r[-5:]))
        shock = -float(r[-1]) * min(abs(float(r[-1] / (vol + .002))), 3.)
        # Volatility-conditioned reversal/CLV: emphasize these only in elevated
        # asset volatility, while keeping the screener's direction intact.
        highvol = min(max(vol / .018, .5), 2.0)
        factors["clv"][s] = clv1
        factors["peer"][s] = ret5[s]
        factors["vix_rev"][s] = reversal5 * highvol
        factors["momentum"][s] = float((c[-1] / max(c[-21], 1e-12) - 1.) / (vol + .01))
        factors["vix_clv"][s] = clv5 * highvol
        factors["shock"][s] = shock
        factors["reversal"][s] = reversal5
        invvol[s] = 1. / vol
    med = float(np.median(list(ret5.values())))
    factors["peer"] = {s: v - med for s, v in ret5.items()}
    ranked = {k: ranks(v) for k, v in factors.items()}
    score = {s: sum(FACTOR_WEIGHTS[k] * ranked[k][s] for k in FACTOR_WEIGHTS) for s in UNIVERSE}

    breadth = float(np.mean([data[s][0][-1] > data[s][0][-21] for s in data]))
    spx = data.get("SPX")
    bear = spx is not None and spx[0][-1] < spx[0][-21] and spx[0][-1] < spx[0][-6]
    if bear or breadth < .40:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .14
        for s in ("BTC", "ETH", "WTI"): score[s] -= .06
    elif breadth > .67:
        for s in ("SPX", "NDX", "SOX", "000300.SH"): score[s] += .04

    mean_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(.03, score[s]) * (.80 + .20 * invvol.get(s, mean_iv) / max(mean_iv, 1e-12)) for s in UNIVERSE}
    weights = bounded_simplex(raw)
    if set(weights) == set(UNIVERSE) and all(np.isfinite(v) and v >= 0 for v in weights.values()) and abs(sum(weights.values()) - 1.) < 1e-8:
        rebalance_to_weights(weights)
        last_decision = decision
