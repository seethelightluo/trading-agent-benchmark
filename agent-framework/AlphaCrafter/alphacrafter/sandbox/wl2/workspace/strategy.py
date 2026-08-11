import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
WEIGHTS = {"clv": 0.36, "peer": 0.28, "momentum": 0.22, "reversal": 0.14}
MIN_WEIGHT, MAX_WEIGHT = 0.015, 0.16
_day = 0


def rank_cs(x):
    out = {s: 0.5 for s in ASSETS}
    valid = sorted((s, float(v)) for s, v in x.items() if np.isfinite(v))
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1.0) / len(valid)
    return out


def bounded_weights(raw):
    # Water-fill onto the simplex with explicit concentration bounds.
    w = {s: max(float(raw.get(s, 0.0)), 1e-10) for s in ASSETS}
    w = {s: v / sum(w.values()) for s, v in w.items()}
    low, high = set(), set()
    for _ in range(100):
        changed = False
        for s in ASSETS:
            if s in low or s in high:
                continue
            if w[s] < MIN_WEIGHT:
                w[s] = MIN_WEIGHT; low.add(s); changed = True
            elif w[s] > MAX_WEIGHT:
                w[s] = MAX_WEIGHT; high.add(s); changed = True
        free = [s for s in ASSETS if s not in low and s not in high]
        if not changed or not free:
            break
        remain = 1.0 - sum(w[s] for s in low | high)
        base = sum(w[s] for s in free)
        for s in free:
            w[s] = remain * w[s] / base
    # Final normalization preserves the complete, cash-free target.
    total = sum(w.values())
    return {s: float(w[s] / total) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    account = get_account_dict()
    positions = account.get("positions", [])
    # One decision per ten trading-day block; first online allocation is made.
    if any(float(p.get("quantity", 0.0)) > 0 for p in positions) and (_day - 1) % 10 != 0:
        return

    data = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 25:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df.close, dtype=float)
        h = np.asarray(df.high, dtype=float)
        l = np.asarray(df.low, dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        data[s] = (c, h, l, r)
    if len(data) < 12:
        return

    clv, peer0, mom, rev, invvol = {}, {}, {}, {}, {}
    for s, (c, h, l, r) in data.items():
        rng = max(float(h[-1] - l[-1]), 1e-12)
        clv[s] = (2.0 * c[-1] - h[-1] - l[-1]) / rng
        peer0[s] = c[-1] / max(c[-6], 1e-12) - 1.0
        vol = max(float(np.std(r[-20:])), 0.008)
        invvol[s] = 1.0 / vol
        mom[s] = (c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01)
        rev[s] = -float(np.mean(r[-5:]))
    med = float(np.median(list(peer0.values())))
    peer = {s: v - med for s, v in peer0.items()}
    ranks = {k: rank_cs(v) for k, v in {"clv": clv, "peer": peer, "momentum": mom, "reversal": rev}.items()}
    score = {s: sum(WEIGHTS[k] * ranks[k][s] for k in WEIGHTS) for s in ASSETS}

    # Bearish regime: use tradable defensive benchmarks, never cash or shorts.
    if "SPX" in data:
        c = data["SPX"][0]
        if c[-1] < c[-21] and c[-1] < c[-6]:
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += 0.18
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(0.05, score[s] - 0.10)

    avg_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(score[s], 0.01) * (0.78 + 0.22 * invvol.get(s, avg_iv) / avg_iv) for s in ASSETS}
    rebalance_to_weights(bounded_weights(raw))
