import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, add_order

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_WEIGHTS = {"clv": 0.3395, "peer": 0.2612, "reversal": 0.2012, "momentum": 0.1981}
MIN_W, MAX_W, REBALANCE_DAYS = 0.015, 0.16, 10
last_date = None


def ranks(values):
    result = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    if len(good) < 2:
        return result
    for i, (s, _) in enumerate(good):
        result[s] = (i + 1.0) / len(good)
    return result


def bounded_weights(raw):
    # Projection onto the complete long-only box/simplex, preserving full investment.
    w = {s: max(0.0, float(raw.get(s, 0.0))) for s in UNIVERSE}
    total = sum(w.values()) or 1.0
    w = {s: x / total for s, x in w.items()}
    fixed = set()
    for _ in range(30):
        low = [s for s in UNIVERSE if s not in fixed and w[s] < MIN_W]
        high = [s for s in UNIVERSE if s not in fixed and w[s] > MAX_W]
        if not low and not high:
            break
        for s in low:
            w[s] = MIN_W
            fixed.add(s)
        for s in high:
            w[s] = MAX_W
            fixed.add(s)
        rem = 1.0 - sum(w[s] for s in fixed)
        free = [s for s in UNIVERSE if s not in fixed]
        if not free:
            break
        base = sum(max(w[s], 1e-12) for s in free)
        for s in free:
            w[s] = rem * w[s] / base
    # Tiny numerical correction keeps the contract exact without violating bounds materially.
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_date
    account = get_account_dict()
    market = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=85)
        if df is None or len(df) < 25:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        market[s] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(market) < 12:
        return
    decision = max(x[4] for x in market.values())
    if last_date is not None:
        try:
            days = (np.datetime64(decision, "D") - np.datetime64(last_date, "D")) / np.timedelta64(1, "D")
            if days < REBALANCE_DAYS:
                return
        except Exception:
            return

    clv, peer, reversal, momentum, invvol = {}, {}, {}, {}, {}
    five = {}
    for s, (c, h, l, r, _) in market.items():
        if len(c) < 22:
            continue
        daily_clv = (2*c - h - l) / np.maximum(h-l, 1e-12)
        clv[s] = float(np.mean(daily_clv[-3:]))
        reversal[s] = float(-np.mean(r[-5:]))
        vol = max(float(np.std(r[-20:])), 0.008)
        invvol[s] = 1.0 / vol
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01))
        five[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
    med = float(np.median(list(five.values())))
    peer = {s: v-med for s, v in five.items()}
    rr = {k: ranks(v) for k, v in (("clv",clv),("peer",peer),("reversal",reversal),("momentum",momentum))}
    score = {s: sum(FACTOR_WEIGHTS[k] * rr[k][s] for k in rr) for s in UNIVERSE}

    # Medium-risk, sideways/mildly bearish regime: defensive tradable tilt, never cash.
    if "SPX" in market:
        c = market["SPX"][0]
        bearish = c[-1] < c[-21] and c[-1] < c[-6]
        if bearish:
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += 0.18
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(0.05, score[s] - 0.10)
    mean_iv = np.mean(list(invvol.values())) if invvol else 1.0
    raw = {s: max(0.01, score[s]) * (0.78 + 0.22 * invvol.get(s, mean_iv) / mean_iv) for s in UNIVERSE}
    weights = bounded_weights(raw)

    total = float(account.get("total_assets", account.get("net_assets", 0.0)) or 0.0)
    held = {p.get("symbol"): p for p in account.get("positions", []) if float(p.get("quantity", 0) or 0) > 0}
    # Sell only reductions of existing longs; then buy all underweights.
    for s, p in held.items():
        if s not in weights:
            continue
        px = float(p.get("current_price", 0) or 0)
        current = float(p.get("market_value", 0) or 0)
        gap = current - total * weights[s]
        if px > 0 and gap > total * 0.002:
            qty = min(float(p.get("quantity", 0)), gap / px)
            if qty > 0:
                add_order(symbol=s, order_type="SELL", price=px, quantity=qty)
    for s in UNIVERSE:
        if s not in market:
            continue
        px = float(market[s][0][-1])
        current = float(held.get(s, {}).get("market_value", 0) or 0)
        gap = total * weights[s] - current
        if px > 0 and gap > total * 0.002:
            qty = gap / px
            if qty > 0:
                add_order(symbol=s, order_type="BUY", price=px, quantity=qty)
    last_date = decision
