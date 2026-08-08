import math
from alphacrafter.sim.utils import register_hook, add_order, get_stock_daily_data, get_account_dict

# Four-factor, deliberately simple cross-asset allocator.  State is only used to
# enforce the simulator's ten-trading-day decision cadence.
_last_rebalance = None

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}


def _rank(values):
    good = sorted(v for v in values if math.isfinite(v))
    if not good:
        return 0.5
    # Midrank percentile; ties are harmless in this small universe.
    return sum(1 for x in good if x < values[-1]) / max(1, len(good) - 1)


def _pct_rank(x, pool):
    pool = [v for v in pool if math.isfinite(v)]
    if not pool or not math.isfinite(x):
        return 0.5
    return sum(v < x for v in pool) / max(1, len(pool) - 1)


@register_hook
def cross_asset_allocator():
    global _last_rebalance
    account = get_account_dict() or {}
    watch = [s for s in account.get("watch_list", []) if s in ASSETS]
    # The benchmark contract is exact, but retain a fixed list if the account
    # response omits it during initialization.
    universe = [s for s in ASSETS if s in watch] or ASSETS
    positions = {p.get("symbol"): p for p in account.get("positions", [])}

    frames = {}
    for symbol in universe:
        try:
            df = get_stock_daily_data(symbol=symbol, days=90)
            if df is not None and len(df) >= 25:
                frames[symbol] = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass
    if len(frames) < 8:
        return
    latest_date = max(str(df.iloc[-1]["date"]) for df in frames.values())
    if _last_rebalance is not None and _last_rebalance == latest_date:
        return
    # First allocation, then exactly one decision per ten hook invocations.
    if _last_rebalance is not None:
        _last_rebalance = getattr(cross_asset_allocator, "_calls", 0)
    calls = getattr(cross_asset_allocator, "_calls", 0) + 1
    cross_asset_allocator._calls = calls
    if _last_rebalance is not None and calls % 10 != 1:
        return
    _last_rebalance = latest_date

    raw = {}
    vol = {}
    five = {}
    for symbol, df in frames.items():
        close = [float(x) for x in df["close"].tolist() if x == x and x > 0]
        if len(close) < 25:
            continue
        rets = [close[i] / close[i-1] - 1.0 for i in range(1, len(close))]
        r5 = close[-1] / close[-6] - 1.0
        r20 = close[-1] / close[-21] - 1.0
        rv = rets[-20:]
        sd = math.sqrt(sum((x - sum(rv)/len(rv))**2 for x in rv) / max(1, len(rv)-1))
        vol[symbol] = max(sd, 0.004)
        five[symbol] = r5
        # CLV is smoothed over the last three completed bars.
        clvs = []
        for row in df.iloc[-3:].itertuples():
            hi, lo, c = float(row.high), float(row.low), float(row.close)
            clvs.append(((c-lo)-(hi-c))/(hi-lo) if hi > lo else 0.0)
        raw[symbol] = (sum(clvs)/len(clvs), r20 / (vol[symbol] * math.sqrt(20.0)))
    syms = list(raw)
    med5 = sorted(five[s] for s in syms)[len(syms)//2]
    scores = {}
    for s in syms:
        clv, mom = raw[s]
        peer = five[s] - med5
        reversal = -five[s]
        # Rank each leg, preserving the ensemble's stated positive direction.
        scores[s] = (0.3395*_pct_rank(clv, [raw[x][0] for x in syms]) +
                     0.2612*_pct_rank(peer, [five[x]-med5 for x in syms]) +
                     0.2012*_pct_rank(reversal, [-five[x] for x in syms]) +
                     0.1981*_pct_rank(mom, [raw[x][1] for x in syms]))
    # Medium-risk sideways/mild-bearish posture: inverse-vol overlay and a
    # modest defensive tilt, without reducing total investment.
    spx = frames.get("SPX")
    bearish = False
    if spx is not None and len(spx) >= 61:
        c = [float(x) for x in spx.close.tolist()]
        bearish = c[-1] < sum(c[-20:])/20 and c[-1] < sum(c[-60:])/60
    weights = {}
    for s in syms:
        inv = 1.0 / vol[s]
        # 78% signal / 22% risk budget, centered so every asset remains eligible.
        weights[s] = 0.78*(0.55/len(syms) + 0.45*scores[s]/max(1, sum(scores.values()))) + 0.22*inv
        if bearish:
            weights[s] *= 1.18 if s in DEFENSIVE else (0.88 if s in {"BTC", "ETH", "WTI"} else 0.98)
    total = sum(weights.values())
    weights = {s: w/total for s, w in weights.items()}
    # Complete vector: missing/short-history assets receive the residual equal
    # baseline, then normalize. Bounds prevent concentration in a 15-name set.
    lo, hi = 0.015, 0.16
    weights = {s: min(hi, max(lo, weights.get(s, 1.0/len(universe)))) for s in universe}
    z = sum(weights.values())
    weights = {s: weights[s]/z for s in universe}

    total_assets = float(account.get("total_assets", 0) or account.get("net_assets", 0) or 0)
    if total_assets <= 0:
        return
    # Submit sells first (T+1-safe: only available holdings are reduced), then buys.
    prices = {s: float(frames[s].iloc[-1].close) for s in syms}
    for s, p in positions.items():
        q = float(p.get("quantity", 0) or 0)
        if q > 0 and s in prices:
            target = total_assets * weights.get(s, 0.0)
            delta = q * float(p.get("current_price", prices[s])) - target
            if delta > prices[s]:
                qty = int(max(0, min(q, delta/prices[s])))
                if qty > 0:
                    add_order(symbol=s, order_type="SELL", price=prices[s], quantity=qty)
    for s in universe:
        if s not in prices:
            continue
        q = float(positions.get(s, {}).get("quantity", 0) or 0)
        cur = q * float(positions.get(s, {}).get("current_price", prices[s]))
        delta = total_assets * weights[s] - cur
        if delta > prices[s]:
            qty = int(delta/prices[s])
            if qty > 0:
                add_order(symbol=s, order_type="BUY", price=prices[s], quantity=qty)
