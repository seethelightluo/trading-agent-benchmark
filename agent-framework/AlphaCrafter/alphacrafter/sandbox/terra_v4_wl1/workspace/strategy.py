import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CADENCE = 10
MIN_W, MAX_W = 0.02, 0.14
last_date = None
blocks = CADENCE


def percentile(values):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    n = len(good)
    if n > 1:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / n
    return out


def bounded_weights(raw):
    # Iterative capped/floored proportional projection, then exact normalization.
    w = {s: max(0.0, float(raw.get(s, 0.0))) for s in UNIVERSE}
    for _ in range(50):
        free = [s for s in UNIVERSE if MIN_W < w[s] < MAX_W]
        fixed = [s for s in UNIVERSE if s not in free]
        rem = 1.0 - sum(w[s] for s in fixed)
        denom = sum(w[s] for s in free)
        if not free or denom <= 0:
            break
        nw = {s: w[s] for s in fixed}
        for s in free:
            nw[s] = rem * w[s] / denom
        w = nw
        clipped = False
        for s in UNIVERSE:
            if w[s] < MIN_W:
                w[s] = MIN_W; clipped = True
            elif w[s] > MAX_W:
                w[s] = MAX_W; clipped = True
        if not clipped:
            break
    z = sum(w.values()) or 1.0
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_date, blocks
    account = get_account_dict()
    market = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=90)
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        market[symbol] = (c, h, l, r, str(df.iloc[-1]["date"]))
    if len(market) < 12:
        return
    date = max(v[4] for v in market.values())
    if date != last_date:
        blocks += 1
        last_date = date
    if blocks < CADENCE:
        return

    # The screener ensemble is authoritative; only four admitted factors are used.
    fallback = {
        "peer_median_leadlag_5d": ("peer", 0.30),
        "miner_2_risk_adjusted_momentum_20d": ("momentum", 0.27),
        "miner_3_clv_1d": ("clv", 0.28),
        "short_term_reversal_5d": ("reversal", 0.15),
    }
    try:
        ensemble = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        selected = ensemble.get("selected_factors", [])[:10]
    except Exception:
        selected = []
    factor_weights = {name: 0.0 for name in ("peer", "momentum", "clv", "reversal")}
    factor_ids = []
    for item in selected:
        fid = str(item.get("factor_id", ""))
        if fid in fallback:
            name, _ = fallback[fid]
            factor_weights[name] += max(0.0, float(item.get("weight", 0.0))) * (1 if int(item.get("direction", 1)) >= 0 else -1)
            factor_ids.append(fid)
    if sum(abs(v) for v in factor_weights.values()) == 0:
        factor_weights = {name: wt for name, wt in ((v[0], v[1]) for v in fallback.values())}
        factor_ids = list(fallback)
    z = sum(abs(v) for v in factor_weights.values()) or 1.0
    factor_weights = {k: v / z for k, v in factor_weights.items()}

    clv, reversal, momentum, lead = {}, {}, {}, {}
    invvol = {}
    for s, (c, h, l, r, _) in market.items():
        vol = max(float(np.std(r[-20:])), 0.008)
        # Winsorization is implicit through cross-sectional ranks; CLV uses recent completed bars.
        clv[s] = float(np.mean((2*c[-3:] - h[-3:] - l[-3:]) / np.maximum(h[-3:] - l[-3:], 1e-12)))
        reversal[s] = -float(np.mean(r[-5:]))
        momentum[s] = float((c[-1] / max(c[-21], 1e-12) - 1.0) / (vol + 0.01))
        lead[s] = float(c[-1] / max(c[-6], 1e-12) - 1.0)
        invvol[s] = 1.0 / vol
    median_lead = float(np.median(list(lead.values())))
    lead = {s: v - median_lead for s, v in lead.items()}
    ranked = {"clv": percentile(clv), "reversal": percentile(reversal),
              "momentum": percentile(momentum), "peer": percentile(lead)}
    score = {s: sum(factor_weights[k] * ranked[k].get(s, 0.5) for k in ranked) for s in UNIVERSE}

    # Bullish-to-sideways: retain broad exposure. Confirmed two-horizon deterioration
    # shifts risk to tradable defensive assets, never to cash.
    spx = market.get("SPX")
    defensive = ("XAU", "US10Y", "CN10Y")
    if spx is not None and spx[0][-1] < spx[0][-21] and spx[0][-1] < spx[0][-6]:
        for s in defensive:
            score[s] += 0.20
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= 0.12
    # Elevated volatility gets a modest extra inverse-vol tilt, not a gross reduction.
    avg_vol = np.mean([1.0 / invvol[s] for s in market]) or 1.0
    raw = {s: max(0.05, score[s]) * (0.70 + 0.30 * invvol.get(s, 1.0 / avg_vol) * avg_vol) for s in UNIVERSE}
    target = bounded_weights(raw)
    vals = np.array(list(score.values()), dtype=float)
    sd = max(float(vals.std()), 1e-8)
    forecast = {s: float(0.008 * (score[s] - vals.mean()) / sd) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=10)
    blocks = 0
