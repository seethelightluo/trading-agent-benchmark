import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CADENCE = 10
MIN_W, MAX_W = 0.025, 0.12
_last_date, _count = None, CADENCE


def rank_map(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = (i + 1) / len(valid)
    elif valid:
        out[valid[0][0]] = 1.0
    return out


def project(raw):
    # Iterative bounded simplex projection: all 15 assets remain invested.
    w = {s: max(1e-9, float(raw.get(s, 0.05))) for s in UNIVERSE}
    for _ in range(60):
        z = sum(w.values())
        w = {s: v / z for s, v in w.items()}
        nw = {s: min(MAX_W, max(MIN_W, v)) for s, v in w.items()}
        if max(abs(nw[s] - w[s]) for s in UNIVERSE) < 1e-10:
            w = nw
            break
        fixed = [s for s in UNIVERSE if nw[s] == MIN_W or nw[s] == MAX_W]
        free = [s for s in UNIVERSE if s not in fixed]
        w = nw
        rem = 1.0 - sum(w[s] for s in fixed)
        free_sum = sum(w[s] for s in free)
        if free and free_sum > 0:
            for s in free:
                w[s] *= rem / free_sum
    z = sum(w.values())
    return {s: max(0.0, w[s] / z) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _last_date, _count
    get_account_dict()
    data = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=100)
        if df is None or len(df) < 35:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
        ret = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
        data[symbol] = (close, high, low, ret, str(df.iloc[-1]["date"]))
    if len(data) < 12:
        return

    decision_date = max(v[4] for v in data.values())
    if decision_date != _last_date:
        _last_date, _count = decision_date, _count + 1
    if _count < CADENCE:
        return

    aliases = {"miner_3_clv_1d": "clv", "peer_median_leadlag_5d": "peer",
               "miner_2_risk_adjusted_momentum_20d": "mom", "short_term_reversal_5d": "rev"}
    try:
        ensemble = json.loads((Path(__file__).parent / "factors/factor_ensemble.json").read_text())
    except Exception:
        ensemble = {}
    factor_weight = {k: 0.0 for k in aliases.values()}
    factor_ids = []
    for item in ensemble.get("selected_factors", [])[:10]:
        key = aliases.get(str(item.get("factor_id", "")))
        if key:
            direction = 1 if int(item.get("direction", 1)) >= 0 else -1
            factor_weight[key] += direction * max(0.0, float(item.get("weight", 0.0)))
            factor_ids.append(str(item["factor_id"]))
    if sum(abs(x) for x in factor_weight.values()) == 0:
        factor_weight = {"clv": .3395, "peer": .2612, "mom": .1981, "rev": .2012}
        factor_ids = list(aliases)
    norm = sum(abs(x) for x in factor_weight.values())
    factor_weight = {k: v / norm for k, v in factor_weight.items()}

    factors = {k: {} for k in factor_weight}
    invvol = {}
    for symbol, (close, high, low, ret, _) in data.items():
        vol = max(float(np.std(ret[-20:])), 0.008)
        invvol[symbol] = 1.0 / vol
        factors["clv"][symbol] = (2*close[-1] - high[-1] - low[-1]) / max(high[-1] - low[-1], 1e-12)
        factors["peer"][symbol] = close[-1] / max(close[-6], 1e-12) - 1.0
        factors["mom"][symbol] = (close[-1] / max(close[-21], 1e-12) - 1.0) / (vol + 0.01)
        factors["rev"][symbol] = -float(np.mean(ret[-5:]))
    median_peer = np.median(list(factors["peer"].values()))
    factors["peer"] = {s: v - median_peer for s, v in factors["peer"].items()}
    ranks = {k: rank_map(v) for k, v in factors.items()}
    score = {s: sum(factor_weight[k] * ranks[k].get(s, 0.5) for k in factor_weight) for s in UNIVERSE}

    # Current screener regime: bearish, high and unstable volatility.  Keep
    # gross exposure at 100%, expressing defense via tradable gold/yields.
    spx = data.get("SPX")
    bearish = bool(spx and spx[0][-1] < spx[0][-6] and spx[0][-1] < spx[0][-21])
    if bearish:
        for symbol in ("XAU", "US10Y", "CN10Y"):
            score[symbol] += 0.30
        for symbol in ("BTC", "ETH", "WTI", "SOX", "NDX"):
            score[symbol] -= 0.18

    mean_invvol = float(np.mean(list(invvol.values())))
    # 55% rank signal / 45% inverse-vol risk control, then bounded projection.
    raw = {s: max(0.05, score[s]) * (0.55 + 0.45 * invvol.get(s, mean_invvol) / max(mean_invvol, 1e-12)) for s in UNIVERSE}
    target = project(raw)
    score_array = np.array([score[s] for s in UNIVERSE])
    scale = max(float(np.std(score_array)), 1e-8)
    center = float(np.mean(score_array))
    forecast = {s: float(0.006 * (score[s] - center) / scale) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=CADENCE)
    _count = 0
