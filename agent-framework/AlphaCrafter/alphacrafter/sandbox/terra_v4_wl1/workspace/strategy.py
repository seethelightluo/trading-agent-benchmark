import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CADENCE = 10
MIN_W, MAX_W = 0.025, 0.12
_last_date, _count = None, CADENCE


def rank_map(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    if len(good) > 1:
        for i, (s, _) in enumerate(good): out[s] = (i + 1) / len(good)
    elif good: out[good[0][0]] = 1.0
    return out


def project(raw):
    # Simple bounded projection, retaining all 15 assets and zero cash.
    w = {s: max(1e-9, float(raw.get(s, 0))) for s in UNIVERSE}
    for _ in range(50):
        z = sum(w.values()); w = {s: v / z for s, v in w.items()}
        nw = {s: min(MAX_W, max(MIN_W, v)) for s, v in w.items()}
        if max(abs(nw[s] - w[s]) for s in UNIVERSE) < 1e-9: w = nw; break
        fixed = [s for s in UNIVERSE if nw[s] in (MIN_W, MAX_W)]
        free = [s for s in UNIVERSE if s not in fixed]
        w = nw
        rem = 1 - sum(w[s] for s in fixed)
        fs = sum(w[s] for s in free)
        if free and fs > 0:
            for s in free: w[s] *= rem / fs
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _last_date, _count
    get_account_dict()  # account access also keeps the hook compatible with online mode
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=100)
        if df is None or len(df) < 35: continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df.close, dtype=float)
        h = np.asarray(df.high, dtype=float); l = np.asarray(df.low, dtype=float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1
        data[s] = (c, h, l, r, str(df.iloc[-1].date))
    if len(data) < 12: return
    date = max(x[4] for x in data.values())
    if date != _last_date:
        _last_date, _count = date, _count + 1
    if _count < CADENCE: return

    aliases = {"miner_3_clv_1d":"clv", "peer_median_leadlag_5d":"peer",
               "miner_2_risk_adjusted_momentum_20d":"mom", "short_term_reversal_5d":"rev"}
    try:
        ens = json.loads((Path(__file__).parent / "factors/factor_ensemble.json").read_text())
    except Exception: ens = {}
    fw = {v: 0.0 for v in aliases.values()}; factor_ids = []
    for item in ens.get("selected_factors", [])[:10]:
        k = aliases.get(str(item.get("factor_id", "")))
        if k:
            fw[k] += max(0., float(item.get("weight", 0.))) * (1 if int(item.get("direction", 1)) >= 0 else -1)
            factor_ids.append(str(item.get("factor_id")))
    if sum(abs(v) for v in fw.values()) == 0:
        fw = {"clv": .378, "peer": .291, "mom": .221, "rev": .110}; factor_ids = list(aliases)
    z = sum(abs(v) for v in fw.values()); fw = {k: v / z for k, v in fw.items()}

    factors = {k: {} for k in fw}; invvol = {}
    for s, (c, h, l, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), .008); invvol[s] = 1 / vol
        factors["clv"][s] = (2*c[-1] - h[-1] - l[-1]) / max(h[-1]-l[-1], 1e-12)
        factors["peer"][s] = c[-1] / max(c[-6], 1e-12) - 1
        factors["mom"][s] = (c[-1] / max(c[-21], 1e-12) - 1) / (vol + .01)
        factors["rev"][s] = -float(np.mean(r[-5:]))
    factors["peer"] = {s: v - np.median(list(factors["peer"].values())) for s, v in factors["peer"].items()}
    ranks = {k: rank_map(v) for k, v in factors.items()}
    score = {s: sum(fw[k] * ranks[k].get(s, .5) for k in fw) for s in UNIVERSE}

    # Current screener regime is sharp bearish/high volatility: full investment is
    # retained, but risk is expressed through tradable metals and bonds.
    spx = data.get("SPX")
    bearish = bool(spx and spx[0][-1] < spx[0][-6] and spx[0][-1] < spx[0][-21])
    if bearish:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .30
        for s in ("BTC", "ETH", "WTI", "SOX", "NDX"): score[s] -= .18
    mean_iv = np.mean(list(invvol.values()))
    raw = {s: max(.05, score[s]) * (.55 + .45 * invvol.get(s, mean_iv) / max(mean_iv, 1e-12)) for s in UNIVERSE}
    target = project(raw)
    a = np.array([score[s] for s in UNIVERSE]); scale = max(float(np.std(a)), 1e-8); center = float(np.mean(a))
    forecast = {s: float(.006 * (score[s] - center) / scale) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=factor_ids, horizon_days=CADENCE)
    _count = 0
