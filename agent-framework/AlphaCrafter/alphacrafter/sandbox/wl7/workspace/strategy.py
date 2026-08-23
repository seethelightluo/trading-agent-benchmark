import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ENSEMBLE = Path(__file__).parent / "factors" / "factor_ensemble.json"
last_decision = None


def factors():
    try:
        return json.loads(ENSEMBLE.read_text(encoding="utf-8")).get("selected_factors", [])[:10]
    except Exception:
        return []


def rank(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + .5) / len(good)
    return out


def bounded(raw):
    w = {s: max(float(raw.get(s, 1e-8)), 1e-8) for s in UNIVERSE}
    for _ in range(30):
        z = sum(w.values())
        w = {s: v / z for s, v in w.items()}
        changed = False
        for s in UNIVERSE:
            if w[s] < .02: w[s] = .02; changed = True
            if w[s] > .14: w[s] = .14; changed = True
        if not changed: break
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_decision
    fs = factors()
    if not fs: return
    feat, vol, dates = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=260)
        if df is None or len(df) < 110: return
        df = df.sort_values("date")
        c = np.asarray(df["close"], float)[:-1]
        if len(c) < 100 or np.any(~np.isfinite(c)) or np.any(c <= 0): return
        r = c[1:] / c[:-1] - 1.0
        v10, v20, v30, v40, v60 = [max(np.std(r[-n:]), .008) for n in (10,20,30,40,60)]
        neg = r[-40:][r[-40:] < 0]
        dn = max(np.std(neg) if len(neg) > 1 else v40, .008)
        stress = max(0., v40 / max(np.std(r[-100:]), .008) - 1.)
        # All features use completed bars only; volume confirmation is a mild, capped gate.
        vv = np.asarray(df["volume"], float)[:-1] if "volume" in df else np.ones(len(c))
        vr = np.log1p(np.maximum(vv[-5:], 0)).mean() - np.log1p(np.maximum(vv[-20:], 0)).mean()
        feat[s] = {
            "breadth": .6 * (c[-1]/c[-21]-1)/v20 + .4 * (c[-1]/c[-6]-1)/v10,
            "range_reversal": -(c[-1]/c[-2]-1)/v10,
            "asym_reversal": -(c[-1]/c[-3]-1)/v10,
            "volume_reversal": -(c[-1]/c[-4]-1)/v10 * (1.0 + .25*np.tanh(vr)),
            "residual": (c[-1]/c[-21]-1)/v20 - .35*(c[-1]/c[-6]-1)/v10,
            "stress_trend": (c[-1]/c[-31]-1)/v30/(1.+stress),
            "risk_momentum": (c[-1]/c[-21]-1)/max(v20, dn),
        }
        vol[s] = max(np.std(r[-25:]), .008)
        dates[s] = str(df.iloc[-1]["date"])
    stamp = max(dates.values())
    if last_decision is not None:
        try:
            if (np.datetime64(stamp)-np.datetime64(last_decision))/np.timedelta64(1,'D') < 10: return
        except Exception: return
    keys = {
      "miner_1_20270827_breadth_defensive_trend":"breadth",
      "miner_3_range_weighted_intraday_reversal_1d":"range_reversal",
      "miner_3_20261130_asymmetric_volatility_reversal_2d_revalidated":"asym_reversal",
      "miner_3_volume_confirmed_reversal_3d":"volume_reversal",
      "miner_2_20270823_residual_momentum_20d":"residual",
      "miner_1_20270830_stress_conditional_trend":"stress_trend",
      "miner_3_20270604_risk_adjusted_20d_momentum":"risk_momentum",
    }
    score = {s:.5 for s in UNIVERSE}
    for f in fs:
        k = keys.get(f.get("factor_id"))
        if k:
            rr = rank({s: float(f.get("direction",1))*feat[s][k] for s in UNIVERSE})
            for s in UNIVERSE: score[s] += float(f.get("weight",0))* (rr[s]-.5)
    spx_bear = feat["SPX"]["breadth"] < 0 and feat["SPX"]["stress_trend"] < 0
    stressed = False
    vx = get_index_daily_data(symbol="VIX", days=70)
    if vx is not None and len(vx)>35:
        x = np.asarray(vx.sort_values("date")["close"], float)[:-1]
        stressed = len(x)>30 and x[-1] > np.mean(x[-30:]) + .5*np.std(x[-30:])
    if spx_bear or stressed:
        for s in ("XAU","US10Y","CN10Y"): score[s] += .08
        for s in ("BTC","ETH","WTI"): score[s] -= .05
    avg_inv = np.mean([1./vol[s] for s in UNIVERSE])
    raw = {s:max(score[s],.05)*(.78+.22*(1./vol[s])/avg_inv) for s in UNIVERSE}
    mu, sd = np.mean(list(score.values())), max(np.std(list(score.values())),1e-8)
    forecast = {s: float(.01*(score[s]-mu)/sd) for s in UNIVERSE}
    rebalance_to_weights(bounded(raw), forecast_returns=forecast,
                         factor_ids=[f["factor_id"] for f in fs], horizon_days=10)
    last_decision = stamp

strategy = cross_asset_strategy
