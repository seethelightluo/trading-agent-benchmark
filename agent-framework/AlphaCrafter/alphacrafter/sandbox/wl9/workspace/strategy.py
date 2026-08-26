import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_index_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ENSEMBLE = Path(__file__).parent / "factors" / "factor_ensemble.json"
_call = 0


def ensemble():
    try:
        fs = json.loads(ENSEMBLE.read_text()).get("selected_factors", [])[:10]
        if fs and abs(sum(float(x["weight"]) for x in fs) - 1.0) < 1e-6:
            return fs
    except Exception:
        pass
    return []


def ranks(x):
    ok = [a for a in ASSETS if np.isfinite(x.get(a, np.nan))]
    ok.sort(key=lambda a: x[a])
    out = {a: 0.5 for a in ASSETS}
    for i, a in enumerate(ok):
        out[a] = (i + 1.0) / len(ok)
    return out


def macro_ret(symbol):
    try:
        d = get_index_daily_data(symbol=symbol, days=100)
        if d is not None and len(d) > 61:
            c = np.asarray(d.sort_values("date")["close"], float)[:-1]
            return float(c[-1] / max(c[-61], 1e-12) - 1.0)
    except Exception:
        pass
    return 0.0


@register_hook
def strategy():
    global _call
    if _call % 10:
        _call += 1
        return
    _call += 1
    fs = ensemble()
    if not fs:
        return

    close = {}
    for a in ASSETS:
        d = get_stock_daily_data(symbol=a, days=450)
        if d is None or len(d) < 180:
            return
        # Exclude the current incomplete bar: decisions use completed data only.
        close[a] = np.asarray(d.sort_values("date")["close"], float)[:-1]
    n = min(map(len, close.values()))
    close = {a: x[-n:] for a, x in close.items()}
    ret = {a: np.diff(close[a]) / np.maximum(close[a][:-1], 1e-12) for a in ASSETS}
    r10 = {a: close[a][-1] / max(close[a][-11], 1e-12) - 1.0 for a in ASSETS}
    r60 = {a: close[a][-1] / max(close[a][-61], 1e-12) - 1.0 for a in ASSETS}
    r120 = {a: close[a][-1] / max(close[a][-121], 1e-12) - 1.0 for a in ASSETS}
    v20 = {a: max(float(np.std(ret[a][-20:], ddof=1)), 0.004) for a in ASSETS}
    v60 = {a: max(float(np.std(ret[a][-60:], ddof=1)), 0.004) for a in ASSETS}
    medv = max(float(np.median(list(v20.values()))), 0.004)
    market60 = float(np.mean(list(r60.values())))
    breadth = float(np.mean([r10[a] > 0 for a in ASSETS]))
    macro = macro_ret("VIX") - macro_ret("DXY")

    raw = {a: {} for a in ASSETS}
    for a in ASSETS:
        path = float(np.sum(np.abs(ret[a][-60:]))) + 1e-12
        dd = close[a][-1] / max(float(np.max(close[a][-121:])), 1e-12) - 1.0
        raw[a]["miner_2_20301212_efficiency_reversal_60d"] = -(r60[a] / path) / v60[a]
        raw[a]["miner_2_20301114_compressed_trend_reversal_60d"] = -r60[a] / v60[a] * np.clip(v60[a] / v20[a], .5, 2.)
        raw[a]["miner_3_20301114_breadth_gated_contrarian_60d"] = -r10[a] * (1. + .5 * (1. - breadth))
        raw[a]["miner_1_20301031_riskadjusted_momentum_60d"] = -r120[a] / v60[a]
        raw[a]["miner_1_20320624_peak_drawdown_recovery_60d"] = -dd / (v60[a] * np.sqrt(60.)) * (.5 + max(float(np.sum(ret[a][-10:])), 0.))
        raw[a]["miner_1_20330203_macro_residual_reversal_vixdxy_60d"] = -(r60[a] - market60) / v60[a] * (1. + np.tanh(abs(macro)))
        raw[a]["miner_3_20300822_volatility_dispersion_60d"] = -(v20[a] / medv - 1.)

    score = {a: 0. for a in ASSETS}
    for f in fs:
        rr = ranks({a: raw[a].get(f["factor_id"], np.nan) for a in ASSETS})
        for a in ASSETS:
            score[a] += float(f["weight"]) * int(f.get("direction", 1)) * rr[a]

    # High-risk, choppy regime: remain fully invested but tilt to tradable defensives.
    if market60 < 0. or breadth < .5:
        for a in ("XAU", "US10Y", "CN10Y"):
            score[a] += .18
        for a in ("BTC", "ETH", "WTI"):
            score[a] -= .10

    z = np.clip((np.array(list(score.values())) - np.mean(list(score.values()))) /
                max(float(np.std(list(score.values()))), 1e-9), -1.15, 1.15)
    w = np.maximum(np.exp(.025 * z), .05)
    w /= float(w.sum())
    target = {a: float(w[i]) for i, a in enumerate(ASSETS)}
    forecast = {a: float(.005 * z[i]) for i, a in enumerate(ASSETS)}
    rebalance_to_weights(target, forecast_returns=forecast,
                         factor_ids=[str(f["factor_id"]) for f in fs], horizon_days=10)
