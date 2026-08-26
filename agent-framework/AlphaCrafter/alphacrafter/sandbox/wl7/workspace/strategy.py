import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_index_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
ENSEMBLE = Path(__file__).parent / "factor_ensemble.json"
last_stamp = None


def _rank(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 0.5) / max(len(good), 1)
    return out


def _weights(score):
    x = np.array([score[s] for s in UNIVERSE], dtype=float)
    raw = np.exp(np.clip(2.0 * (x - x.mean()), -2.5, 2.5))
    w = 0.02 + 0.70 * raw / raw.sum()
    for _ in range(20):
        high = w > 0.16
        if not high.any(): break
        excess = float((w[high] - 0.16).sum())
        w[high] = 0.16
        room = ~high
        w[room] += excess * raw[room] / max(float(raw[room].sum()), 1e-12)
    w = np.maximum(w, 0.02)
    w /= w.sum()
    return {s: float(v) for s, v in zip(UNIVERSE, w)}


@register_hook
def cross_asset_strategy():
    global last_stamp
    try:
        fs = json.loads(ENSEMBLE.read_text(encoding="utf-8")).get("selected_factors", [])[:10]
    except Exception:
        return
    if not fs or abs(sum(float(f["weight"]) for f in fs) - 1.0) > 1e-6:
        return
    ids = [str(f["factor_id"]) for f in fs]
    feat, stamp = {}, None
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None or len(df) < 100: return
        df = df.sort_values("date")
        stamp = str(df.iloc[-2]["date"])
        p = np.asarray(df["close"], dtype=float)[:-1]
        if len(p) < 80 or np.any(~np.isfinite(p)) or np.any(p <= 0): return
        r = p[1:] / p[:-1] - 1.0
        v20, v60 = max(np.std(r[-20:]), .008), max(np.std(r[-60:]), .008)
        down = max(np.sqrt(np.mean(np.minimum(r[-20:], 0.0) ** 2)), .008)
        mom20 = p[-1] / p[-21] - 1.0
        feat[s] = {"trend": .70*mom20/v20 + .30*np.mean(r[-20:] > 0),
                   "consistency": .50*mom20/v20 + .50*(p[-1]/p[-61]-1)/v60 + .25*np.mean(r[-20:] > 0),
                   "relative": mom20/v20, "sharpe": np.mean(r[-20:])/down,
                   "anti": -mom20/v20, "downpath": -np.sum(np.minimum(r[-20:],0))/down,
                   "resid": -mom20/v20, "vol": v20}
    if last_stamp is not None:
        try:
            if (np.datetime64(stamp)-np.datetime64(last_stamp))/np.timedelta64(1,"D") < 14: return
        except Exception: return
    stress = np.mean([v["vol"] for v in feat.values()]) > .018
    vx = get_index_daily_data(symbol="VIX", days=80)
    if vx is not None and len(vx) > 62:
        a = np.asarray(vx.sort_values("date")["close"], dtype=float)[:-1]
        stress = stress or (np.isfinite(a[-1]) and a[-1] >= np.median(a[-60:]))
    score = {s: .5 for s in UNIVERSE}
    keys = {
      "miner_1_20270827_breadth_defensive_trend": ("trend", .85 if stress else 1.0),
      "miner_2_20281009_trend_consistency_volscaled": ("consistency", 1.0),
      "miner_1_20280717_breadth_gated_relative_trend20_vol40": ("relative", .80 if stress else 1.0),
      "miner_2_20281023_downside_sharpe20": ("sharpe", 1.0),
      "miner_3_20341225_antipersistence20": ("anti", .55 if stress else 1.0),
      "miner_3_20350205_downsidepath_reversal20": ("downpath", .55 if stress else 1.0),
      "miner_3_20341030_residual_volscaled_reversal15": ("resid", .35 if stress else .60)}
    for f in fs:
        fid, wt, direction = str(f["factor_id"]), float(f["weight"]), float(f.get("direction",1))
        if fid not in keys: return
        key, gate = keys[fid]
        rr = _rank({s: direction*gate*feat[s][key] for s in UNIVERSE})
        for s in UNIVERSE: score[s] += wt*(rr[s]-.5)
    if stress:
        for s in DEFENSIVE: score[s] += .10
        for s in ("BTC", "ETH", "WTI"): score[s] -= .06
    inv = {s: 1.0/feat[s]["vol"] for s in UNIVERSE}
    avg = np.mean(list(inv.values()))
    score = {s: score[s]*(.88+.12*inv[s]/avg) for s in UNIVERSE}
    z = (np.array([score[s] for s in UNIVERSE])-np.mean(list(score.values()))) / max(np.std(list(score.values())),1e-8)
    forecast = {s: float(.01*z[i]) for i,s in enumerate(UNIVERSE)}
    rebalance_to_weights(_weights(score), forecast_returns=forecast, factor_ids=ids, horizon_days=10)
    last_stamp = stamp

strategy = cross_asset_strategy
if __name__ == "__main__": pass
