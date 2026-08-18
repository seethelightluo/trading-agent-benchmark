"""Trader verification for block 2031-04-03 -> 2031-04-17.
Replicates strategy.py scoring (pure functions) to compute the v22 proposal
target and estimate turnover vs the drifted holdings. Does NOT call
rebalance_to_weights (no side effects on live account)."""
import json
from pathlib import Path
from math import isfinite, copysign
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict, get_stock_daily_data, get_index_daily_data,
)

FETCH = 200
MAX_W, MIN_W = 0.18, 0.005
DEF = {"XAU", "US10Y", "CN10Y"}


def _closes(assets):
    out = {}
    for a in assets:
        df = None
        try:
            df = get_stock_daily_data(a, days=FETCH)
        except Exception:
            df = None
        if df is None or len(df) < 140:
            try:
                df = get_index_daily_data(a, days=FETCH)
            except Exception:
                df = None
        if df is not None and len(df) >= 140 and "close" in df:
            s = df[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"])
            out[a] = s.set_index("date")["close"].astype(float)
    return out


def _macro_close(symbol):
    df = None
    try:
        df = get_index_daily_data(symbol, days=150)
    except Exception:
        df = None
    if df is None or "close" not in df or len(df) < 80:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)


def _rank_map(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out


def _trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return copysign(cov * cov / (vy * vx), cov)


def _semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0


def _mom_120(c):
    if len(c) < 126:
        return None
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _mom_10(c):
    if len(c) < 17:
        return None
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def _vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not isfinite(out) else float(out)


def _tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return None
    return q95 / abs(q05)


def _beta_60(r, m_r):
    z = pd.concat([r.rename("a"), m_r.rename("m")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vm = float(z["m"].var())
    if vm < 1e-14:
        return None
    return float(z["a"].cov(z["m"]) / vm)


def _vix_beta_cond(r, vix_r, vix_c):
    z = pd.concat([r.rename("a"), vix_r.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vv = float(z["v"].var())
    if vv < 1e-14:
        return None
    beta = float(z["a"].cov(z["v"]) / vv)
    if vix_c is None or len(vix_c) < 22:
        return None
    v0 = float(vix_c.iloc[-21])
    if v0 <= 0:
        return None
    vmove = float(vix_c.iloc[-1]) / v0 - 1.0
    return -beta * vmove


def _to_weights(score, assets, regime_w):
    vals = np.array([score[a] for a in assets], dtype=float)
    mu, sd = float(vals.mean()), float(vals.std())
    if sd < 1e-12:
        return {a: 1.0 / len(assets) for a in assets}
    z = (vals - mu) / sd
    w = np.exp(z / 0.85)
    raw = {a: float(w[i]) for i, a in enumerate(assets)}
    total = sum(raw.values())
    if total <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    wts = {a: raw[a] / total for a in assets}
    for a in assets:
        wts[a] = 0.70 * wts[a] + 0.30 * regime_w.get(a, 1.0 / len(assets))
    for _ in range(60):
        excess = sum(max(0.0, x - MAX_W) for x in wts.values())
        wts = {a: min(MAX_W, max(MIN_W, x)) for a, x in wts.items()}
        room = [a for a in wts if wts[a] < MAX_W - 1e-12]
        if excess < 1e-12 or not room:
            break
        den = sum(max(0.0, regime_w.get(a, 0.0)) for a in room)
        for a in room:
            wts[a] += excess * (regime_w.get(a, 0.0) / den if den else 1.0 / len(room))
    s = sum(wts.values())
    wts = {a: x / s for a, x in wts.items()}
    wts[assets[-1]] += 1.0 - sum(wts.values())
    return wts


def main():
    acct = get_account_dict()
    assets = list(acct["watch_list"])
    pos = {p["symbol"]: p for p in acct["positions"]}
    total = acct["total_assets"]

    ens = json.loads((Path("factors/factor_ensemble.json")).read_text())
    ensemble = [(it["factor_id"], float(it["weight"]), int(it["direction"]))
                for it in ens["selected_factors"]][:10]
    print("ensemble v:", ens.get("as_of"), "| n factors:", len(ensemble),
          "| sum w:", round(sum(w for _, w, _ in ensemble), 4))

    closes = _closes(assets)
    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change()

    dxy_c, dxy_r = _macro_close("DXY"), None
    if dxy_c is not None:
        dxy_r = dxy_c.pct_change()
    vix_c, vix_r = _macro_close("VIX"), None
    if vix_c is not None:
        vix_r = vix_c.pct_change()
    cny_c, cny_r = _macro_close("USDCNY"), None
    if cny_c is not None:
        cny_r = cny_c.pct_change()

    fvals = {fid: {} for fid, _, _ in ensemble}
    for a in assets:
        c, r = closes.get(a), (rets[a] if a in rets else None)
        if c is None or r is None:
            continue
        for fid, _, _ in ensemble:
            try:
                if fid == "trend_r2_30_signed":
                    v = _trend_r2(c)
                elif fid == "semi_down_ratio_20":
                    v = _semi_down_ratio(r)
                elif fid == "mom_120d_skip5":
                    v = _mom_120(c)
                elif fid == "mom_10d_skip5":
                    v = _mom_10(c)
                elif fid == "vol_of_vol20x60":
                    v = _vol_of_vol(r)
                elif fid == "time_under_water_120":
                    v = _underwater(c)
                elif fid == "tail_ratio_20":
                    v = _tail_ratio(r)
                elif fid == "dxy_beta_60":
                    v = _beta_60(r, dxy_r) if dxy_r is not None else None
                elif fid == "cny_beta_60":
                    v = _beta_60(r, cny_r) if cny_r is not None else None
                elif fid == "vix_beta_cond_60x20":
                    v = _vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None
                else:
                    v = None
            except Exception:
                v = None
            fvals[fid][a] = v

    score = {a: 0.0 for a in assets}
    for fid, w, direction in ensemble:
        rk = _rank_map(fvals[fid], assets)
        for a in assets:
            score[a] += w * direction * rk[a]

    market = rets.mean(axis=1)
    trend20 = float(market.tail(20).mean())
    avg_px = float(panel.mean(axis=1).iloc[-1])
    ma60 = float(panel.mean(axis=1).tail(60).mean())
    bearish = trend20 < 0.0 and avg_px < ma60
    print(f"trend20={trend20:.4%} avg_px={avg_px:.2f} ma60={ma60:.2f} bearish={bearish}")
    regime_w = {}
    for a in assets:
        if bearish and a in DEF:
            regime_w[a] = 2.2
        elif bearish:
            regime_w[a] = 0.75
        else:
            regime_w[a] = 1.0

    wts = _to_weights(score, assets, regime_w)
    print("\nproposal weights vs drifted holdings:")
    cur = {a: pos[a]["quantity"] * pos[a]["current_price"] / total for a in assets}
    turn = 0.0
    for a in assets:
        d = wts[a] - cur[a]
        turn += abs(d)
        print(f"  {a:8s} prop={wts[a]*100:6.2f}% cur={cur[a]*100:6.2f}% diff={d*100:+6.2f}pp")
    print(f"\none-way turnover (sum|diff|/2): {turn/2:.4f}  | 3bp cost on migrated notional: {turn/2*0.0003*100:.4f}% of NAV")
    print("weights sum:", round(sum(wts.values()), 6))

    # forecast-return scale sanity (same as strategy)
    sv = np.array([score[a] for a in assets], dtype=float)
    smu, ssd = float(sv.mean()), float(sv.std())
    rs = float(panel.tail(200).pct_change().std(axis=1, ddof=0).median()) if len(panel) >= 40 else 0.01
    print(f"\nreturn_scale={rs:.4f} score sd={ssd:.4f}")


if __name__ == "__main__":
    main()
