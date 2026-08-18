"""Trader: replicate strategy.py score computation with current data to inspect dispersion."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict


def closes(assets):
    out = {}
    for a in assets:
        df = None
        try:
            df = get_stock_daily_data(a, days=200)
        except Exception:
            df = None
        if df is None or len(df) < 140:
            try:
                df = get_index_daily_data(a, days=200)
            except Exception:
                df = None
        if df is not None and len(df) >= 140 and "close" in df:
            s = df[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"])
            out[a] = s.set_index("date")["close"].astype(float)
    return out


def rank_map(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and math.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out


def trend_r2(cx):
    s = cx.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return math.copysign(cov * cov / (vy * vx), cov)


def semi_down(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0


def mom120(cx):
    if len(cx) < 126:
        return None
    p0 = float(cx.iloc[-126])
    return float(cx.iloc[-6]) / p0 - 1.0 if p0 > 0 else None


def mom10(cx):
    if len(cx) < 17:
        return None
    p0 = float(cx.iloc[-16])
    return float(cx.iloc[-6]) / p0 - 1.0 if p0 > 0 else None


def vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not math.isfinite(out) else float(out)


def underwater(cx):
    s = cx.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    return q95 / abs(q05) if abs(q05) > 1e-12 else None


def dxy_beta(r, dr):
    if dr is None:
        return None
    z = pd.concat([r.rename("a"), dr.rename("d")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vd = float(z["d"].var())
    return float(z["a"].cov(z["d"]) / vd) if vd > 1e-14 else None


def vix_beta(r, vr, vc):
    if vr is None or vc is None:
        return None
    z = pd.concat([r.rename("a"), vr.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vv = float(z["v"].var())
    if vv < 1e-14:
        return None
    beta = float(z["a"].cov(z["v"]) / vv)
    if len(vc) < 22:
        return None
    v0 = float(vc.iloc[-21])
    return -beta * (float(vc.iloc[-1]) / v0 - 1.0) if v0 > 0 else None


def main():
    assets = list(get_account_dict()["watch_list"])
    c = closes(assets)
    print("assets with closes:", len(c), "of", len(assets))
    panel = pd.DataFrame(c).sort_index()
    print("panel rows:", len(panel), "last:", panel.index[-1].date())
    rets = panel.pct_change()

    dxy_df = get_index_daily_data("DXY", days=150)
    vix_df = get_index_daily_data("VIX", days=150)
    dxy_r = dxy_df.pct_change() if dxy_df is not None else None
    vix_r = vix_df.pct_change() if vix_df is not None else None

    ens = json.loads(Path("factors/factor_ensemble.json").read_text())
    ens = [(it["factor_id"], float(it["weight"]), int(it["direction"])) for it in ens["selected_factors"]]

    fvals = {fid: {} for fid, _, _ in ens}
    for a in assets:
        cx = c.get(a)
        r = rets[a] if a in rets else None
        if cx is None or r is None:
            continue
        for fid, _, _ in ens:
            try:
                if fid == "trend_r2_30_signed":
                    v = trend_r2(cx)
                elif fid == "semi_down_ratio_20":
                    v = semi_down(r)
                elif fid == "mom_120d_skip5":
                    v = mom120(cx)
                elif fid == "mom_10d_skip5":
                    v = mom10(cx)
                elif fid == "vol_of_vol20x60":
                    v = vol_of_vol(r)
                elif fid == "time_under_water_120":
                    v = underwater(cx)
                elif fid == "tail_ratio_20":
                    v = tail_ratio(r)
                elif fid == "dxy_beta_60":
                    v = dxy_beta(r, dxy_r)
                elif fid == "vix_beta_cond_60x20":
                    v = vix_beta(r, vix_r, vix_df)
                else:
                    v = None
            except Exception:
                v = None
            fvals[fid][a] = v

    print("\nper-factor valid counts:")
    for fid, _, _ in ens:
        nv = sum(1 for a in assets if fvals[fid].get(a) is not None)
        print(f"  {fid}: {nv}/15 valid")

    score = {a: 0.0 for a in assets}
    for fid, w, d in ens:
        rk = rank_map(fvals[fid], assets)
        for a in assets:
            score[a] += w * d * rk[a]

    sv = np.array([score[a] for a in assets])
    sd = float(sv.std())
    print(f"\nscore sd: {sd:.6f} (equal-weight fallback if < 1e-12)")
    for a in sorted(score, key=lambda x: -score[x]):
        print(f"  {a:>10}: score={score[a]:+.4f}")


if __name__ == "__main__":
    main()
