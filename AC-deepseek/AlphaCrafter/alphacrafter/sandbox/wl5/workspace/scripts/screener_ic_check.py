"""Screener live-IC check through visible_through date (2030-10-02).
Computes factor values (mirroring strategy.py) on the 15-asset universe and
cross-sectional rank IC vs 10d forward returns over trailing windows.
Read-only: no account/date writes.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

VT = pd.Timestamp(json.load(open("../persistent/date.json"))["visible_through"])
FROZEN = {"000300.SH", "000688.SH", "HSI", "US10Y", "CN10Y"}
ACTIVE = [c for c in [
    "000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
    "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"] if c not in FROZEN]


def load_close(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VT].set_index("date").sort_index()
    return df["close"].astype(float)


def load_index(sym):
    df = pd.read_csv(f"../persistent/index_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VT].set_index("date").sort_index()
    return df["close"].astype(float)


def trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return np.nan
    y = np.log(s.values.astype(float)); x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1]); vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return np.nan
    return np.copysign(cov * cov / (vy * vx), cov)


def semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return np.nan
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return np.nan
    return down / up - 1.0


def mom_120(c):
    if len(c) < 126:
        return np.nan
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return np.nan
    return float(c.iloc[-6]) / p0 - 1.0


def mom_10(c):
    if len(c) < 17:
        return np.nan
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return np.nan
    return float(c.iloc[-6]) / p0 - 1.0


def underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return np.nan
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return np.nan
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return float(out) if np.isfinite(out) else np.nan


def tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return np.nan
    q95 = float(np.percentile(s.values, 95)); q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return np.nan
    return q95 / abs(q05)


def dxy_beta(r, dxy_r):
    z = pd.concat([r.rename("a"), dxy_r.rename("d")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vd = float(z["d"].var())
    if vd < 1e-14:
        return np.nan
    return float(z["a"].cov(z["d"]) / vd)


def vix_beta_cond(r, vix_r, vix_c):
    z = pd.concat([r.rename("a"), vix_r.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vv = float(z["v"].var())
    if vv < 1e-14:
        return np.nan
    beta = float(z["a"].cov(z["v"]) / vv)
    if vix_c is None or len(vix_c) < 22:
        return np.nan
    v0 = float(vix_c.iloc[-21])
    if v0 <= 0:
        return np.nan
    vmove = float(vix_c.iloc[-1]) / v0 - 1.0
    return -beta * vmove


def kurt_20(r):
    s = r.dropna().tail(40)
    if len(s) < 20:
        return np.nan
    k = s.rolling(20, min_periods=8).kurt().iloc[-1]
    return float(k) if np.isfinite(k) else np.nan


def wti_beta(r, wti_r):
    z = pd.concat([r.rename("a"), wti_r.rename("w")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vw = float(z["w"].var())
    if vw < 1e-14:
        return np.nan
    return float(z["a"].cov(z["w"]) / vw)


def main():
    closes = {s: load_close(s) for s in ACTIVE}
    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change()
    dxy_c = load_index("DXY"); dxy_r = dxy_c.pct_change()
    vix_c = load_index("VIX"); vix_r = vix_c.pct_change()
    wti_c = panel["WTI"]; wti_r = wti_c.pct_change()

    funcs = {
        "trend_r2_30_signed": (lambda c, r: trend_r2(c), 1),
        "semi_down_ratio_20": (lambda c, r: semi_down_ratio(r), -1),
        "mom_120d_skip5": (lambda c, r: mom_120(c), 1),
        "mom_10d_skip5": (lambda c, r: mom_10(c), 1),
        "vol_of_vol20x60": (lambda c, r: vol_of_vol(r), 1),
        "time_under_water_120": (lambda c, r: underwater(c), -1),
        "tail_ratio_20": (lambda c, r: tail_ratio(r), 1),
        "dxy_beta_60": (lambda c, r: dxy_beta(r, dxy_r), 1),
        "vix_beta_cond_60x20": (lambda c, r: vix_beta_cond(r, vix_r, vix_c), -1),
        "kurt_20": (lambda c, r: kurt_20(r), 1),
        "WTI_BETA_60": (lambda c, r: wti_beta(r, wti_r), 1),
    }

    # Build daily factor panel (rank IC per date)
    fwd10 = panel.shift(-10) / panel - 1.0
    ic_daily = {}
    for fid, (fn, _) in funcs.items():
        fvals = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
        for a in ACTIVE:
            c = closes[a]; r = rets[a]
            # rolling factor value series
            vals = {}
            for dt in panel.index:
                cc = c.loc[:dt]; rr = r.loc[:dt]
                try:
                    vals[dt] = fn(cc, rr)
                except Exception:
                    vals[dt] = np.nan
            fvals[a] = pd.Series(vals)
        ics = []
        for dt in panel.index:
            fv = fvals.loc[dt]
            fr = fwd10.loc[dt]
            m = fv.notna() & fr.notna()
            if m.sum() >= 6:
                ics.append((dt, fv[m].rank().corr(fr[m].rank())))
        icdf = pd.DataFrame(ics, columns=["date", "ic"]).set_index("date")["ic"]
        ic_daily[fid] = icdf

    print("=== Live rank IC (cross-sectional, active movers, 10d fwd) ===")
    for fid, (_, d) in funcs.items():
        icdf = ic_daily[fid]
        for wname, w in [("90d", 90), ("180d", 180), ("365d", 365)]:
            sub = icdf.tail(w)
            ic = sub.mean(); icir = sub.mean() / sub.std() if sub.std() > 0 else np.nan
            hit = (sub > 0).mean()
            print(f"{fid:24s} dir={d:+d}  {wname}: IC={ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f} n={len(sub)}")

    print()
    print("=== Latest factor cross-section (2030-10-02, active movers) ===")
    for fid, (fn, d) in funcs.items():
        vals = {a: fn(closes[a], rets[a]) for a in ACTIVE}
        s = pd.Series(vals).sort_values()
        print(f"{fid:24s} dir={d:+d}: " + ", ".join(f"{a}:{v:+.3f}" for a, v in s.items()))


if __name__ == "__main__":
    main()
