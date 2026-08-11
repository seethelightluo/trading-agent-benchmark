"""Screener 2026-11-05: recent cross-sectional IC sanity check (vectorized).

Data through last completed trading day 2026-11-04 (visible_through).
Implements each factor as in strategy.py but vectorized with rolling windows.
"""
import json, math
import numpy as np
import pandas as pd

DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
END = "2026-11-04"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_closes(sym):
    df = pd.read_csv(f"{DATA}/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date")["close"].astype(float)
    return df


panel = pd.DataFrame({a: load_closes(a) for a in ASSETS}).sort_index()
rets = panel.pct_change()
n = len(panel)

# --- vectorized factor panels ---
fv = {}

# trend_r2_30_signed: rolling 30d log-price R2 * sign(slope)
lp = np.log(panel)
x = np.arange(30, dtype=float)
x_c = x - x.mean()
def roll_r2_signed(s):
    y = s.values
    out = np.full(len(s), np.nan)
    for i in range(29, len(s)):
        yw = y[i-29:i+1]
        if not np.all(np.isfinite(yw)) or np.var(yw) <= 0:
            continue
        cov = np.cov(yw, x)[0, 1]
        r2 = cov * cov / (np.var(yw) * np.var(x))
        out[i] = math.copysign(r2, cov)
    return pd.Series(out, index=s.index)
fv["trend_r2_30_signed"] = lp.apply(roll_r2_signed)

# semi_down_ratio_20
def semi_down(s):
    r = s.values
    out = np.full(len(s), np.nan)
    for i in range(19, len(s)):
        w = r[i-19:i+1]
        if not np.all(np.isfinite(w)):
            continue
        down = float((np.clip(w, None, 0) ** 2).mean() ** 0.5)
        up = float((np.clip(w, 0, None) ** 2).mean() ** 0.5)
        if up < 1e-12:
            continue
        out[i] = down / up - 1.0
    return pd.Series(out, index=s.index)
fv["semi_down_ratio_20"] = rets.apply(semi_down)

# mom_120d_skip5 and mom_10d_skip5
fv["mom_120d_skip5"] = panel.shift(6) / panel.shift(126) - 1.0
fv["mom_10d_skip5"] = panel.shift(6) / panel.shift(16) - 1.0

# time_under_water_120
def tuw(s):
    v = s.values
    out = np.full(len(s), np.nan)
    for i in range(59, len(s)):
        w = v[i-119:i+1]
        if not np.all(np.isfinite(w)):
            continue
        roll = np.maximum.accumulate(w)
        idx = np.flatnonzero(w == roll)
        out[i] = float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))
    return pd.Series(out, index=s.index)
fv["time_under_water_120"] = panel.apply(tuw)

# vol_of_vol20x60
rv = rets.rolling(20).std()
fv["vol_of_vol20x60"] = rv.rolling(60).std()

# kurt_20
fv["kurt_20"] = rets.rolling(20, min_periods=8).kurt()

# dxy_beta_60
dxy = pd.read_csv(f"{IDX}/DXY.csv"); dxy["date"] = pd.to_datetime(dxy["date"])
dxy = dxy[dxy["date"] <= END].set_index("date")["close"].astype(float)
dxy_r = dxy.pct_change()
dxy_r = dxy_r.reindex(panel.index).ffill()
fv["dxy_beta_60"] = rets.rolling(60).cov(dxy_r) / dxy_r.rolling(60).var()

# vix_beta_cond_60x20
vix = pd.read_csv(f"{IDX}/VIX.csv"); vix["date"] = pd.to_datetime(vix["date"])
vix = vix[vix["date"] <= END].set_index("date")["close"].astype(float)
vix_r = vix.pct_change().reindex(panel.index).ffill()
fv["vix_beta_cond_60x20"] = rets.rolling(60).cov(vix_r) / vix_r.rolling(60).var()

# WTI_BETA_60
wti_r = panel["WTI"].pct_change()
fv["WTI_BETA_60"] = rets.rolling(60).cov(wti_r) / wti_r.rolling(60).var()

DIRS = {"trend_r2_30_signed": 1, "semi_down_ratio_20": -1, "mom_120d_skip5": 1,
        "vol_of_vol20x60": 1, "dxy_beta_60": 1, "time_under_water_120": -1,
        "kurt_20": 1, "mom_10d_skip5": 1, "vix_beta_cond_60x20": -1, "WTI_BETA_60": 1}

fwd = panel.shift(-10) / panel - 1.0


def ic_stats(fvals, fwd_ret, start):
    sub_f = fvals.loc[start:]
    sub_r = fwd_ret.loc[start:]
    ics = []
    for dt in sub_f.index:
        x = sub_f.loc[dt].dropna()
        y = sub_r.loc[dt].reindex(x.index).dropna()
        if len(x) >= 8 and len(y) >= 8:
            ic = x.rank().corr(y.rank())
            if math.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    ic_m = ics.mean()
    icir = ic_m / (ics.std(ddof=1) / math.sqrt(len(ics))) if ics.std(ddof=1) > 0 else 0.0
    hit = float((ics > 0).mean())
    return {"n": len(ics), "ic": ic_m, "icir": icir, "hit": hit}


print(f"{'factor':24s} | {'dir':>3s} | {'120d ic/icir/hit':>20s} | {'60d ic/icir/hit':>20s} | {'30d ic':>7s}")
rows = []
for fid in DIRS:
    fvals = fv[fid]
    r120 = ic_stats(fvals, fwd, panel.index[-120])
    r60 = ic_stats(fvals, fwd, panel.index[-60])
    r30 = ic_stats(fvals, fwd, panel.index[-30])
    def fmt(r):
        if r is None:
            return "n/a"
        return f"{r['ic']:+.3f}/{r['icir']:+.2f}/{r['hit']:.2f}"
    print(f"{fid:24s} | {DIRS[fid]:>3d} | {fmt(r120):>20s} | {fmt(r60):>20s} | {r30['ic'] if r30 else float('nan'):+.3f}")
    rows.append((fid, DIRS[fid], r120, r60, r30))

# recent correlation matrix (60d)
print("\nRecent 60d avg cross-sectional spearman correlation:")
fids = list(DIRS.keys())
corr_mat = pd.DataFrame(np.eye(len(fids)), index=fids, columns=fids)
for i, f1 in enumerate(fids):
    for j, f2 in enumerate(fids):
        if j <= i:
            continue
        cs = []
        for dt in panel.index[-60:]:
            x = fv[f1].loc[dt].dropna()
            y = fv[f2].loc[dt].reindex(x.index).dropna()
            if len(x) >= 8:
                c = x.rank().corr(y.rank())
                if math.isfinite(c):
                    cs.append(c)
        if cs:
            corr_mat.loc[f1, f2] = corr_mat.loc[f2, f1] = float(np.mean(cs))
print(corr_mat.round(2).to_string())

print("\nLatest factor snapshot (2026-11-04):")
for fid in fids:
    last = fv[fid].iloc[-1].dropna()
    print(f"  {fid:24s} valid={len(last):2d}  top3={list(last.nlargest(3).index)}  bot3={list(last.nsmallest(3).index)}")

json.dump({fid: {"dir": d, "ic120": (r120 and r120["ic"]), "icir120": (r120 and r120["icir"]),
                 "ic60": (r60 and r60["ic"]), "icir60": (r60 and r60["icir"]),
                 "ic30": (r30 and r30["ic"])}
           for fid, d, r120, r60, r30 in rows},
          open("scripts/screener_20261105_ic_results.json", "w"), indent=1)
print("\nsaved scripts/screener_20261105_ic_results.json")
