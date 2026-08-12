"""Screener cycle 2027-08-12: regime assessment + fresh factor IC/quality evaluation.

Data visible only through 2027-08-11. No live-account/backtest/step interaction.
"""
import numpy as np
import pandas as pd

END = "2027-08-11"
ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_close(fn):
    df = pd.read_csv(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date")["close"].astype(float)
    return df

closes = {}
for a in ASSETS:
    try:
        closes[a] = load_close(f"../persistent/stock_data/{a}.csv")
    except Exception as e:
        print("ERR", a, e)

panel = pd.DataFrame(closes).sort_index()
panel = panel.dropna(how="all")
rets = panel.pct_change()
print("Panel dates:", panel.index.min().date(), "->", panel.index.max().date(), "n=", len(panel))

dxy_c = load_close("../persistent/index_data/DXY.csv")
vix_c = load_close("../persistent/index_data/VIX.csv")
dxy_r = dxy_c.pct_change()
vix_r = vix_c.pct_change()

# ---------------- per-asset factor implementations (mirror strategy.py) ----------------
def trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return np.nan
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
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

def kurt_20(r):
    s = r.dropna().tail(40)
    if len(s) < 20:
        return np.nan
    k = s.rolling(20, min_periods=8).kurt().iloc[-1]
    return float(k) if np.isfinite(k) else np.nan

def tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return np.nan
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return np.nan
    return q95 / abs(q05)

def dxy_beta(r, dr):
    z = pd.concat([r.rename("a"), dr.rename("d")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vd = float(z["d"].var())
    if vd < 1e-14:
        return np.nan
    return float(z["a"].cov(z["d"]) / vd)

def vix_beta_cond(r, vr, vc):
    z = pd.concat([r.rename("a"), vr.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vv = float(z["v"].var())
    if vv < 1e-14:
        return np.nan
    beta = float(z["a"].cov(z["v"]) / vv)
    if vc is None or len(vc) < 22:
        return np.nan
    v0 = float(vc.iloc[-21])
    if v0 <= 0:
        return np.nan
    vmove = float(vc.iloc[-1]) / v0 - 1.0
    return -beta * vmove

# Build rolling factor panel per asset via expanding apply (only compute on last 400 rows for speed)
def build_panel(fn, src, need_macro=None):
    out = {}
    for a in ASSETS:
        s = src[a]
        if need_macro:
            out[a] = fn(s, *need_macro)
        else:
            out[a] = fn(s)
    return pd.Series(out)

fvals = {}
# trend & momentum & vol factors: compute on last 260 rows to get history for IC
for a in ASSETS:
    panel[a] = panel[a]

def hist_trend_r2(c):
    out = pd.Series(np.nan, index=c.index)
    y = np.log(c)
    x = np.arange(len(c))
    for i in range(29, len(c)):
        yy = y.values[i-29:i+1]
        xx = x[i-29:i+1].astype(float)
        cov = np.cov(yy, xx)[0, 1]
        vy, vx = np.var(yy), np.var(xx)
        if vy > 0 and vx > 0:
            out.iloc[i] = np.copysign(cov * cov / (vy * vx), cov)
    return out

def hist_semi(r):
    d = (r.clip(upper=0) ** 2).rolling(20, min_periods=10).mean() ** 0.5
    u = (r.clip(lower=0) ** 2).rolling(20, min_periods=10).mean() ** 0.5
    return d / u - 1.0

def hist_underwater(c):
    # vectorized: days since last rolling max within trailing 120 window
    rollmax = c.rolling(120, min_periods=60).max()
    eq = (c == rollmax)
    # for each date, distance back to last date where c==rollmax
    out = pd.Series(np.nan, index=c.index)
    idx = np.flatnonzero(eq.values)
    j = 0
    for i in range(len(c)):
        while j < len(idx) and idx[j] <= i:
            j += 1
        if j == 0:
            out.iloc[i] = np.nan
        else:
            out.iloc[i] = i - idx[j-1]
    return out

def hist_mom120(c):
    return c.shift(5) / c.shift(125) - 1.0

def hist_mom10(c):
    return c.shift(5) / c.shift(15) - 1.0

def hist_volvol(r):
    return r.rolling(20).std().rolling(60, min_periods=45).std()

def hist_kurt(r):
    return r.rolling(20, min_periods=8).kurt()

def hist_tail(r):
    return r.rolling(20, min_periods=10).quantile(0.95) / r.rolling(20, min_periods=10).quantile(0.05).abs()

def hist_dxy_beta(r):
    z = pd.concat([r, dxy_r], axis=1)
    cov = z.rolling(60, min_periods=30).cov().iloc[0::2, 1].droplevel(0)
    var = z.iloc[:, 1].rolling(60, min_periods=30).var()
    return cov / var

def hist_vix_beta(r):
    z = pd.concat([r, vix_r], axis=1)
    cov = z.rolling(60, min_periods=30).cov().iloc[0::2, 1].droplevel(0)
    var = z.iloc[:, 1].rolling(60, min_periods=30).var()
    beta = cov / var
    vmove = vix_c / vix_c.shift(20) - 1.0
    return -beta * vmove

hist_fns = {
    "trend_r2_30_signed": hist_trend_r2,
    "semi_down_ratio_20": hist_semi,
    "mom_120d_skip5": hist_mom120,
    "mom_10d_skip5": hist_mom10,
    "vol_of_vol20x60": hist_volvol,
    "time_under_water_120": hist_underwater,
    "tail_ratio_20": hist_tail,
    "kurt_20": hist_kurt,
    "dxy_beta_60": hist_dxy_beta,
    "vix_beta_cond_60x20": hist_vix_beta,
}

# use last 300 rows for speed
panel_tail = panel.tail(400)
fvals = {}
for fid, fn in hist_fns.items():
    df = panel_tail.apply(fn, axis=0)
    fvals[fid] = df

fwd10 = panel.shift(-10) / panel - 1.0
fwd10_tail = fwd10.tail(400)

def rank_ic(fv, fwd, min_assets=8):
    ics = []
    dates = []
    for dt in fwd.index:
        if dt not in fv.index:
            continue
        x = fv.loc[dt]
        y = fwd.loc[dt]
        m = pd.concat([x, y], axis=1).dropna()
        if len(m) >= min_assets:
            ics.append(m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank()))
            dates.append(dt)
    return pd.Series(ics, index=dates)

ic_series = {}
for fid in fvals:
    ic_series[fid] = rank_ic(fvals[fid], fwd10_tail)

# ---------------- regime assessment ----------------
mkt = rets.mean(axis=1)
print("\n=== REGIME (through 2027-08-11) ===")
print("Avg cross-asset 20d ret: {:+.3%}".format(mkt.tail(20).mean()))
print("Avg cross-asset 60d ret: {:+.3%}".format(mkt.tail(60).mean()))
print("Avg cross-asset 120d ret: {:+.3%}".format(mkt.tail(120).mean()))
print("20d cross-sectional dispersion (std of 20d asset rets): {:.3%}".format(
    rets.tail(20).mean().std()))
print("60d cross-sectional dispersion: {:.3%}".format(rets.tail(60).mean().std()))
vol20 = rets.tail(20).std()
print("Median 20d realized vol: {:.3%} (min {:.3%} max {:.3%})".format(
    vol20.median(), vol20.min(), vol20.max()))
print("VIX latest:", round(float(vix_c.iloc[-1]), 2), " 20d chg: {:+.1%}".format(
    float(vix_c.iloc[-1] / vix_c.iloc[-21] - 1)))
print("DXY 20d chg: {:+.2%}".format(float(dxy_c.iloc[-1] / dxy_c.iloc[-21] - 1)))
r20 = rets.tail(20).mean().sort_values(ascending=False)
print("\n20d asset returns:")
for a, v in r20.items():
    print("  {:>12s} {:+.2%}".format(a, v))
r60 = rets.tail(60).mean().sort_values(ascending=False)
print("\n60d asset returns:")
for a, v in r60.items():
    print("  {:>12s} {:+.2%}".format(a, v))

# ---------------- factor IC stats ----------------
print("\n=== FACTOR IC STATS ===")
rows = []
for fid, s in ic_series.items():
    s = s.dropna()
    for label, win in [("ic60", 60), ("ic120", 120), ("ic250", 250), ("full", None)]:
        seg = s if win is None else s.tail(win)
        if len(seg) < 20:
            continue
        ic = seg.mean()
        icir = seg.mean() / seg.std() if seg.std() > 0 else 0.0
        hit = (seg > 0).mean()
        rows.append({"factor": fid, "window": label, "ic": ic, "icir": icir,
                     "hit": hit, "n": len(seg), "q": abs(ic) * abs(icir)})

tbl = pd.DataFrame(rows)
for w in ["ic60", "ic120", "ic250", "full"]:
    seg = tbl[tbl.window == w].sort_values("q", ascending=False)
    print("\n--- window:", w, "---")
    print(seg[["factor", "ic", "icir", "hit", "n", "q"]].to_string(index=False))

# ---------------- factor cross-sectional correlation (latest) ----------------
print("\n=== FACTOR CROSS-SECTIONAL CORRELATION (latest date) ===")
last = panel.index[-1]
mat = {}
for fid, fv in fvals.items():
    if last in fv.index:
        mat[fid] = fv.loc[last]
corr = pd.DataFrame(mat).corr()
print(corr.round(2).to_string())

print("\n=== AVG PAIRWISE |CORR| (last 60d cross-sections) ===")
pairs = []
recent_dates = fwd10_tail.index[-60:]
for dt in recent_dates:
    cols = {}
    for fid, fv in fvals.items():
        if dt in fv.index:
            cols[fid] = fv.loc[dt]
    if len(cols) >= 4:
        c = pd.DataFrame(cols).corr().abs()
        tri = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
        pairs.append(tri.stack())
if pairs:
    pc = pd.concat(pairs).groupby(level=[0, 1]).mean()
    print(pc.sort_values(ascending=False).head(15).round(3).to_string())

pd.to_pickle(ic_series, "scripts/_screener_ic_20270812.pkl")
print("\nSaved IC series to scripts/_screener_ic_20270812.pkl")
