"""SCREENER cycle 2027-11-18: regime assessment + fresh factor IC/quality evaluation.

Data visible only through 2027-11-17. No live-account/backtest/step interaction.
Mirrors strategy.py factor implementations on the 15-asset tradable universe.
"""
import numpy as np
import pandas as pd
import json

END = "2027-11-17"
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


# ---------------- rolling factor implementations (mirror strategy.py) ----------------
def hist_trend_r2(c):
    out = pd.Series(np.nan, index=c.index)
    y = np.log(c)
    x = np.arange(len(c))
    for i in range(29, len(c)):
        yy = y.values[i - 29:i + 1]
        xx = x[i - 29:i + 1].astype(float)
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
    rollmax = c.rolling(120, min_periods=60).max()
    eq = (c == rollmax)
    out = pd.Series(np.nan, index=c.index)
    idx = np.flatnonzero(eq.values)
    j = 0
    for i in range(len(c)):
        while j < len(idx) and idx[j] <= i:
            j += 1
        if j == 0:
            out.iloc[i] = np.nan
        else:
            out.iloc[i] = i - idx[j - 1]
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
    cov = r.rolling(60, min_periods=30).cov(dxy_r)
    var = dxy_r.rolling(60, min_periods=30).var()
    return cov / var


def hist_vix_beta(r):
    cov = r.rolling(60, min_periods=30).cov(vix_r)
    var = vix_r.rolling(60, min_periods=30).var()
    beta = cov / var
    vmove = vix_c / vix_c.shift(20) - 1.0
    return -beta * vmove


wti_c = load_close("../persistent/stock_data/WTI.csv")
wti_r = wti_c.pct_change()


def hist_wti_beta(r):
    cov = r.rolling(60, min_periods=30).cov(wti_r)
    var = wti_r.rolling(60, min_periods=30).var()
    return cov / var


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
    "wti_beta_60": hist_wti_beta,
}

panel_tail = panel.tail(400)
fvals = {}
for fid, fn in hist_fns.items():
    fvals[fid] = panel_tail.apply(fn, axis=0)

fwd10 = panel.shift(-10) / panel - 1.0
fwd10_tail = fwd10.tail(400)


def rank_ic(fv, fwd, min_assets=8):
    ics, dates = [], []
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


ic_series = {fid: rank_ic(fvals[fid], fwd10_tail) for fid in fvals}

# ---------------- regime assessment ----------------
mkt = rets.mean(axis=1)
print("\n=== REGIME (through %s) ===" % END)
print("Avg cross-asset 20d ret: {:+.3%}".format(mkt.tail(20).mean()))
print("Avg cross-asset 60d ret: {:+.3%}".format(mkt.tail(60).mean()))
print("Avg cross-asset 120d ret: {:+.3%}".format(mkt.tail(120).mean()))
print("20d cross-sectional dispersion (std of 20d asset rets): {:.3%}".format(rets.tail(20).mean().std()))
print("60d cross-sectional dispersion: {:.3%}".format(rets.tail(60).mean().std()))
vol20 = rets.tail(20).std()
print("Median 20d realized vol: {:.3%} (min {:.3%} max {:.3%})".format(vol20.median(), vol20.min(), vol20.max()))
print("VIX last: {:.2f}  (20d ago: {:.2f}, 60d ago: {:.2f})".format(
    vix_c.iloc[-1], vix_c.iloc[-21], vix_c.iloc[-61]))
print("DXY last: {:.2f}  (20d ago: {:.2f})".format(dxy_c.iloc[-1], dxy_c.iloc[-21]))
ma20 = panel.rolling(20).mean().iloc[-1]
ma60 = panel.rolling(60).mean().iloc[-1]
above = (ma20 / ma60 - 1.0).dropna()
print("MA20/MA60 ratio: assets above MA60: %d/15" % (above > 0).sum())
print("MA20/MA60 distribution: min {:+.3f} max {:+.3f} mean {:+.3f}".format(
    above.min(), above.max(), above.mean()))
print("Last 5d market avg ret: {:+.3%}".format(mkt.tail(5).mean()))
print("Per-asset 20d ret:")
for a in ASSETS:
    print("  {:10s} {:+.2%}".format(a, rets[a].tail(20).sum()))

# ---------------- factor IC summary ----------------
print("\n=== FACTOR RECENT IC (10d fwd, cross-sectional rank) ===")
exp_dir = {
    "trend_r2_30_signed": 1, "semi_down_ratio_20": -1, "mom_120d_skip5": 1,
    "mom_10d_skip5": 1, "vol_of_vol20x60": 1, "time_under_water_120": -1,
    "tail_ratio_20": 1, "kurt_20": 1, "dxy_beta_60": 1,
    "vix_beta_cond_60x20": -1, "wti_beta_60": 1,
}
rows = []
for fid, ics in ic_series.items():
    for lbl, w in [("ic120", 120), ("ic60", 60), ("ic40", 40)]:
        s = ics.tail(w)
        if len(s) < 10:
            continue
        ic = s.mean()
        icir = ic / (s.std() + 1e-9)
        rows.append((fid, lbl, ic, icir, len(s)))
for fid, lbl, ic, icir, n in rows:
    ed = exp_dir.get(fid, 1)
    print("{:22s} {:5s} ic={:+.4f} icir={:+.2f} n={:3d} consistent={:+.4f} {}".format(
        fid, lbl, ic, icir, n, ic * ed, "OK" if ic * ed > 0 else "INVERTED"))

# ---------------- factor correlation (recent 120d) ----------------
print("\n=== FACTOR CROSS-CORRELATION (recent 120d rank exposures) ===")
rank_dates = fvals[list(fvals)[0]].tail(120).index
corr_rows = {}
for fid in fvals:
    fv = fvals[fid].tail(120)
    corr_rows[fid] = fv.T.apply(lambda col: col.rank(pct=True))
corr_df = pd.DataFrame(corr_rows, index=rank_dates)
corr_mat = corr_df.corr()
corr_pairs = []
for i, f1 in enumerate(corr_mat.columns):
    for f2 in corr_mat.columns[i + 1:]:
        r = corr_mat.loc[f1, f2]
        if abs(r) > 0.6:
            corr_pairs.append((f1, f2, r))
for f1, f2, r in sorted(corr_pairs, key=lambda t: -abs(t[2])):
    print("{:22s} {:22s} r={:+.2f}".format(f1, f2, r))

# ---------------- quality tilt ensemble ----------------
print("\n=== QUALITY TILT (q=|IC120|*|ICIR120|, sign preserved) ===")
quals = []
for fid, ics in ic_series.items():
    s = ics.tail(120)
    if len(s) < 40:
        continue
    ic = s.mean()
    icir = ic / (s.std() + 1e-9)
    if not np.isfinite(ic) or not np.isfinite(icir):
        continue
    q = abs(ic) * abs(icir)
    quals.append((fid, ic, icir, q))
quals.sort(key=lambda t: -t[3])
for fid, ic, icir, q in quals:
    print("{:22s} ic120={:+.4f} icir={:+.2f} q={:.5f}".format(fid, ic, icir, q))

print("\n--- 60d quality view ---")
quals60 = []
for fid, ics in ic_series.items():
    s = ics.tail(60)
    if len(s) < 20:
        continue
    ic = s.mean()
    icir = ic / (s.std() + 1e-9)
    if not np.isfinite(ic) or not np.isfinite(icir):
        continue
    q = abs(ic) * abs(icir)
    quals60.append((fid, ic, icir, q))
quals60.sort(key=lambda t: -t[3])
for fid, ic, icir, q in quals60:
    print("{:22s} ic60={:+.4f} icir={:+.2f} q={:.5f}".format(fid, ic, icir, q))

# ---------------- current factor values snapshot ----------------
print("\n=== CURRENT FACTOR RANK EXPOSURES (last date) ===")
last = panel_tail.index[-1]
for fid in fvals:
    fv = fvals[fid].loc[last]
    print("{:22s} vals: ".format(fid), " ".join(
        "{:.3f}".format(x) if np.isfinite(x) else "  NA" for x in fv.values))

# ---------------- save IC series for reference ----------------
ic_series.to_csv("scripts/_ic_series_20271118.csv")
