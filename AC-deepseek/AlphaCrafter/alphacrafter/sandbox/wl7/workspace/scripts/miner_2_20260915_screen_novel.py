"""
miner_2 cycle 2026-09-15: batch screen of novel factor families.
Target: low pairwise correlation vs the 8 live library factors
  (rel_mom_20d_skip5, beta_ew_60d, downside_vol_ratio_20, max_ret_20d,
   eurusd_beta_cond_60x20, corr_ew_60, dxy_beta_cond_60x20, kurt_20d_skip5)
Admission gates (benchmark-wide): |IC| >= 0.0070, |ICIR| >= 0.0840 on h10,
validated on warm-up 2020-01-01..2026-07-15; pairwise rho < 0.5 gate.

New families explored:
  1. wd_ret_12w        calendar weekday seasonality (mean same-weekday ret, 12w)
  2. mon_ret_3y        calendar month seasonality (mean same-month ret, 3y)
  3. ovn_ret_20        overnight gap momentum (mean open/prev_close-1, 20d)
  4. intra_ret_20      intraday strength (mean close/open-1, 20d)
  5. ovn_share_20      overnight variance share (20d)
  6. vol_exp_20x60     volume expansion (vol20/vol60)
  7. retvol_corr_20    return-volume correlation (20d)
  8. vol_slope_20x120  vol term slope ln(vol20/vol120)
  9. skew_20d_skip5    realized skewness (20d, skip5)
 10. days_since_high_60 duration since 60d high
 11. clv_20            close location value (20d mean of (C-L)/(H-L))
 12. up_frac_20        upside participation (20d)
 13. res_mom_20_skip5  residual momentum vs EW basket (60d beta, 20d skip5)
 14. range_exp_20x60   range expansion (range20/range60)
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8
WARM_END = "2026-07-15"
DAYS = 4000
H = 10


def load_ohlcv(days=DAYS):
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is None or not len(df):
            continue
        df = df.set_index("date")
        closes[s] = df["close"].astype(float)
        opens[s] = df["open"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        vols[s] = df["volume"].astype(float)

    def _p(d):
        p = pd.concat(d, axis=1, sort=True)
        return p[~p.index.duplicated(keep="last")].sort_index()
    return _p(closes), _p(opens), _p(highs), _p(lows), _p(vols)


def load_macro():
    out = {}
    for s in MACRO:
        df = get_index_daily_data(s, days=DAYS)
        if df is not None and len(df):
            out[s] = df.set_index("date")["close"].astype(float)
    return out


def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor, fwd):
    f = factor.stack(dropna=False).rename("f")
    r = fwd.stack(dropna=False).rename("r")
    j = pd.concat([f, r], axis=1).dropna()
    if len(j) == 0:
        return pd.Series(dtype=float)
    j["fr"] = j.groupby(level=0)["f"].rank()
    j["rr"] = j.groupby(level=0)["r"].rank()
    cnt = j.groupby(level=0).size()
    keep = cnt[cnt >= MIN_ASSETS].index
    j = j[j.index.get_level_values(0).isin(keep)]
    g = j.groupby(level=0)[["fr", "rr"]]
    n = g.size()
    sx, sy = g["fr"].sum(), g["rr"].sum()
    sxx, syy = (g["fr"] ** 2).sum(), (g["rr"] ** 2).sum()
    sxy = (g["fr"] * g["rr"]).sum()
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    ic = num / den
    return ic.sort_index()


def summarize(factor, fwd, lo=None, hi=None):
    ic = rank_ic_series(factor, fwd)
    if lo is not None:
        ic = ic[(ic.index >= lo) & (ic.index <= hi)]
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ic) == 0:
        return None
    icr = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0.0
    return {"ic": float(ic.mean()), "icir": float(icr),
            "hit": float((ic > 0).mean()), "n": int(len(ic))}


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1)
    out, dates = [], ranks.index
    for i in range(10, len(dates)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def spearman_panel(fa, fb):
    a = fa.stack(dropna=False).rename("a")
    b = fb.stack(dropna=False).rename("b")
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 30:
        return float("nan")
    ra = j["a"].rank()
    rb = j["b"].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------- load data ----------------
close, open_, high, low, vol = load_ohlcv()
macro = load_macro()
ret = close.pct_change()
fwd10 = fwd_returns(close, H)
print("data span:", close.index.min().date(), "->", close.index.max().date(),
      "assets:", close.shape[1], "dates:", close.shape[0])
cov_asset_days = close.notna().sum().sum() / (close.shape[0] * close.shape[1])
print(f"coverage asset-days: {cov_asset_days:.3f}")

# ---------------- library factors (8 live) ----------------
ew = close.mean(axis=1)
ew_r = ew.pct_change()
lib = {}
m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
cols = {}
for a in close.columns:
    s = close[a].dropna()
    er = ew_r.reindex(s.index)
    z = pd.concat([s.pct_change().rename("r"), er.rename("m")], axis=1).dropna()
    cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
lib["beta_ew_60d"] = pd.DataFrame(cols, index=close.index)

def dsvr(s):
    rr = s.pct_change()
    down = rr.where(rr < 0, 0.0)
    ds = np.sqrt((down ** 2).rolling(20).mean())
    tot = rr.rolling(20).std()
    return -(ds / tot)
lib["downside_vol_ratio_20"] = per_asset(dsvr)(close)
lib["max_ret_20d"] = ret.rolling(20).max()
cols = {}
for a in close.columns:
    s = close[a].dropna()
    er = ew_r.reindex(s.index)
    z = pd.concat([s.pct_change().rename("r"), er.rename("m")], axis=1).dropna()
    c = z["r"].rolling(60).corr(z["m"])
    corrs = pd.Series(np.nan, index=s.index)
    corrs.loc[c.index] = c
    cols[a] = corrs
lib["corr_ew_60"] = pd.DataFrame(cols, index=close.index)
eur = macro["EURUSD"].dropna()
eur20 = (eur / eur.shift(20) - 1.0)
cols = {}
for a in close.columns:
    s = close[a].dropna()
    z = pd.concat([s.pct_change().rename("r"), eur.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
    beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
    cols[a] = (beta * eur20.reindex(s.index))
lib["eurusd_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)
dxy = macro["DXY"].dropna()
dxy20 = (dxy / dxy.shift(20) - 1.0)
cols = {}
for a in close.columns:
    s = close[a].dropna()
    z = pd.concat([s.pct_change().rename("r"), dxy.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
    beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
    cols[a] = (-beta * dxy20.reindex(s.index))
lib["dxy_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)

def kurt(s):
    rr = s.pct_change().shift(5)
    return rr.rolling(20, min_periods=12).kurt()
lib["kurt_20d_skip5"] = per_asset(kurt)(close)

# ---------------- candidate factors ----------------
cand = {}

# 1. weekday seasonality: mean same-weekday return over trailing 12 weeks, skip 5
def wd_ret(s):
    rr = s.pct_change()
    out = pd.Series(np.nan, index=s.index)
    for i in range(60, len(s)):
        past = rr.iloc[i - 60:i - 5]
        same = past[past.index.weekday == s.index[i].weekday()]
        if len(same) >= 6:
            out.iloc[i] = same.mean()
    return out
cand["wd_ret_12w"] = per_asset(wd_ret)(close)

# 2. month seasonality: mean return in same calendar month over prior 3 years
def mon_ret(s):
    rr = s.pct_change()
    out = pd.Series(np.nan, index=s.index)
    for i in range(250, len(s)):
        d = s.index[i]
        past = rr.iloc[:i]
        same = past[(past.index.month == d.month) & (past.index.year >= d.year - 3)]
        if len(same) >= 30:
            out.iloc[i] = same.mean()
    return out
cand["mon_ret_3y"] = per_asset(mon_ret)(close)

# 3. overnight gap momentum 20d
ovn = (open_ / close.shift(1) - 1.0)
cand["ovn_ret_20"] = ovn.rolling(20).mean()

# 4. intraday strength 20d
intra = (close / open_ - 1.0)
cand["intra_ret_20"] = intra.rolling(20).mean()

# 5. overnight variance share 20d
ovn2 = (ovn ** 2).rolling(20).mean()
intra2 = (intra ** 2).rolling(20).mean()
cand["ovn_share_20"] = ovn2 / (ovn2 + intra2)

# 6. volume expansion 20x60
cand["vol_exp_20x60"] = vol.rolling(20).mean() / vol.rolling(60).mean()

# 7. return-volume correlation 20d
vchg = vol.pct_change()
def rv_corr(s):
    rr = s.pct_change()
    vv = vchg[s.name].reindex(s.index)
    z = pd.DataFrame({"r": rr, "v": vv}).dropna()
    c = z["r"].rolling(20, min_periods=10).corr(z["v"])
    out = pd.Series(np.nan, index=s.index)
    out.loc[c.index] = c
    return out
cand["retvol_corr_20"] = per_asset(rv_corr)(close)

# 8. vol term slope ln(vol20/vol120)
rv20 = ret.rolling(20).std()
rv120 = ret.rolling(120).std()
cand["vol_slope_20x120"] = np.log(rv20 / rv120)

# 9. realized skewness 20d skip5
def skew(s):
    rr = s.pct_change().shift(5)
    return rr.rolling(20, min_periods=12).skew()
cand["skew_20d_skip5"] = per_asset(skew)(close)

# 10. days since 60d high
def days_high(s, w=60):
    out = pd.Series(np.nan, index=s.index)
    roll_max = s.rolling(w, min_periods=w // 2).max()
    cnt = 0
    prev_max = np.nan
    for i in range(len(s)):
        if np.isnan(roll_max.iloc[i]):
            continue
        if np.isnan(prev_max) or s.iloc[i] >= roll_max.iloc[i]:
            cnt = 0
            prev_max = s.iloc[i]
        else:
            cnt += 1
        out.iloc[i] = cnt
    return out
cand["days_since_high_60"] = per_asset(days_high)(close)

# 11. close location value 20d
clv = ((close - low) / (high - low)).rolling(20).mean()
cand["clv_20"] = clv

# 12. upside participation 20d
cand["up_frac_20"] = (ret > 0).rolling(20).mean()

# 13. residual momentum vs EW basket (60d beta), 20d cum residual, skip 5
def res_mom(s):
    er = ew_r.reindex(s.index)
    rr = s.pct_change()
    z = pd.DataFrame({"r": rr, "m": er}).dropna()
    beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    resid = z["r"] - beta * z["m"]
    out = pd.Series(np.nan, index=s.index)
    cum = resid.rolling(20).sum().shift(5)
    out.loc[cum.index] = cum
    return out
cand["res_mom_20_skip5"] = per_asset(res_mom)(close)

# 14. range expansion 20x60
rng20 = (high.rolling(20).max() - low.rolling(20).min())
rng60 = (high.rolling(60).max() - low.rolling(60).min())
cand["range_exp_20x60"] = rng20 / rng60

# ---------------- evaluation ----------------
print(f"\n{'factor':<22}{'IC':>8}{'ICIR':>8}{'hit':>6}{'n':>6}{'cov':>6}{'turn':>7}{'maxrho':>8}  recentIC")
for name, panel in cand.items():
    cov = panel.notna().sum().sum() / (panel.shape[0] * panel.shape[1])
    s = summarize(panel, fwd10, lo="2020-01-01", hi=WARM_END)
    sr = summarize(panel, fwd10, lo="2026-01-01", hi="2026-09-14")
    if s is None:
        print(f"{name:<22} no-valid-dates")
        continue
    turn = turnover_10d_rank(panel)
    rhos = [spearman_panel(panel, lv) for lv in lib.values()]
    maxrho = max((abs(r) for r in rhos if r == r), default=float("nan"))
    ric = sr["ic"] if sr else float("nan")
    print(f"{name:<22}{s['ic']:>8.4f}{s['icir']:>8.3f}{s['hit']:>6.2f}{s['n']:>6d}"
          f"{cov:>6.2f}{turn:>7.2f}{maxrho:>8.2f}  {ric:+.4f}")

print("\n--- library maxrho per candidate (max |rho| vs live factors) ---")
for name, panel in cand.items():
    best = (float("nan"), "")
    for k, lv in lib.items():
        r = spearman_panel(panel, lv)
        if r == r and abs(r) > abs(best[0]):
            best = (r, k)
    print(f"{name:<22} maxrho={best[0]:.2f} vs {best[1]}")
