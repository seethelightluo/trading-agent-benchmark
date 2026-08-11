"""miner_1 2027-04-02: screen NOVEL factor families (trend-quality, risk-adj momentum,
xsec relative strength, vol asymmetry, macro-beta, path efficiency) for the 15-instrument
cross-asset universe. Data through last completed trading day 2027-04-01.

Gates (shared): abs(daily paper IC) >= 0.0070 and abs(daily paper ICIR) >= 0.0840.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_VALID = 8
DAYS = 2000
END = pd.Timestamp("2027-04-01")

frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=DAYS)
    if df is None or len(df) < 300:
        print("SKIP", a, 0 if df is None else len(df)); continue
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    frames[a] = df.set_index("date").sort_index()

idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
idx = pd.DatetimeIndex([d for d in idx if pd.Timestamp("2020-01-01") <= d <= END])
C = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
O = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
H = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
L = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
V = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
for a in WATCH:
    if a in frames:
        f = frames[a]
        C.loc[f.index, a] = f["close"].values
        O.loc[f.index, a] = f["open"].values
        H.loc[f.index, a] = f["high"].values
        L.loc[f.index, a] = f["low"].values
        V.loc[f.index, a] = f["volume"].values
C = C.ffill(); O = O.ffill(); H = H.ffill(); L = L.ffill(); V = V.ffill()
R = C.pct_change()
lnC = np.log(C)

# Macro observation-only signals
def load_macro(name):
    m = pd.read_csv(f"../persistent/index_data/{name}.csv")
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] <= END].set_index("date")
    col = "close" if "close" in m.columns else m.columns[1]
    return m[col].astype(float).reindex(C.index).ffill()

DXY = load_macro("DXY"); VIX = load_macro("VIX")

def ols_tstat_series(y, w):
    """rolling OLS slope t-stat of a single Series y over window w"""
    out = pd.Series(index=y.index, dtype=float)
    x = np.arange(w, dtype=float)
    xm = x - x.mean()
    sxx = (xm ** 2).sum()
    yv = y.values
    for i in range(w - 1, len(y)):
        yw = yv[i - w + 1:i + 1]
        if not np.all(np.isfinite(yw)):
            continue
        ym = yw.mean()
        slope = np.sum((yw - ym) * xm) / sxx
        resid = yw - ym - slope * xm
        s2 = np.sum(resid ** 2) / (w - 2)
        se = np.sqrt(s2 / sxx)
        out.iloc[i] = slope / se
    return out

def build(fid):
    fdf = pd.DataFrame(index=C.index, columns=WATCH, dtype=float)
    for a in WATCH:
        c, o, h, l, v = C[a], O[a], H[a], L[a], V[a]
        r = R[a]
        if fid == "slope_tstat_60":
            fdf[a] = ols_tstat_series(lnC[a], 60)
        elif fid == "sharpe_mom_60_20":
            fdf[a] = (c.shift(5) / c.shift(65) - 1.0) / r.rolling(20).std()
        elif fid == "vol_ratio_5_60":
            fdf[a] = r.rolling(5).std() / r.rolling(60).std()
        elif fid == "range_pos_20":
            fdf[a] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min())
        elif fid == "xsec_mom_60_skip5":
            fdf[a] = c.shift(5) / c.shift(65) - 1.0
        elif fid == "dd_20":
            fdf[a] = c / c.rolling(20).max() - 1.0
        elif fid == "dd_60":
            fdf[a] = c / c.rolling(60).max() - 1.0
        elif fid == "vol_asym_20":
            dn = r.where(r < 0, np.nan); up = r.where(r > 0, np.nan)
            fdf[a] = dn.rolling(20).std() / up.rolling(20).std()
        elif fid == "gainloss_20":
            dn = r.where(r < 0, np.nan); up = r.where(r > 0, np.nan)
            fdf[a] = up.rolling(20).mean() / (-dn.rolling(20).mean())
        elif fid == "ret_vol_corr_20":
            vr = v.pct_change()
            fdf[a] = r.rolling(20).corr(vr)
        elif fid == "hl_range_20":
            fdf[a] = ((h - l) / c).rolling(20).mean()
        elif fid == "efficiency_60":
            fdf[a] = (c.shift(5) / c.shift(65) - 1.0).abs() / r.abs().rolling(60).sum()
        elif fid == "max_ret_20":
            fdf[a] = r.rolling(20).max()
        elif fid == "skew_20":
            fdf[a] = r.rolling(20).skew()
        elif fid == "wti_beta_60":
            wr = R["WTI"]
            fdf[a] = r.rolling(60).cov(wr) / wr.rolling(60).var()
        elif fid == "spx_beta_60":
            sr = R["SPX"]
            fdf[a] = r.rolling(60).cov(sr) / sr.rolling(60).var()
        elif fid == "btc_beta_60":
            br = R["BTC"]
            fdf[a] = r.rolling(60).cov(br) / br.rolling(60).var()
        elif fid == "dxy_beta_60":
            dr = DXY.pct_change()
            fdf[a] = r.rolling(60).cov(dr) / dr.rolling(60).var()
        elif fid == "vix_beta_60":
            vr2 = VIX.pct_change()
            fdf[a] = r.rolling(60).cov(vr2) / vr2.rolling(60).var()
        elif fid == "gap_ret_1":
            fdf[a] = o / c.shift(1) - 1.0
        elif fid == "intraday_ret_1":
            fdf[a] = c / o - 1.0
        elif fid == "mom_20_skip5":
            fdf[a] = c.shift(5) / c.shift(25) - 1.0
        else:
            return None
    if fid == "xsec_mom_60_skip5":
        fdf = fdf.sub(fdf.mean(axis=1), axis=0)
    return fdf.replace([np.inf, -np.inf], np.nan)

def daily_ic(fdf, fwd):
    fr = C.shift(-fwd) / C - 1.0
    out = []
    for i in range(len(fdf)):
        fv = fdf.iloc[i].values; rv = fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_VALID: continue
        if np.all(fv[m] == fv[m][0]): continue
        rho = spearmanr(fv[m], rv[m]).correlation
        out.append(rho if np.isfinite(rho) else np.nan)
    return np.array(out)

def stats(ic):
    ok = np.isfinite(ic)
    if ok.sum() < 30: return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan, hit=np.nan)
    ic = ic[ok]
    return dict(n=int(ok.sum()), ic=float(np.nanmean(ic)), icir=float(np.nanmean(ic) / np.nanstd(ic)),
                hit=float((ic > 0).mean()))

FIDS = ["slope_tstat_60", "sharpe_mom_60_20", "vol_ratio_5_60", "range_pos_20",
        "xsec_mom_60_skip5", "dd_20", "dd_60", "vol_asym_20", "gainloss_20",
        "ret_vol_corr_20", "hl_range_20", "efficiency_60", "max_ret_20", "skew_20",
        "wti_beta_60", "spx_beta_60", "btc_beta_60", "dxy_beta_60", "vix_beta_60",
        "gap_ret_1", "intraday_ret_1", "mom_20_skip5"]

print(f"Data: {len(idx)} dates, {len(WATCH)} assets, through {END.date()}")
print("=" * 120)
for win_name, win in [("FULL(2020-)", slice(None)), ("POST(2026-08-)", slice("2026-08-01", None)),
                      ("RECENT250", slice(idx[-250], None))]:
    print(f"\n=== {win_name} window: IC / ICIR at h=1,5,10 (hit ratio) ===")
    print(f"{'factor':<22}{'ic1':>8}{'icir1':>8}{'ic5':>8}{'icir5':>8}{'ic10':>9}{'icir10':>9}{'hit10':>7}")
    for fid in FIDS:
        fdf = build(fid)
        if fdf is None: continue
        w = fdf.loc[win]
        s1, s5, s10 = stats(daily_ic(w, 1)), stats(daily_ic(w, 5)), stats(daily_ic(w, 10))
        print(f"{fid:<22}{s1['ic']:>+8.4f}{s1['icir']:>+8.3f}{s5['ic']:>+8.4f}{s5['icir']:>+8.3f}"
              f"{s10['ic']:>+9.4f}{s10['icir']:>+9.3f}{s10['hit']:>7.2f}")

# Turnover & coverage on full sample at h=10 admission horizon
print("\n=== Turnover (mean |rank change|/day) & coverage (full sample) ===")
for fid in FIDS:
    fdf = build(fid)
    if fdf is None: continue
    cov = float(np.isfinite(fdf).sum().sum() / fdf.size)
    rk = fdf.rank(axis=1)
    turn = rk.diff().abs().mean().mean()
    print(f"{fid:<22} coverage={cov:>7.3f} turnover_rank={turn:>7.3f}")
