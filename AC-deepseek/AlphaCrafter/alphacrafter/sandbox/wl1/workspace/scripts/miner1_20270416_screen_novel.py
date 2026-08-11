"""miner_1 2027-04-16: screen NOVEL factor families for the 15-instrument cross-asset universe.
Data through last completed trading day 2027-04-15.

Focus this cycle (given ensemble feedback: momentum anchor strong, reversal/vol dragged):
 - multi-horizon momentum alignment (trend consistency)
 - long-range position/mean-reversion with vol guard
 - downside-vs-upside beta asymmetry
 - path efficiency / trend quality hybrids
 - macro-regime conditional (VIX/DXY) variants
 - yield-curve relative signals

Gates (shared): abs(daily paper IC) >= 0.0070 and abs(daily paper ICIR) >= 0.0840.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_VALID = 8
DAYS = 2100
END = pd.Timestamp("2027-04-15")

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

def load_macro(name):
    m = pd.read_csv(f"../persistent/index_data/{name}.csv")
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] <= END].set_index("date")
    col = "close" if "close" in m.columns else m.columns[1]
    return m[col].astype(float).reindex(C.index).ffill()

DXY = load_macro("DXY"); VIX = load_macro("VIX")
USDJPY = load_macro("USDJPY"); EURUSD = load_macro("EURUSD")

def build(fid):
    fdf = pd.DataFrame(index=C.index, columns=WATCH, dtype=float)
    for a in WATCH:
        c, o, h, l, v = C[a], O[a], H[a], L[a], V[a]
        r = R[a]
        lc = lnC[a]
        if fid == "mom_align_10_20_60":
            # sign agreement across 10/20/60d momentum -> trend consistency
            m10 = np.sign(lc.diff(10)); m20 = np.sign(lc.diff(20)); m60 = np.sign(lc.diff(60))
            fdf[a] = (m10 + m20 + m60) / 3.0
        elif fid == "mom_align_20_60_120":
            m20 = np.sign(lc.diff(20)); m60 = np.sign(lc.diff(60)); m120 = np.sign(lc.diff(120))
            fdf[a] = (m20 + m60 + m120) / 3.0
        elif fid == "hl_pos_60":
            fdf[a] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()) - 0.5
        elif fid == "hl_pos_120":
            fdf[a] = (c - l.rolling(120).min()) / (h.rolling(120).max() - l.rolling(120).min()) - 0.5
        elif fid == "dd_120":
            fdf[a] = c / c.rolling(120).max() - 1.0
        elif fid == "vol_scaled_mom_20":
            fdf[a] = (lc.diff(20)) / (r.rolling(20).std() * np.sqrt(20))
        elif fid == "updown_beta_60":
            sr = R["SPX"]
            cov_up = r.where(sr > 0).rolling(60).cov(sr.where(sr > 0))
            var_up = sr.where(sr > 0).rolling(60).var()
            cov_dn = r.where(sr < 0).rolling(60).cov(sr.where(sr < 0))
            var_dn = sr.where(sr < 0).rolling(60).var()
            bu = cov_up / var_up; bd = cov_dn / var_dn
            fdf[a] = bu - bd
        elif fid == "skew_60":
            fdf[a] = r.rolling(60).skew()
        elif fid == "kurt_60":
            fdf[a] = r.rolling(60).kurt()
        elif fid == "efficiency_20":
            fdf[a] = (lc.diff(20)).abs() / r.abs().rolling(20).sum()
        elif fid == "efficiency_60":
            fdf[a] = (lc.diff(60)).abs() / r.abs().rolling(60).sum()
        elif fid == "vol_ratio_10_60":
            fdf[a] = r.rolling(10).std() / r.rolling(60).std()
        elif fid == "vol_ratio_20_120":
            fdf[a] = r.rolling(20).std() / r.rolling(120).std()
        elif fid == "vix_cond_mom60":
            # momentum weighted by low-vol regime (inverse VIX percentile)
            vixp = VIX.rank(pct=True)
            base = np.sign(lc.diff(60))
            fdf[a] = base * (1.0 - vixp)
        elif fid == "dxy_cond_mom60":
            dxy_mom = np.sign(DXY.diff(60))
            base = np.sign(lc.diff(60))
            fdf[a] = base * dxy_mom  # asset trend aligned with dollar trend
        elif fid == "gap_vol_20":
            fdf[a] = (o / c.shift(1) - 1.0).abs().rolling(20).mean()
        elif fid == "intraday_mom_60":
            fdf[a] = (c / o - 1.0).rolling(60).mean()
        elif fid == "shadow_ratio_20":
            fdf[a] = ((h - np.maximum(c, o)) / (h - l)).rolling(20).mean()
        elif fid == "lower_shadow_20":
            fdf[a] = ((np.minimum(c, o) - l) / (h - l)).rolling(20).mean()
        elif fid == "max_ret_60":
            fdf[a] = r.rolling(60).max()
        elif fid == "min_ret_60":
            fdf[a] = r.rolling(60).min()
        elif fid == "us10y_mom_20":
            fdf[a] = R["US10Y"].rolling(20).mean()  # cross-asset yield momentum applied to all
        elif fid == "wti_cond_equity_20":
            # commodities vs equities relative momentum (cross-sectional rank diff)
            pass
        elif fid == "corr_ret_vol_60":
            vv = V.pct_change()
            fdf[a] = r.rolling(60).corr(vv)
        elif fid == "range_std_20":
            fdf[a] = ((h - l) / c).rolling(20).std()
        else:
            return None
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

FIDS = ["mom_align_10_20_60", "mom_align_20_60_120", "hl_pos_60", "hl_pos_120", "dd_120",
        "vol_scaled_mom_20", "updown_beta_60", "skew_60", "kurt_60", "efficiency_20",
        "efficiency_60", "vol_ratio_10_60", "vol_ratio_20_120", "vix_cond_mom60",
        "dxy_cond_mom60", "gap_vol_20", "intraday_mom_60", "shadow_ratio_20",
        "lower_shadow_20", "max_ret_60", "min_ret_60", "corr_ret_vol_60", "range_std_20"]

print(f"Data: {len(idx)} dates, {len(WATCH)} assets, through {END.date()}")
print("=" * 130)
for win_name, win in [("FULL(2020-)", slice(None)), ("POST(2026-08-)", slice("2026-08-01", None)),
                      ("RECENT250", slice(idx[-250], None))]:
    print(f"\n=== {win_name} window: IC / ICIR at h=1,5,10 (hit ratio) ===")
    print(f"{'factor':<24}{'ic1':>8}{'icir1':>8}{'ic5':>8}{'icir5':>8}{'ic10':>9}{'icir10':>9}{'hit10':>7}")
    for fid in FIDS:
        fdf = build(fid)
        if fdf is None or fdf.isnull().all().all(): continue
        w = fdf.loc[win]
        s1, s5, s10 = stats(daily_ic(w, 1)), stats(daily_ic(w, 5)), stats(daily_ic(w, 10))
        print(f"{fid:<24}{s1['ic']:>+8.4f}{s1['icir']:>+8.3f}{s5['ic']:>+8.4f}{s5['icir']:>+8.3f}"
              f"{s10['ic']:>+9.4f}{s10['icir']:>+9.3f}{s10['hit']:>7.2f}")

print("\n=== Turnover (mean |rank change|/day) & coverage (full sample) ===")
for fid in FIDS:
    fdf = build(fid)
    if fdf is None or fdf.isnull().all().all(): continue
    cov = float(np.isfinite(fdf).sum().sum() / fdf.size)
    rk = fdf.rank(axis=1)
    turn = rk.diff().abs().mean().mean()
    print(f"{fid:<24} coverage={cov:>7.3f} turnover_rank={turn:>7.3f}")
