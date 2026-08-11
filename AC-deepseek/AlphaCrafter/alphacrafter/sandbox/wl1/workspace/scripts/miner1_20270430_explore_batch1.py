"""miner_1 2027-04-30: screen NOVEL factor families for the 15-instrument cross-asset universe.
Data through last completed trading day 2027-04-29 (panel_cache.pkl rebuilt by miner3).

Ensemble feedback (memory): momentum anchor strong (SOX/WTI/BTC/COPPER/ETH), reversal/vol mixed,
defensive XAU/US10Y mixed. Focus this cycle on:
 - trend quality / path efficiency hybrids (momentum that survives chop)
 - risk-managed momentum (vol-scaled, drawdown-gated)
 - multi-horizon momentum alignment
 - downside-vs-upside beta asymmetry (crash sensitivity)
 - macro-conditional signals (VIX regime, DXY trend, yield-curve relative)
 - cross-asset dispersion / relative strength vs. own volatility

Gates (shared): abs(daily paper IC) >= 0.0070 and abs(daily paper ICIR) >= 0.0840.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_VALID = 8

with open("scripts/panel_cache.pkl", "rb") as fh:
    p = pd.read_pickle(fh)
C = p["close"].copy()
O = p["open"].copy()
H = p["high"].copy()
L = p["low"].copy()
V = p["vol"].copy()
M = p["macro"].copy()

# weekday-only calendar, >=8 valid assets, ffill
wk = C.index.dayofweek < 5
C = C[wk]; O = O[wk]; H = H[wk]; L = L[wk]; V = V[wk]
keep = C.notna().sum(axis=1) >= 8
C = C[keep].ffill(); O = O[keep].ffill(); H = H[keep].ffill()
L = L[keep].ffill(); V = V[keep].ffill()
M = M[wk][keep].ffill()

R = C.pct_change()
lnC = np.log(C)
DXY, VIX = M["DXY"], M["VIX"]
USDJPY, EURUSD = M["USDJPY"], M["EURUSD"]

print(f"Data: {len(C)} dates, {C.shape[1]} assets, {C.index.min().date()} -> {C.index.max().date()}")


def build(fid):
    fdf = pd.DataFrame(index=C.index, columns=WATCH, dtype=float)
    for a in WATCH:
        c, o, h, l, v = C[a], O[a], H[a], L[a], V[a]
        r = R[a]
        lc = lnC[a]
        if fid == "eff_mom_60":
            # trend quality: |60d move| / 60d realized path * sign
            fdf[a] = np.sign(lc.diff(60)) * (lc.diff(60)).abs() / r.abs().rolling(60).sum()
        elif fid == "eff_mom_20":
            fdf[a] = np.sign(lc.diff(20)) * (lc.diff(20)).abs() / r.abs().rolling(20).sum()
        elif fid == "vol_scaled_mom_60":
            fdf[a] = (lc.diff(60)) / (r.rolling(60).std() * np.sqrt(60))
        elif fid == "vol_scaled_mom_120":
            fdf[a] = (lc.diff(120)) / (r.rolling(120).std() * np.sqrt(120))
        elif fid == "dd_gated_mom_20":
            dd = c / c.rolling(120).max() - 1.0
            base = lc.diff(20)
            fdf[a] = np.where(dd > -0.08, base, base * 0.5)  # damp momentum when in drawdown
        elif fid == "mom_align_20_60_120":
            m20 = np.sign(lc.diff(20)); m60 = np.sign(lc.diff(60)); m120 = np.sign(lc.diff(120))
            fdf[a] = (m20 + m60 + m120) / 3.0
        elif fid == "hl_pos_60":
            fdf[a] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()) - 0.5
        elif fid == "hl_pos_120":
            fdf[a] = (c - l.rolling(120).min()) / (h.rolling(120).max() - l.rolling(120).min()) - 0.5
        elif fid == "updown_beta_60":
            sr = R["SPX"]
            bu = r.where(sr > 0).rolling(60).cov(sr.where(sr > 0)) / sr.where(sr > 0).rolling(60).var()
            bd = r.where(sr < 0).rolling(60).cov(sr.where(sr < 0)) / sr.where(sr < 0).rolling(60).var()
            fdf[a] = bu - bd
        elif fid == "downside_beta_60":
            sr = R["SPX"]
            fdf[a] = r.where(sr < 0).rolling(60).cov(sr.where(sr < 0)) / sr.where(sr < 0).rolling(60).var()
        elif fid == "vix_cond_mom60":
            vixp = VIX.rank(pct=True)
            fdf[a] = np.sign(lc.diff(60)) * (1.0 - vixp)
        elif fid == "dxy_cond_mom60":
            fdf[a] = np.sign(lc.diff(60)) * np.sign(DXY.diff(60))
        elif fid == "dxy_up_mom60":
            # momentum only counts when USD trending up (risk-off rotation)
            fdf[a] = np.sign(lc.diff(60)) * np.clip(np.sign(DXY.diff(60)), 0, 1)
        elif fid == "yield_slope_rel":
            # US10Y vs CN10Y relative momentum applied cross-sectionally
            fdf[a] = R["US10Y"].rolling(20).mean() - R["CN10Y"].rolling(20).mean()
        elif fid == "us10y_trend_20":
            fdf[a] = R["US10Y"].rolling(20).mean()
        elif fid == "risk_parity_vol_60":
            # inverse-vol weighted trend: trend scaled by 1/vol (defensive momentum)
            fdf[a] = np.sign(lc.diff(60)) * (1.0 / r.rolling(60).std())
        elif fid == "vol_ratio_10_60":
            fdf[a] = r.rolling(10).std() / r.rolling(60).std()
        elif fid == "vol_ratio_20_120":
            fdf[a] = r.rolling(20).std() / r.rolling(120).std()
        elif fid == "max_ret_60":
            fdf[a] = r.rolling(60).max()
        elif fid == "min_ret_60":
            fdf[a] = r.rolling(60).min()
        elif fid == "skew_60":
            fdf[a] = r.rolling(60).skew()
        elif fid == "intraday_mom_60":
            fdf[a] = (c / o - 1.0).rolling(60).mean()
        elif fid == "gap_vol_20":
            fdf[a] = (o / c.shift(1) - 1.0).abs().rolling(20).mean()
        elif fid == "range_std_20":
            fdf[a] = ((h - l) / c).rolling(20).std()
        elif fid == "corr_ret_vol_60":
            vv = V.pct_change()
            fdf[a] = r.rolling(60).corr(vv)
        elif fid == "cross_disp_20":
            # how an asset's 20d ret ranks within cross-section, vol-normalized z
            fdf[a] = lc.diff(20) / (r.rolling(20).std() * np.sqrt(20))
        elif fid == "lnret_std_ratio_20_120":
            fdf[a] = r.rolling(20).std() / r.rolling(120).std()
        elif fid == "wti_equity_rel_20":
            # WTI vs equity index relative strength applied cross-sectionally
            base = R["WTI"].rolling(20).mean() - R["SPX"].rolling(20).mean()
            fdf[a] = base
        elif fid == "copper_equity_rel_20":
            base = R["COPPER"].rolling(20).mean() - R["SPX"].rolling(20).mean()
            fdf[a] = base
        else:
            return None
    return fdf.replace([np.inf, -np.inf], np.nan)


def daily_ic(fdf, fwd):
    fr = C.shift(-fwd) / C - 1.0
    out = []
    for i in range(len(fdf)):
        fv = fdf.iloc[i].values; rv = fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_VALID:
            continue
        if np.all(fv[m] == fv[m][0]):
            continue
        rho = spearmanr(fv[m], rv[m]).correlation
        out.append(rho if np.isfinite(rho) else np.nan)
    return np.array(out)


def stats(ic):
    ok = np.isfinite(ic)
    if ok.sum() < 30:
        return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan, hit=np.nan)
    ic = ic[ok]
    return dict(n=int(ok.sum()), ic=float(np.nanmean(ic)), icir=float(np.nanmean(ic) / np.nanstd(ic)),
                hit=float((ic > 0).mean()))


FIDS = ["eff_mom_60", "eff_mom_20", "vol_scaled_mom_60", "vol_scaled_mom_120", "dd_gated_mom_20",
        "mom_align_20_60_120", "hl_pos_60", "hl_pos_120", "updown_beta_60", "downside_beta_60",
        "vix_cond_mom60", "dxy_cond_mom60", "dxy_up_mom60", "yield_slope_rel", "us10y_trend_20",
        "risk_parity_vol_60", "vol_ratio_10_60", "vol_ratio_20_120", "max_ret_60", "min_ret_60",
        "skew_60", "intraday_mom_60", "gap_vol_20", "range_std_20", "corr_ret_vol_60",
        "cross_disp_20", "lnret_std_ratio_20_120", "wti_equity_rel_20", "copper_equity_rel_20"]

for win_name, win in [("FULL(2020-)", slice(None)), ("POST(2026-08-)", slice("2026-08-01", None)),
                      ("RECENT250", slice(C.index[-250], None))]:
    print(f"\n=== {win_name}: IC / ICIR at h=1,5,10 (hit) ===")
    print(f"{'factor':<24}{'ic1':>8}{'icir1':>8}{'ic5':>8}{'icir5':>8}{'ic10':>9}{'icir10':>9}{'hit10':>7}")
    for fid in FIDS:
        fdf = build(fid)
        if fdf is None or fdf.isnull().all().all():
            continue
        w = fdf.loc[win]
        s1, s5, s10 = stats(daily_ic(w, 1)), stats(daily_ic(w, 5)), stats(daily_ic(w, 10))
        print(f"{fid:<24}{s1['ic']:>+8.4f}{s1['icir']:>+8.3f}{s5['ic']:>+8.4f}{s5['icir']:>+8.3f}"
              f"{s10['ic']:>+9.4f}{s10['icir']:>+9.3f}{s10['hit']:>7.2f}")

print("\n=== Turnover (mean |rank change|/10d) & coverage (full sample) ===")
for fid in FIDS:
    fdf = build(fid)
    if fdf is None or fdf.isnull().all().all():
        continue
    cov = float(np.isfinite(fdf).sum().sum() / fdf.size)
    rk = fdf.rank(axis=1)
    turn = rk.diff(10).abs().mean().mean()
    print(f"{fid:<24} coverage={cov:>7.3f} turnover_10d_rank={turn:>7.3f}")
