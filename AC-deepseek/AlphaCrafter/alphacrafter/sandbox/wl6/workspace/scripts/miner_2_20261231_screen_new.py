"""miner_2 screen (2026-12-31 cycle, visible through 2026-12-30):
fresh factor directions after the 2026-12-17 risk-off block (-4.61pct) where
beta_vix_60d_neg overweighted negative-VIX-beta assets (BTC/WTI/COPPER) and got
hurt. Memory feedback: recheck factor directions; momentum family decayed.

Ideas this cycle (interpretable, price-based + macro-linkage):
  1. skew_60d            : crash-risk skewness premium
  2. down_up_beta_60d    : downside minus upside market beta asymmetry
  3. mkt_beta_60d        : plain beta vs equal-weight cross-asset market
  4. autocorr_10d/60d    : 1-lag return autocorrelation (trend persistence)
  5. ovn_20d / intra_20d : overnight vs intraday cumulative return split
  6. ovn_intra_diff_20d  : overnight minus intraday (where returns come from)
  7. mom60_vol_ratio     : Sharpe-style momentum (60d mom / 60d vol)
  8. dd_depth_60d        : distance from 60d high (resilience)
  9. updown_vol_asym_60d : downside/upside vol asymmetry (60d variant)
 10. us10y_beta_60d      : rates sensitivity (US10Y, unlike inert CN10Y)
 11. corr_mkt_60d        : systematic-ness (correlation with EW market)
 12. rev10_vixhi         : short-term reversal conditioned on VIX>MA60
 13. eff_60d             : trend efficiency |ret60|/sum|ret| (60d)
 14. zscore_20d          : Bollinger z-score (close-MA20)/std20

Gate: |IC|>=0.0070, |ICIR|>=0.0840 at H=10, >=250 IC dates, >=8 valid
instruments/date, 15-instrument tradable universe. Report max_abs_library
correlation vs the 8 root library signals (recomputed from data, mirroring the
deterministic gate inputs); gate threshold 0.5.
"""
import json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2026-12-30"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes = {s: load_close(s, cutoff)["close"].astype(float) for s in TRADABLE}
    px = pd.DataFrame(closes).dropna(how="all")
    return px


t0 = time.time()
px = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
vix_ma60 = vix.rolling(60, min_periods=30).mean()
vix_hi = (vix > vix_ma60).astype(float).reindex(px.index, fill_value=np.nan)
us10y = px["US10Y"]
us10y_r = us10y.pct_change()
print(f"vix_last={vix.iloc[-1]:.1f} vix_ma60_last={vix_ma60.iloc[-1]:.1f} vix_hi_last={vix_hi.iloc[-1]:.1f}", flush=True)
print(f"us10y 20d chg last: {(us10y/us10y.shift(20)-1).dropna().tail(3).round(4).tolist()}", flush=True)
print(f"cn10y nonzero chg days: {int((px['CN10Y'].pct_change().dropna()!=0).sum())}", flush=True)

# equal-weight cross-asset market (demeaned, vol-normalized for beta stability)
mkt = px.mean(axis=1)
mkt_r = mkt.pct_change()
print(f"recent mkt 20d chg: {(mkt/mkt.shift(20)-1).dropna().tail(3).round(4).tolist()}", flush=True)


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def rsum(x, w):
    return x.rolling(w, min_periods=mp(w)).sum()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    var_m = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / var_m


def corr_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w, 2)).corr(mdf)


down = ret.clip(upper=0) * -1
up = ret.clip(lower=0)
mkt_down = mkt_r.clip(upper=0) * -1
mkt_up = mkt_r.clip(lower=0)

C = {}
# 1) crash-risk skewness
C["skew_60d"] = ret.rolling(60, min_periods=40).skew()
C["skew_20d"] = ret.rolling(20, min_periods=12).skew()
# 2) downside minus upside market beta
C["down_beta_60d"] = beta_of(ret, mkt_down, 60)
C["up_beta_60d"] = beta_of(ret, mkt_up, 60)
C["down_up_beta_60d"] = C["down_beta_60d"] - C["up_beta_60d"]
# 3) plain market beta & correlation
C["mkt_beta_60d"] = beta_of(ret, mkt_r, 60)
C["corr_mkt_60d"] = corr_of(ret, mkt_r, 60)
# 4) autocorrelation
C["autocorr_10d"] = ret.rolling(10, min_periods=8).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], raw=True)
C["autocorr_60d"] = ret.rolling(60, min_periods=40).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], raw=True)
# 5) overnight vs intraday
opn = px["open"] if "open" in px else None
if opn is not None:
    ovn = (opn / px.shift(1) - 1.0)
    intr = (px / opn - 1.0)
    C["ovn_20d"] = rsum(ovn, 20)
    C["intra_20d"] = rsum(intr, 20)
    C["ovn_intra_diff_20d"] = C["ovn_20d"] - C["intra_20d"]
# 7) Sharpe-style momentum
C["mom60_vol_ratio"] = (px.shift(5) / px.shift(65) - 1.0) / rs(ret, 60)
C["mom20_vol_ratio"] = (px.shift(5) / px.shift(25) - 1.0) / rs(ret, 20)
# 8) distance from 60d high
C["dd_depth_60d"] = px / px.rolling(60, min_periods=30).max() - 1.0
# 9) up/down vol asymmetry (60d)
C["updown_vol_asym_60d"] = rs(down, 60) / rs(up, 60).replace(0, np.nan)
# 10) rates sensitivity
C["us10y_beta_60d"] = beta_of(ret, us10y_r, 60)
# 12) reversal x VIX regime
C["rev10_vixhi"] = -(px.shift(5) / px.shift(15) - 1.0) * vix_hi
# 13) trend efficiency
C["eff_60d"] = (px / px.shift(60) - 1.0).abs() / rsum(ret.abs(), 60)
# 14) Bollinger z-score
C["zscore_20d"] = (px - rm(px, 20)) / rs(ret, 20).replace(0, np.nan)

C = {k: v for k, v in C.items() if v is not None}
print(f"candidates built: {len(C)} in {time.time()-t0:.1f}s", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    common = factor.index.intersection(fwd.index)
    fr = factor.reindex(common).rank(axis=1, pct=True)
    rr = fwd.reindex(common).rank(axis=1, pct=True)
    mask = fr.isna().values | rr.isna().values
    nvalid = (~mask).sum(axis=1)
    F = np.ma.array(fr.values, mask=mask)
    R = np.ma.array(rr.values, mask=mask)
    Fm = F - F.mean(axis=1, keepdims=True)
    Rm = R - R.mean(axis=1, keepdims=True)
    num = (Fm * Rm).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Rm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic = np.ma.filled(ic, np.nan)
    ic[nvalid < min_valid] = np.nan
    return pd.Series(ic, index=common)


def ic_summary(ic):
    ic = ic.dropna()
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = m / s if s and math.isfinite(s) and s > 0 else 0.0
    hit = float((ic > 0).mean()) if len(ic) else np.nan
    return m, icir, hit, int(len(ic))


# library signals recomputed from data (mirrors gate's pairwise-rho inputs)
ret_l = ret
down_l = ret_l.clip(upper=0) * -1
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret_l.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret_l, vixr, 60) * (vix / vix.shift(20) - 1.0),
    "down_vol_ratio_20x120": rs(down_l, 20) / rs(down_l, 120),
    "beta_vix_60d_neg": -beta_of(ret_l, vixr, 60),
    "beta_cn10y_60d": beta_of(ret_l, px["CN10Y"].pct_change(), 60),
    "low_vol_20d": -rs(ret_l, 20),
}
lib = {k: v.reindex(px.index) for k, v in lib.items()}


def max_lib_rho(fv, lib_sigs):
    best, arg = 0.0, None
    for name, lsig in lib_sigs.items():
        ic = fast_ic_series(fv, lsig, min_valid=5).dropna()
        if len(ic) < 50:
            continue
        r = float(abs(ic.mean()))
        if r > best:
            best, arg = r, name
    return best, arg


fwd10 = px.shift(-H_ADMIT) / px - 1.0
sub_windows = {
    "full": px.index.min(),
    "2024+": pd.Timestamp("2024-01-01"),
    "2025+": pd.Timestamp("2025-01-01"),
    "2026+": pd.Timestamp("2026-01-01"),
}

print(f"\n{'factor':<22}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  librho  vs  | 2024+ 2025+ 2026+ (ic/icir)", flush=True)
results = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lr, larg = max_lib_rho(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_rho": round(lr, 3),
                     "lib_arg": larg, "recent": rec, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v)
    print(f"{name:<22}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lr:>6.3f} {str(larg):<18} {'PASS' if ok else '':<4} {rstr}", flush=True)

print("\n=== DETAIL (gate-passing candidates) ===", flush=True)
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
for name, r in results.items():
    if abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840 and r["n"] >= MIN_IC_DATES:
        f = r["signal"]
        dec = {}
        for h, fr_ in fwd_all.items():
            ic = fast_ic_series(f, fr_)
            mm, _, _, nn = ic_summary(ic)
            dec[str(h)] = round(mm, 4) if nn > 0 else None
        r["decay"] = dec
        ranks = f.rank(axis=1, pct=True)
        r["turnover_10d_rank"] = round(float(ranks.diff(10).abs().mean().mean()), 3)
        valid = f.notna()
        r["coverage_asset_days"] = round(float(valid.sum().sum()) / float(f.shape[0] * f.shape[1]), 3)
        r["coverage_dates_ge8"] = round(float((valid.sum(axis=1) >= 8).mean()), 3)
        print(f"  {name:<22} decay={dec} turnover={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)

print("\n=== LIBRARY RE-VALIDATION (window ..2026-12-30) ===", flush=True)
for fid, lsig in lib.items():
    lsig = lsig.reindex(px.index)
    ic = fast_ic_series(lsig, fwd10)
    m, icir, hit, n = ic_summary(ic)
    ic26 = ic[ic.index >= pd.Timestamp("2026-01-01")]
    mm, ii, _, nn = ic_summary(ic26)
    rec26 = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    print(f"{fid:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  2026+:{rec26}", flush=True)

with open("scripts/miner_2_20261231_screen_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
