import json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2027-03-24"
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


def load_ohlc(cutoff):
    out = {}
    for s in TRADABLE:
        df = pd.read_csv(f"{DATA_DIR}/{s}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(cutoff)].set_index("date").sort_index()
        out[s] = df
    return out


t0 = time.time()
ohlc = load_ohlc(VISIBLE)
px = pd.DataFrame({s: ohlc[s]["close"].astype(float) for s in TRADABLE}).dropna(how="all")
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdcny = load_close("USDCNY", VISIBLE, INDEX_DIR)["close"].astype(float)
print(f"vix_last={vix.iloc[-1]:.1f} dxy_last={dxy.iloc[-1]:.1f} usdcny_last={usdcny.iloc[-1]:.2f}", flush=True)


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


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
sign = np.sign(ret)

# ---- candidate factors (path / flow structure + cross-asset regime) ----
C = {}
# 1) lag-1 autocorrelation of daily returns (momentum persistence vs whipsaw)
C["autocorr_20d"] = ret.rolling(20, min_periods=12).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
# 2) up/down vol asymmetry: (up_vol - down_vol)/(up_vol + down_vol), 60d
C["updown_asym_60d"] = (rs(up, 60) - rs(down, 60)) / (rs(up, 60) + rs(down, 60)).replace(0, np.nan)
# 3) beta to US10Y returns (bond-stock interplay; library has CN10Y beta only)
us10y_r = px["US10Y"].pct_change()
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)
# 4) beta to XAU returns (inflation-hedge sensitivity)
xau_r = px["XAU"].pct_change()
C["beta_xau_60d"] = beta_of(ret, xau_r, 60)
# 5) beta to EW-15 cross-asset portfolio (cross-asset "market" beta)
ew = px.mean(axis=1)
ew_r = ew.pct_change()
C["market_beta_60d"] = beta_of(ret, ew_r, 60)
# 6) distance below 60d high (drawdown depth)
hi60 = px.rolling(60, min_periods=mp(60)).max()
C["dist_high_60d"] = px / hi60 - 1.0
# 7) worst single-day return over 60d (tail risk)
C["worst_day_60d"] = ret.rolling(60, min_periods=mp(60)).min()
# 8) overnight share of total price move (flow structure): mean|overnight|/(mean|overnight|+mean|intraday|)
overnight = pd.DataFrame({s: ohlc[s]["open"] / ohlc[s]["close"].shift(1) - 1.0 for s in TRADABLE})
intraday = pd.DataFrame({s: ohlc[s]["close"] / ohlc[s]["open"] - 1.0 for s in TRADABLE})
mo = overnight.abs().rolling(20, min_periods=12).mean()
mi = intraday.abs().rolling(20, min_periods=12).mean()
C["overnight_share_20d"] = mo / (mo + mi).replace(0, np.nan)
# 9) volume trend 20x60 (per-asset volume expansion)
vol_px = pd.DataFrame({s: ohlc[s]["volume"].astype(float) for s in TRADABLE})
C["volume_trend_20x60"] = vol_px.rolling(20, min_periods=10).mean() / vol_px.rolling(60, min_periods=30).mean().replace(0, np.nan)
# 10) rolling skewness 20d (retest of tail asymmetry, cheap)
C["skew_20d"] = ret.rolling(20, min_periods=12).skew()
# 11) corr with US10Y 60d (regime tilt: risk-on = negative stock-bond corr)
C["corr_us10y_60d"] = corr_of(ret, us10y_r, 60)
# 12) beta to DXY change (dollar sensitivity; DXY observation-only macro)
dxy_r = dxy.pct_change()
C["beta_dxy_60d"] = beta_of(ret, dxy_r, 60)
# 13) beta to USDCNY change (China credit / EM risk channel)
cny_r = usdcny.pct_change()
C["beta_usdcny_60d"] = beta_of(ret, cny_r, 60)
# 14) vol-of-return 20d vs 120d (vol term structure; complements vol_of_vol)
C["vol_ratio_20x120"] = rs(ret, 20) / rs(ret, 120).replace(0, np.nan)

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


# library signals (existing persisted factor family)
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret, vixr, 60).mul((vix / vix.shift(20) - 1.0).reindex(px.index), axis=0),
    "down_vol_ratio_20x120": down.rolling(20).std() / down.rolling(120).std(),
    "beta_vix_60d_neg": -beta_of(ret, vixr, 60),
    "beta_cn10y_60d": beta_of(ret, px["CN10Y"].pct_change(), 60),
    "low_vol_20d": -ret.rolling(20).std(),
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
    "2027+": pd.Timestamp("2027-01-01"),
}

print(f"\n{'factor':<26}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  librho  vs  | 2024+ 2025+ 2026+ 2027+ (ic/icir)", flush=True)
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
    print(f"{name:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lr:>6.3f} {str(larg):<16} {'PASS' if ok else '':<4} {rstr}", flush=True)

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
        print(f"  {name:<26} decay={dec} turnover={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)

print("\n=== LIBRARY RE-VALIDATION (..2027-03-24) ===", flush=True)
for fid, lsig in lib.items():
    lsig = lsig.reindex(px.index)
    ic = fast_ic_series(lsig, fwd10)
    m, icir, hit, n = ic_summary(ic)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    print(f"{fid:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {rec}", flush=True)

with open("scripts/miner_2_20270325_screen_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
