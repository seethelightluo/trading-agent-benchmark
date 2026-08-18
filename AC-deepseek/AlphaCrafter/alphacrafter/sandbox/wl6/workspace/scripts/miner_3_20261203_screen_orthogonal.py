"""miner_3 screen (2026-12-03 cycle, visible through 2026-12-02):
orthogonal macro-linkage factors + defensive composites.

Context: last cycle's vol_imb_10d/20d were EVICTED by the deterministic gate
(pairwise |rho|>0.5 vs beta_cn10y_60d, lower quality). The momentum family
(mom_10d, mom_120d, vol_of_vol20x60, vix_beta_cond) was QUARANTINED for missing
signal artifacts. The live ensemble (beta_vix_60d_neg 0.39, beta_cn10y_60d 0.29
[inert: CN10Y flat], vol_of_vol20x60 0.20, low_vol_20d 0.12) just posted its
first positive block (+1.46pct, memory 20261203).

Plans: (1) composite defensive (bvix x rate sensitivity), (2) orthogonal macro
linkages: USDJPY/USDCNY/EURUSD/DXY betas, commodity betas (COPPER/WTI/XAU), BTC
risk appetite, safe-haven spread (XAU/SPX), (3) vol term structure & downside/
upside beta splits.

Gate: |IC|>=0.0070, |ICIR|>=0.0840 at H=10, >=250 IC dates, >=8 valid
instruments/date, 15-instrument tradable universe. Pairwise rho threshold 0.5
vs the FULL library (8 root factors) to survive the deterministic gate.
"""
import sys, json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2026-12-02"
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
    closes, vols = {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    vol = pd.DataFrame(vols)
    return px, vol


t0 = time.time()
px, vol = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
vix_ma60 = vix.rolling(60, min_periods=30).mean()
vix_hi = (vix > vix_ma60).astype(float).reindex(px.index, fill_value=np.nan)
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
dxy_r = dxy.pct_change()
usdjpy = load_close("USDJPY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdjpy_r = usdjpy.pct_change()
usdcny = load_close("USDCNY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdcny_r = usdcny.pct_change()
eurusd = load_close("EURUSD", VISIBLE, INDEX_DIR)["close"].astype(float)
eurusd_r = eurusd.pct_change()
us10y = px["US10Y"]
cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
cn10y_r = cn10y.pct_change()
print(f"vix_last={vix.iloc[-1]:.1f} vix_ma60_last={vix_ma60.iloc[-1]:.1f} vix_hi_last={vix_hi.iloc[-1]:.1f}", flush=True)
print(f"us10y 20d chg recent: {(us10y/us10y.shift(20)-1).dropna().tail(3).round(4).tolist()}", flush=True)
print(f"cn10y 20d chg recent: {(cn10y/cn10y.shift(20)-1).dropna().tail(3).round(4).tolist()}", flush=True)
print(f"cn10y nonzero chg days: {int((cn10y_r.dropna()!=0).sum())} / {int(cn10y_r.dropna().shape[0])}", flush=True)
print(f"usdjpy last: {usdjpy.iloc[-1]:.2f}, 60d chg: {usdjpy.iloc[-1]/usdjpy.iloc[-61]-1:.3f}", flush=True)
print(f"dxy last: {dxy.iloc[-1]:.2f}, 60d chg: {dxy.iloc[-1]/dxy.iloc[-61]-1:.3f}", flush=True)


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
bvix60 = -beta_of(ret, vixr, 60)
bc_cn10y = beta_of(ret, cn10y_r, 60)

C = {}
# --- 1) FX / macro linkage betas (observation-only macro signals) ---
C["usdjpy_beta_60d_neg"] = -beta_of(ret, usdjpy_r, 60)       # yen-funding defensive
C["usdcny_beta_60d"] = beta_of(ret, usdcny_r, 60)            # CNH depreciation linkage
C["eurusd_beta_60d"] = beta_of(ret, eurusd_r, 60)            # EUR risk-on linkage
C["dxy_beta_60d_neg"] = -beta_of(ret, dxy_r, 60)             # USD strength defensive
# --- 2) commodity / real-asset betas ---
C["copper_beta_60d"] = beta_of(ret, px["COPPER"].pct_change(), 60)   # global growth
C["wti_beta_60d"] = beta_of(ret, px["WTI"].pct_change(), 60)         # energy/inflation
C["xau_beta_60d"] = beta_of(ret, px["XAU"].pct_change(), 60)         # gold safe-haven linkage
C["btc_beta_60d"] = beta_of(ret, px["BTC"].pct_change(), 60)         # crypto risk appetite
# --- 3) safe-haven spread (XAU/SPX) linkage ---
hav = (px["XAU"] / px["SPX"]).pct_change()
C["havenspread_beta_60d"] = beta_of(ret, hav, 60)
# --- 4) vol term structure & asymmetry ---
C["vol_term_10x60"] = rs(ret, 10) / rs(ret, 60).replace(0, np.nan)   # rising-vol stress
C["vol_term_20x120"] = rs(ret, 20) / rs(ret, 120).replace(0, np.nan)
C["updown_vol_asym_20d"] = rs(down, 20) / rs(up, 20).replace(0, np.nan)  # downside asymmetry
# --- 5) downside/upside beta split vs SPX ---
spx_ret = px["SPX"].pct_change()
down_days = (spx_ret < 0).astype(float).reindex(px.index).fillna(0)
spx_down = spx_ret.where(down_days > 0)
spx_up = spx_ret.where(down_days == 0)
C["downside_beta_60d"] = beta_of(ret, spx_down, 60)   # beta in down markets
C["upside_beta_60d"] = beta_of(ret, spx_up, 60)       # beta in up markets
# --- 6) composites (reference; expected high librho) ---
C["bvix_x_cn10y"] = bvix60 * np.sign(bc_cn10y.replace(0, np.nan))   # stacked defensive
C["bvix_x_usdjpy"] = bvix60 * np.sign(C["usdjpy_beta_60d_neg"])
# --- 7) other quick probes ---
C["mdd_60d_neg"] = -ret.rolling(60, min_periods=mp(60)).apply(lambda r: float(np.maximum.accumulate(1 + r.fillna(0))).__class__ and 0 or 0, raw=False) if False else None
skew20 = ret.rolling(20, min_periods=mp(20)).skew()
C["skew_20d"] = -skew20                                        # crash-risk aversion
hilo = (px - px.rolling(20, min_periods=mp(20)).min()) / (px.rolling(20, min_periods=mp(20)).max() - px.rolling(20, min_periods=mp(20)).min()).replace(0, np.nan)
C["hilo_pos_20d"] = hilo
C["us10y_beta_cond_rise20"] = beta_of(ret, us10y_r, 60).where((us10y / us10y.shift(20) - 1.0) > 0)
C["gap_ratio_20d"] = (px["open"] / px["close"].shift(1) - 1.0).abs().rolling(20, min_periods=mp(20)).mean() if "open" in px.columns else None
del C["mdd_60d_neg"]
if C["gap_ratio_20d"] is None:
    del C["gap_ratio_20d"]
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
ret_l = px.pct_change()
down_l = ret_l.clip(upper=0) * -1
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret_l.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret_l, vixr, 60).mul((vix / vix.shift(20) - 1.0).reindex(px.index), axis=0),
    "beta_vix_60d_neg": -beta_of(ret_l, vixr, 60),
    "down_vol_ratio_20x120": -(rs(down_l, 20) / rs(down_l, 120).replace(0, np.nan)),
    "low_vol_20d": -rs(ret_l, 20),
    "beta_cn10y_60d": beta_of(ret_l, cn10y_r, 60),
}
upday_l = (ret_l > 0).astype(float)
up_vol_l = (vol * upday_l).rolling(20, min_periods=mp(20)).sum()
dn_vol_l = (vol * (1 - upday_l)).rolling(20, min_periods=mp(20)).sum()
lib["vol_imb_20d"] = (up_vol_l - dn_vol_l) / (up_vol_l + dn_vol_l).replace(0, np.nan)


def max_lib_rho(fv, lib_sigs):
    best, arg = 0.0, None
    for fid, lsig in lib_sigs.items():
        lsig = lsig.reindex(index=fv.index, columns=fv.columns)
        ic = fast_ic_series(fv, lsig, min_valid=MIN_INSTR).dropna()
        if len(ic) and abs(float(ic.mean())) > best:
            best = abs(float(ic.mean()))
            arg = fid
    return best, arg


fwd10 = px.shift(-H_ADMIT) / px - 1.0
sub_windows = {
    "full": px.index.min(),
    "2024+": pd.Timestamp("2024-01-01"),
    "2025+": pd.Timestamp("2025-01-01"),
    "2026+": pd.Timestamp("2026-01-01"),
}

print(f"\n{'factor':<26}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  librho  vs  | recent ic/icir", flush=True)
results = {}
for name, f in C.items():
    if f is None:
        continue
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
    print(f"{name:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lr:>6.3f} {str(larg):<18} {'PASS' if ok else '':<4} {rstr}", flush=True)

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

print("\n=== LIBRARY RE-VALIDATION (window ..2026-12-02) ===", flush=True)
for fid, lsig in lib.items():
    lsig = lsig.reindex(px.index)
    ic = fast_ic_series(lsig, fwd10)
    m, icir, hit, n = ic_summary(ic)
    ic26 = ic[ic.index >= pd.Timestamp("2026-01-01")]
    mm, ii, _, nn = ic_summary(ic26)
    rec26 = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    print(f"{fid:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  2026+:{rec26}", flush=True)

with open("scripts/miner_3_20261203_screen_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
