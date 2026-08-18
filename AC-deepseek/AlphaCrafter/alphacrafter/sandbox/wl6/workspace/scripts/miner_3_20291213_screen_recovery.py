"""
miner_3 cycle 2029-12-13 screen (data visible through 2029-12-12).
Regime: high-vol risk-off transitioning to recovery — SPX +7.2%/30d bounce, VIX 52->43,
SX5E +11.8%/20d & BTC +10.9%/20d leaders, ETH -18.9%/20d & SOX -11.7%/20d laggards,
DXY -3.4%/20d weakness, US10Y 7.57 (-2.2%/20d), COPPER +3.7%/20d recovering.
Focus families:
  - short-horizon reversal / mean reversion after sharp dispersion
  - VIX-decline & vol-normalization beneficiaries
  - risk-on beta (SPX/NDX beta) in recovery
  - USD-weakness beneficiaries (DXY/USDJPY beta)
  - rate/yield sensitivity (US10Y beta)
  - momentum variants at 20d/30d/60d (library only has 10d/120d)
  - trend quality (Sharpe ratios)
  - cross-asset association (crypto beta, cyclical vs defensive basket)
  - volume/liquidity participation
Also re-validates the 15 persisted library factors for drift.
Admission gates (10d forward, daily cross-sectional rank IC):
  |IC| >= 0.0070, |ICIR| >= 0.0840, n>=250 dates, cov_ge8>=0.5, librho<0.5,
  plus stability in 2027+ and recent (2029-01-01+) windows.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2029-12-12"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH, CORR_TH = 0.0070, 0.0840, 0.5
WARM_END = pd.Timestamp("2026-07-15")
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

t0 = time.time()


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
        highs[s] = df["high"].astype(float) if "high" in df else pd.Series(np.nan, index=df.index)
        lows[s] = df["low"].astype(float) if "low" in df else pd.Series(np.nan, index=df.index)
        opens[s] = df["open"].astype(float) if "open" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    return px, pd.DataFrame(vols), pd.DataFrame(highs), pd.DataFrame(lows), pd.DataFrame(opens)


px, vol, hi, lo, op = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
vix = obs["VIX"]; vixr = vix.pct_change()
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change(); cn10y_r = cn10y.pct_change()
dxy = obs["DXY"]; dxy_r = dxy.pct_change()
usdjpy = obs["USDJPY"]; usdjpy_r = usdjpy.pct_change()
xau_r = px["XAU"].pct_change(); wti_r = px["WTI"].pct_change(); spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change(); chi_r = px["000300.SH"].pct_change(); cop_r = px["COPPER"].pct_change()
ndx_r = px["NDX"].pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    cov = a.rolling(w, min_periods=mp(w, 2)).cov(mdf)
    var = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return cov / var


def corr_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w)).corr(mdf)


# ---------------- library signals (15 persisted factors, drift re-validation) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
vix_move20 = (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, cn10y_r, 60)
lib["beta_chi_60d"] = beta_of(ret, chi_r, 60)
lib["corr_us10y_60d"] = corr_of(ret, us10y_r, 60)
vov = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vol_of_vol_chg_20d"] = vov / vov.shift(20) - 1.0
xau_copper_ratio = px["XAU"] / px["COPPER"]
lib["xau_copper_cond_20d"] = beta_of(ret, xau_copper_ratio.pct_change(), 60).mul(
    xau_copper_ratio.pct_change(20).reindex(ret.index), axis=0)
vol20_all = rs(ret, 20)
lib["vol_beta_spx_60d"] = beta_of(vol20_all, vol20_all["SPX"], 60)
lib["sign_ewma_60d"] = np.sign(px.ewm(span=60, adjust=False).mean().diff())
sk20 = ret.rolling(20, min_periods=mp(20)).skew()
lib["skew_20d_neg"] = -sk20
print(f"library signals rebuilt: {len(lib)} ({time.time()-t0:.1f}s)", flush=True)

# ---------------- new candidates (recovery / transition regime theme) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom5 = px.shift(1) / px.shift(6) - 1.0
mom10 = px.shift(1) / px.shift(11) - 1.0
mom20 = px.shift(1) / px.shift(21) - 1.0
mom60 = px.shift(1) / px.shift(61) - 1.0

# A. short-horizon reversal / mean reversion
C["rev5_20d"] = -mom5
C["rev10_20d"] = -mom10
up = ret.clip(lower=0); dwn = (ret.clip(upper=0) * -1.0)
gain = up.rolling(14, min_periods=mp(14)).mean()
loss = dwn.rolling(14, min_periods=mp(14)).mean()
rsi14 = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
C["rsi14_neg"] = -rsi14
rmin20 = px.rolling(20, min_periods=mp(20)).min()
rmin60 = px.rolling(60, min_periods=mp(60)).min()
C["dist_trough_20d"] = px / rmin20.replace(0, np.nan) - 1.0
C["dist_trough_60d"] = px / rmin60.replace(0, np.nan) - 1.0

# B. VIX-decline / vol-normalization beneficiaries
C["vol_contr_10d"] = -(vol10 / vol10.shift(10) - 1.0)
C["vol_ratio_10x60_neg"] = -(vol10 / vol60.replace(0, np.nan) - 1.0)
C["vix_beta_neg_20d"] = -beta_of(ret, vixr, 20)
vix_lev = (vix / rm(vix, 60)).reindex(ret.index)
highvix = (vix > rm(vix, 120)).astype(float).reindex(ret.index)
C["vix_gated_beta_neg_60d"] = lib["beta_vix_60d_neg"].mul(
    pd.DataFrame({c: highvix for c in ret.columns}, index=ret.index))

# C. risk-on beta in recovery
C["spx_beta_20d"] = beta_of(ret, spx_r, 20)
C["spx_beta_60d"] = beta_of(ret, spx_r, 60)
C["ndx_beta_60d"] = beta_of(ret, ndx_r, 60)

# D. USD-weakness beneficiaries
C["dxy_beta_neg_60d"] = -beta_of(ret, dxy_r, 60)
C["usdjpy_beta_neg_60d"] = -beta_of(ret, usdjpy_r, 60)

# E. rate/yield sensitivity
C["us10y_beta_60d"] = beta_of(ret, us10y_r, 60)

# F. momentum variants at intermediate horizons + trend quality
C["mom_20d_skip5"] = px.shift(5) / px.shift(25) - 1.0
C["mom_30d_skip5"] = px.shift(5) / px.shift(35) - 1.0
C["mom_60d_skip5"] = px.shift(5) / px.shift(65) - 1.0
C["sharpe20"] = mom20 / vol20.replace(0, np.nan)
C["sharpe60"] = mom60 / vol60.replace(0, np.nan)

# G. cross-asset association
C["btc_beta_60d"] = beta_of(ret, btc_r, 60)
C["gold_beta_60d"] = beta_of(ret, xau_r, 60)
cyc_basket = (cop_r + wti_r) / 2.0
def_basket = (xau_r + us10y_r) / 2.0
C["cyc_corr_60d"] = corr_of(ret, cyc_basket, 60)
C["def_corr_60d"] = corr_of(ret, def_basket, 60)

# H. volume / liquidity participation
volma5 = vol.rolling(5, min_periods=3).mean()
volma20 = vol.rolling(20, min_periods=8).mean()
C["vol_trend_10d"] = volma5 / volma20.replace(0, np.nan) - 1.0
C["amihud_neg_10d"] = -(ret.abs() / vol.replace(0, np.nan)).rolling(10, min_periods=mp(10)).mean()
print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    common = factor.index.intersection(fwd.index)
    fr = factor.reindex(common)
    rr = fwd.reindex(common)
    ccols = fr.columns.intersection(rr.columns)
    fr = fr[ccols].rank(axis=1, pct=True)
    rr = rr[ccols].rank(axis=1, pct=True)
    mask = fr.isna().values | rr.isna().values
    fr = fr.where(~mask); rr = rr.where(~mask)
    nvalid = fr.notna().sum(axis=1)
    fr = fr[nvalid >= min_valid]
    rr = rr[nvalid >= min_valid]
    if len(fr) == 0:
        return pd.Series(dtype=float)
    return fr.corrwith(rr, axis=1)


def ic_summary(ic):
    ic = ic.dropna()
    if len(ic) < 30:
        return np.nan, np.nan, np.nan, len(ic)
    m = float(ic.mean()); s = float(ic.std(ddof=1))
    icir = m / s if s > 0 else 0.0
    return m, icir, float((ic > 0).mean()), len(ic)


def turnover_10d(f):
    rk = f.rank(axis=1, pct=True)
    return float(rk.diff(10).abs().mean(axis=1).mean())


def max_lib_corr(f, libs):
    best = 0.0; det = {}
    fs = f.stack().rename("c")
    for k, sig in libs.items():
        both = pd.concat([fs, sig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rho = float(both["c"].rank().corr(both["l"].rank()))
        det[k] = round(rho, 3)
        best = max(best, abs(rho))
    return best, det


def coverage_stats(f):
    valid = f.notna()
    return float(valid.values.mean()), float((valid.sum(axis=1) >= 8).mean())


fwd10 = px.shift(-H_ADMIT) / px - 1.0
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
sub_windows = {"full": None, "warm": WARM_END, "2024+": pd.Timestamp("2024-01-01"),
               "2026+": pd.Timestamp("2026-01-01"), "online": pd.Timestamp("2026-07-16"),
               "2027+": pd.Timestamp("2027-01-01"), "recent": pd.Timestamp("2029-01-01")}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'cov':>6s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    turn = turnover_10d(f)
    cov_ad, cov_ge8 = coverage_stats(f)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic if wname == "full" else ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    dec = {}
    for h, fh in fwd_all.items():
        ich = fast_ic_series(f, fh)
        mm, ii, _, _ = ic_summary(ich)
        dec[h] = (round(mm, 4), round(ii, 4)) if np.isfinite(mm) else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "librho": lc,
                     "turn": turn, "sub": rec, "decay": dec, "det": det,
                     "cov_ad": cov_ad, "cov_ge8": cov_ge8}
    s27 = rec.get("2027+", (None, None)); srec = rec.get("recent", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.2f}  {turn:>6.3f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}  {cov_ge8:>6.2f}", flush=True)

print("\n--- candidates passing admission gate (|IC|>=%.4f, |ICIR|>=%.3f, n>=%d, cov_ge8>=0.5, librho<%.1f) ---" % (IC_TH, ICIR_TH, MIN_IC_DATES, CORR_TH), flush=True)
for name, r in results.items():
    if name in lib:
        continue
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < CORR_TH)
    if ok:
        s27 = r["sub"].get("2027+", (0, 0)); srec = r["sub"].get("recent", (0, 0))
        stab = (s27[0] is not None and abs(s27[0]) >= IC_TH * 0.6) and (srec[0] is not None and abs(srec[0]) >= IC_TH * 0.6)
        print(f"  PASS {name:<26} ic={r['ic']:.4f} icir={r['icir']:.3f} librho={r['librho']:.3f} "
              f"turn={r['turn']:.3f} 2027+({s27[0]},{s27[1]}) recent({srec[0]},{srec[1]}) stab={stab}", flush=True)

print("\n--- library drift summary (effective if |IC|>=th and |ICIR|>=th and recent still aligned) ---", flush=True)
for name, r in results.items():
    if name not in lib:
        continue
    s27 = r["sub"].get("2027+", (0, 0)); srec = r["sub"].get("recent", (0, 0))
    flag = ""
    if abs(r["ic"]) < IC_TH or abs(r["icir"]) < ICIR_TH:
        flag = "DRIFT_LOW"
    elif srec[0] is not None and srec[0] * np.sign(r["ic"]) < 0:
        flag = "DRIFT_SIGN_FLIP"
    print(f"  {name:<26} ic={r['ic']:.4f} icir={r['icir']:.3f} hit={r['hit']:.2f} n={r['n']} "
          f"2027+({s27[0] if s27 else None},{s27[1] if s27 else None}) recent({srec[0] if srec else None},{srec[1] if srec else None}) {flag}", flush=True)

with open("scripts/miner_3_20291213_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s; results saved", flush=True)
