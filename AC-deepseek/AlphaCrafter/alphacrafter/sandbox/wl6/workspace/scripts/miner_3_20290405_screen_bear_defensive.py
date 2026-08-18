"""
miner_3 cycle 2029-04-05 screen (data visible through 2029-04-04).
Bear regime (SPX trend ~ -2.0). Focus families:
  - defensive association (XAU/WTI/US10Y betas, defensive-basket correlation)
  - China leadership (relative momentum vs 000300.SH) & SOX tech beta
  - short-term reversal / risk-adjusted reversal (bear contrarian)
  - drawdown freshness / recovery position
  - bear-gated momentum (momentum only when SPX trend negative)
  - VIX-level amplified beta, gain/loss ratio, autocorrelation, trend slope z,
    kurtosis, range expansion, downside-vol change, dxy inverse beta
Admission gates (10d forward, daily cross-sectional rank IC):
  |IC| >= 0.0070, |ICIR| >= 0.0840, n>=250 dates, cov_ge8>=0.5, librho<0.5,
  plus stability in 2027+ and recent (2028-04+) windows.
Also re-validates the 14 persisted library factors for drift.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2029-04-04"
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
wti_r = px["WTI"].pct_change()
xau_r = px["XAU"].pct_change()
spx_r = px["SPX"].pct_change()


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


# ---------------- library signals (14 persisted factors, drift re-validation) ----------------
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
lib["beta_chi_60d"] = beta_of(ret, px["000300.SH"].pct_change(), 60)
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

# ---------------- new candidates (bear / defensive theme) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom5 = px.shift(1) / px.shift(6) - 1.0
mom10 = px.shift(1) / px.shift(11) - 1.0
mom20 = px.shift(1) / px.shift(21) - 1.0
mom60 = px.shift(1) / px.shift(61) - 1.0

# A. short-term reversal / bear contrarian
C["rev_5d"] = -mom5
C["rev_10d_voladj"] = -(mom10 / vol10.replace(0, np.nan))
C["rev_20d_voladj"] = -(mom20 / vol20.replace(0, np.nan))

# B. defensive association
C["def_beta_xau_60d"] = beta_of(ret, xau_r, 60)                     # gold beta
C["def_beta_wti_60d"] = beta_of(ret, wti_r, 60)                     # energy beta
def_basket = (xau_r + us10y_r) / 2.0                                # safe-haven basket return
C["def_corr_basket_60d"] = corr_of(ret, def_basket, 60)             # defensive-basket correlation

# C. China leadership / SOX tech beta
chn_r = px["000300.SH"].pct_change()
C["chn_rel_mom20"] = mom20.sub(chn_r.reindex(mom20.index), axis=0)
C["chn_rel_mom60"] = mom60.sub(chn_r.reindex(mom60.index), axis=0)
C["sox_beta_60d"] = beta_of(ret, px["SOX"].pct_change(), 60)

# D. drawdown freshness / recovery position
rmax20 = px.rolling(20, min_periods=mp(20)).max()
rmax60 = px.rolling(60, min_periods=mp(60)).max()
rmin20 = px.rolling(20, min_periods=mp(20)).min()
dd20 = 1.0 - px / rmax20.replace(0, np.nan)
dd60 = 1.0 - px / rmax60.replace(0, np.nan)
C["dd_freshness_20x60"] = dd20 / dd60.replace(0, np.nan)           # high = fresh decline vs older high
C["recovery_pos_20x60"] = (px - rmin20) / (rmax60 - rmin20).replace(0, np.nan)

# E. bear-gated momentum (momentum only matters in bear regime)
spx_trend60 = (spx_r.rolling(60, min_periods=mp(60)).mean())
bear = (spx_trend60 < 0).astype(float)
C["bear_gated_mom20"] = mom20.mul(pd.DataFrame({c: bear for c in ret.columns}, index=ret.index))
C["bear_gated_mom60"] = mom60.mul(pd.DataFrame({c: bear for c in ret.columns}, index=ret.index))

# F. VIX-level amplified beta (tail amplification)
vix_lev = (vix / rm(vix, 60)).reindex(ret.index)
C["vix_level_beta_60d"] = beta_of(ret, vixr, 60).mul(
    pd.DataFrame({c: vix_lev for c in ret.columns}, index=ret.index))

# G. return-distribution / dynamics
up = ret.clip(lower=0)
upm = up.replace(0, np.nan).rolling(20, min_periods=mp(20)).mean()
dnm = (ret.clip(upper=0) * -1.0).replace(0, np.nan).rolling(20, min_periods=mp(20)).mean()
C["gain_loss_ratio_20d"] = upm / dnm.replace(0, np.nan)            # RSI-style gain/loss ratio
C["kurt_20d"] = ret.rolling(20, min_periods=mp(20)).kurt()
C["autocorr_5d"] = ret.rolling(20, min_periods=mp(20)).apply(
    lambda s: pd.Series(s).autocorr(1) if s.count() >= 10 else np.nan, raw=False)

# H. trend slope z / range dynamics / vol change
def slope_z(x, w=60):
    idx = np.arange(w)
    def f(s):
        y = np.asarray(s, dtype=float)
        if np.isnan(y).any() or np.std(y) == 0:
            return np.nan
        b, a = np.polyfit(idx, np.log(y), 1)
        return b * w
    return x.rolling(w, min_periods=mp(w)).apply(f, raw=True)

C["trend_slope_z60"] = slope_z(px, 60)
C["hl_pos_20d"] = (px - rmin20) / (rmax20 - rmin20).replace(0, np.nan)   # range position
hl = (hi - lo) / px.replace(0, np.nan)
C["range_exp_5x20"] = hl.rolling(5, min_periods=mp(5)).std() / hl.rolling(20, min_periods=mp(20)).std().replace(0, np.nan) - 1.0
C["vol_ratio_5x20"] = vol5_ := vol10.rolling(5, min_periods=mp(5)).mean()  # placeholder replaced below
C["vol_ratio_5x20"] = rs(ret, 5) / vol20.replace(0, np.nan)
dv20 = rs(down, 20); dv60 = rs(down, 60)
C["down_vol_chg_20x60"] = dv20 / dv60.replace(0, np.nan) - 1.0     # fresh downside-vol expansion
C["dxy_beta_neg_60d"] = -beta_of(ret, dxy_r, 60)
sh20 = mom20 / vol20.replace(0, np.nan)
sh60 = mom60 / vol60.replace(0, np.nan)
C["sharpe_chg_20x60"] = sh20 - sh60
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
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01"),
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01"),
               "recent": pd.Timestamp("2028-04-01")}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'decay10/20':>11s}", flush=True)
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
    d10 = dec.get(10, (None, None))[0]; d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None)); srec = rec.get("recent", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}  "
          f"{d10 if d10 is not None else float('nan'):>6.4f}/{d20 if d20 is not None else float('nan'):>6.4f}", flush=True)

print("\n--- DRIFT CHECK (library factors) ---", flush=True)
drift_flags = []
for name in lib:
    r = results[name]
    full_ok = abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH
    rec = r["sub"].get("recent"); s27 = r["sub"].get("2027+")
    flag = ""
    if not full_ok:
        flag += "FULL_FAIL "
    if rec is not None and (abs(rec[0]) < IC_TH or abs(rec[1]) < ICIR_TH):
        flag += "RECENT_WEAK "
    if s27 is not None and (abs(s27[0]) < IC_TH or abs(s27[1]) < ICIR_TH):
        flag += "2027_WEAK "
    if flag:
        drift_flags.append((name, flag))
    print(f"{name:<26} full={r['ic']:+.4f}/{r['icir']:+.3f} 2027+={s27} recent={rec} {flag}", flush=True)

passers = []
for name in C:
    r = results[name]
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < CORR_TH)
    rec = r["sub"].get("recent"); s27 = r["sub"].get("2027+")
    stable = True
    if rec is not None:
        stable = stable and (abs(rec[0]) >= IC_TH * 0.7 and abs(rec[1]) >= ICIR_TH * 0.7)
    if s27 is not None:
        stable = stable and (abs(s27[0]) >= IC_TH * 0.5 and abs(s27[1]) >= ICIR_TH * 0.5)
    ok = ok and stable
    if ok:
        passers.append(name)
        print(f"*** PASSER: {name} IC={r['ic']:.4f} ICIR={r['icir']:.3f} librho={r['librho']:.3f} "
              f"n={r['n']} turn={r['turn']:.3f} cov_ge8={r['cov_ge8']:.2f} 2027={s27} recent={rec}", flush=True)
    else:
        print(f"    fail: {name} IC={r['ic']:.4f} ICIR={r['icir']:.3f} librho={r['librho']:.3f} "
              f"n={r['n']} cov_ge8={r['cov_ge8']:.2f} 2027={s27} recent={rec}", flush=True)

print(f"\npassers={passers} drift_flags={[f[0] for f in drift_flags]}", flush=True)
with open("scripts/miner_3_20290405_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
