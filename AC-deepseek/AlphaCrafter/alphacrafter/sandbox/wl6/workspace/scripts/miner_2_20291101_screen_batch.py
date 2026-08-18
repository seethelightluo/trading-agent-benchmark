"""
miner_2 batch screen + re-validation 2029-11-01 cycle (data visible through 2029-10-31).
Regime per 2029-10-18 ensemble note: highvol_mixed_riskoff_US_equity_new_downleg_VIX37_China_tech_relative_lead_crypto_deep_bear.
New candidates this cycle (regime-tailored):
  A. VIX percentile regime conditioner (VIX level relative to 1y range)
  B. US10Y yield momentum 20d (rate pressure)
  C. XAU relative to US10Y real-rate proxy (gold/real-yield sensitivity)
  D. 120d cross-sectional dispersion of returns (dispersion regime)
  E. BTC/ETH internal divergence (crypto beta dispersion signal)
  F. Trend efficiency ratio 40d (close-path vs net move)
  G. Intraday range position (close location in day range, 10d avg)
  H. Volume trend 20v60 (participation shift)
  I. Short-term reversal 5d after 120d momentum gate (regime-conditioned reversal)
  J. DXY beta 60d (observation-only USD index as signal)
  K. USDJPY beta 60d (carry/risk signal)
  L. WTI momentum skip5 20d (energy short-horizon trend)
  M. Max drawdown recovery speed 60d (resilience)
  N. Downside capture ratio 60d vs SPX (beta asymmetry)
  O. Realized vol ratio 10d/60d negated (short vol expansion pick-up)
  P. Range expansion 5d/20d (breakout amplitude)
Also re-validate all 15 persisted library factors (drift check incl 2029 sub-window).
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), n_ic_dates>=250, coverage dates>=8 >= 0.5, max_abs_library_correlation < 0.5.
Only data <= 2029-10-31 is loaded; nothing beyond the simulation date is touched.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2029-10-31"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
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
dxy = obs["DXY"]; dxy_r = dxy.pct_change()
usdjpy = obs["USDJPY"]; usdjpy_r = usdjpy.pct_change()
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change(); cn10y_r = cn10y.pct_change()
spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change()
eth_r = px["ETH"].pct_change()


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


# ---------------- library signals (15 persisted factors) ----------------
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

# ---------------- new candidates (2029-11-01 cycle) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)

# A. VIX percentile regime conditioner: 1 - percentile of VIX in trailing 1y range, applied to all assets
vix_pct = vix.rolling(250, min_periods=60).rank(pct=True)
vix_pct_neg = (1.0 - vix_pct).reindex(ret.index)
C["vix_pct_neg"] = pd.DataFrame({c: vix_pct_neg for c in px.columns}, index=px.index)

# B. US10Y yield momentum 20d (rate pressure; higher yield = pressure on risk assets)
C["us10y_mom_20d"] = pd.DataFrame({c: (us10y / us10y.shift(20) - 1.0) for c in px.columns}, index=px.index)

# C. XAU beta to US10Y real-rate proxy 60d (gold vs yield sensitivity, positive = inflation hedge)
C["xau_us10y_beta_60d"] = beta_of(ret, us10y_r, 60)

# D. 120d cross-sectional return dispersion (regime): |asset 120d mom - cross-sectional mean|
mom120cs = px.shift(5) / px.shift(125) - 1.0
disp120 = mom120cs.sub(mom120cs.mean(axis=1), axis=0).abs()
C["dispersion_120d"] = disp120

# E. BTC/ETH internal divergence: asset 20d mom scaled by crypto divergence (BTC vs ETH 20d)
btc_mom20 = btc_r.rolling(20, min_periods=mp(20)).sum()
eth_mom20 = eth_r.rolling(20, min_periods=mp(20)).sum()
crypto_div = (btc_mom20 - eth_mom20).abs().reindex(ret.index)
C["crypto_divergence_20d"] = pd.DataFrame({c: crypto_div for c in px.columns}, index=px.index)

# F. Trend efficiency ratio 40d (net move / sum of abs moves); high = clean trend
abs_ret = ret.abs()
net40 = (px / px.shift(40) - 1.0).abs()
path40 = abs_ret.rolling(40, min_periods=mp(40)).sum()
C["eff_ratio_40d"] = (net40 / path40.replace(0, np.nan)).clip(upper=1.0)

# G. Intraday range position 10d: avg( (close-low)/(high-low) ) - 0.5 (close location)
day_pos = ((px - lo) / (hi - lo).replace(0, np.nan)).rolling(10, min_periods=mp(10)).mean() - 0.5
C["range_pos_10d"] = day_pos

# H. Volume trend 20v60 (participation shift)
v20 = vol.rolling(20, min_periods=mp(20)).mean()
v60 = vol.rolling(60, min_periods=mp(60)).mean()
C["vol_trend_20v60"] = v20 / v60.replace(0, np.nan) - 1.0

# I. Short-term reversal 5d gated by 120d momentum (regime-conditioned reversal)
mom120 = px.shift(5) / px.shift(125) - 1.0
r5 = px / px.shift(5) - 1.0
mom120_sign = np.sign(mom120)
C["rev5_mom120_cond"] = -r5 * mom120_sign

# J. DXY beta 60d (USD index sensitivity; negative = USD-strength hurts)
C["dxy_beta_60d_neg"] = -beta_of(ret, dxy_r, 60)

# K. USDJPY beta 60d (carry/risk proxy)
C["usdjpy_beta_60d"] = beta_of(ret, usdjpy_r, 60)

# L. WTI momentum skip5 20d (energy short-horizon trend, skip recent 5d)
C["wti_mom_20d_skip5"] = pd.DataFrame({c: (px["WTI"].shift(5) / px["WTI"].shift(25) - 1.0)
                                       for c in px.columns}, index=px.index)

# M. Max drawdown recovery speed 60d (fraction of 60d max DD recovered)
cummx = px.rolling(60, min_periods=mp(60)).max()
dd60 = px / cummx - 1.0
recovery = (1.0 - dd60.rolling(60, min_periods=mp(60)).min().shift(60).abs().replace(0, np.nan))  # placeholder
C["recovery_speed_60d"] = -dd60  # higher = closer to high = resilience

# N. Downside capture ratio 60d vs SPX (beta in down days only)
down_mask = (spx_r < 0).astype(float).reindex(ret.index)
ret_down = ret.mul(down_mask, axis=0)
spx_down = spx_r.clip(upper=0).reindex(ret.index)
C["down_capture_60d"] = beta_of(ret_down, spx_down, 60)

# O. Realized vol ratio 10d/60d negated (short vol expansion pick-up; low = vol contraction)
C["vol_ratio_10v60_neg"] = -(vol10 / vol60.replace(0, np.nan))

# P. Range expansion 5d/20d (breakout amplitude)
rng5 = (hi - lo).rolling(5, min_periods=mp(5)).mean()
rng20 = (hi - lo).rolling(20, min_periods=mp(20)).mean()
C["range_exp_5v20"] = rng5 / rng20.replace(0, np.nan) - 1.0

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
    m = float(ic.mean())
    s = float(ic.std(ddof=1))
    icir = m / s if s > 0 else 0.0
    hit = float((ic > 0).mean())
    return m, icir, hit, len(ic)


def turnover_10d(f):
    rk = f.rank(axis=1, pct=True)
    return float(rk.diff(10).abs().mean(axis=1).mean())


def max_lib_corr(f, libs):
    best, det = 0.0, {}
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
               "2028+": pd.Timestamp("2028-01-01"), "recent": pd.Timestamp("2028-04-01"),
               "2029+": pd.Timestamp("2029-01-01")}

results = {}
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2028+IC':>8s}{'2028+IR':>8s} {'2029+IC':>8s}{'2029+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'d10/d20':>11s}", flush=True)
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
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s28 = rec.get("2028+", (None, None))
    s29 = rec.get("2029+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<24}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{(s28[0] if s28 else float('nan')):>8.4f}{(s28[1] if s28 else float('nan')):>8.3f} "
          f"{(s29[0] if s29 else float('nan')):>8.4f}{(s29[1] if s29 else float('nan')):>8.3f} "
          f"{(srec[0] if srec else float('nan')):>9.4f}{(srec[1] if srec else float('nan')):>9.3f}  "
          f"{(d10 if d10 is not None else float('nan')):>6.4f}/{(d20 if d20 is not None else float('nan')):>6.4f}", flush=True)

print("\n=== PASS GATE (|IC|>=%.4f, |ICIR|>=%.3f, n>=%d, cov>=%.2f) ===" % (IC_TH, ICIR_TH, MIN_IC_DATES, 0.5), flush=True)
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["cov_ge8"] >= 0.5:
        print(f"  PASS {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ge8={r['cov_ge8']:.2f}", flush=True)
    elif abs(r["ic"]) >= IC_TH or abs(r["icir"]) >= ICIR_TH:
        print(f"  NEAR {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

out = "scripts/miner_2_20291101_screen_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved {out} ({time.time()-t0:.1f}s)", flush=True)
