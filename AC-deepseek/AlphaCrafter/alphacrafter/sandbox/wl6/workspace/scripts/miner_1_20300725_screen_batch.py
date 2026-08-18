"""
miner_1 batch screen + re-validation 2030-07-25 cycle (data visible through 2030-07-24).
Regime snapshot: VIX 46.7 RISING (41.6 ten days ago) = high-vol risk-off pressure;
SPX -6.7%/10d & -10.1%/60d weak; 000300.SH +14.8%/60d China strong; ETH +115%/60d but
-15.6%/10d sharp correction; WTI +11.8%/10d rebound; BTC +5.9%/10d; COPPER +8.6%/10d firm;
XAU -3.6%/10d soft. Ensemble 7f kept by trader (beta_vix_60d_neg 0.28 anchor).

New candidates this cycle (regime-tailored, DISTINCT from miner_2's 2030-07-11 batch):
  A. vix_level_gate_def    : -60d VIX beta gated by VIX level > 40 (defensive regime switch)
  B. dxy_beta_60d_neg      : negative beta to DXY (dollar-hedged exposure)
  C. us10y_beta_60d_neg    : negative beta to US10Y changes (rate-hedged)
  D. eth_btc_ratio_mom20   : ETH/BTC relative momentum (crypto internal rotation)
  E. wti_beta_60d          : beta to WTI (energy beta diversity)
  F. copper_beta_60d       : beta to COPPER (cyclical beta)
  G. autocorr_5d           : 5-day lag autocorrelation of returns (trend persistence)
  H. range_pos_20d         : close position within 20d high-low range (near-high strength)
  I. dist_ma200            : distance from 200d SMA (long-term trend)
  J. updown_capture_60d    : upside/downside capture ratio (return asymmetry)
  K. vol_ratio_5x60        : 5d/60d realized vol ratio (term-structure proxy)
  L. mom_blend_5_20_60     : blended multi-horizon momentum (rank avg 5/20/60 skip5)
  M. rsi_14d_neg           : RSI(14) mean-reversion (overbought -> underperform)
  N. hl_range_20d          : intraday high-low range / close (OHLC vol proxy)
  O. volume_mom_20d        : 20d volume trend (participation)
  P. vix_slope_10d_neg     : -VIX 10d slope (fear abating = risk-on)
  Q. crypto_tilt_10d       : crypto-mean vs broad cross-section tilt (ETH/BTC avg)
  R. skew_60d_neg          : -60d return skewness (longer-horizon tail asymmetry)
  S. max_dd_20d_neg        : -max drawdown depth over 20d (recent resilience)
  T. xau_beta_60d          : beta to XAU (gold beta defensive)

Also re-validate all 15 persisted library factors (drift check incl 2028+/2029+/2030+ sub-windows).
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), n_ic_dates >= 250, coverage_dates_ge8 >= 0.5,
max_abs_library_correlation < 0.5. Only data <= 2030-07-24 is loaded.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2030-07-24"
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
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change()
eth_r = px["ETH"].pct_change()
wti_r = px["WTI"].pct_change()
xau_r = px["XAU"].pct_change()


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
lib["beta_cn10y_60d"] = beta_of(ret, cn10y.pct_change(), 60)
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

# ---------------- new candidates (2030-07-25 cycle) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)

# A. VIX level gate: defensive (neg VIX beta) only when VIX > 40, else 0
vix_hi = (vix > 40.0).astype(float).reindex(ret.index)
C["vix_level_gate_def"] = lib["beta_vix_60d_neg"].mul(vix_hi, axis=0)

# B. Negative beta to DXY (dollar-hedged)
C["dxy_beta_60d_neg"] = -beta_of(ret, dxy_r, 60)

# C. Negative beta to US10Y changes (rate-hedged)
C["us10y_beta_60d_neg"] = -beta_of(ret, us10y_r, 60)

# D. ETH/BTC relative momentum 20d (crypto internal rotation)
eth_btc_ratio = px["ETH"] / px["BTC"]
C["eth_btc_ratio_mom20"] = pd.DataFrame(
    {c: eth_btc_ratio.pct_change(20).reindex(ret.index) for c in px.columns}, index=px.index)

# E. Beta to WTI (energy beta)
C["wti_beta_60d"] = beta_of(ret, wti_r, 60)

# F. Beta to COPPER (cyclical beta)
C["copper_beta_60d"] = beta_of(ret, px["COPPER"].pct_change(), 60)

# G. 5-day lag autocorrelation (trend persistence): corr(ret_t, ret_{t-5}) over 30d window
auto5 = ret.rolling(30, min_periods=mp(30)).apply(
    lambda x: pd.Series(x).autocorr(lag=5) if len(x) >= 20 else np.nan, raw=False)
C["autocorr_5d"] = auto5

# H. Close position within 20d high-low range (0..1; near-high = strength)
hi20 = hi.rolling(20, min_periods=mp(20)).max()
lo20 = lo.rolling(20, min_periods=mp(20)).min()
C["range_pos_20d"] = (px - lo20) / (hi20 - lo20).replace(0, np.nan)

# I. Distance from 200d SMA (long-term trend)
ma200 = px.rolling(200, min_periods=mp(200, 2)).mean()
C["dist_ma200"] = px / ma200 - 1.0

# J. Upside/downside capture 60d: mean positive ret / mean |negative ret|
up = ret.clip(lower=0); dn = ret.clip(upper=0).abs()
up_mean = up.rolling(60, min_periods=mp(60)).mean()
dn_mean = dn.rolling(60, min_periods=mp(60)).mean()
C["updown_capture_60d"] = up_mean / dn_mean.replace(0, np.nan)

# K. 5d/60d realized vol ratio (term-structure proxy)
C["vol_ratio_5x60"] = rs(ret, 5) / rs(ret, 60).replace(0, np.nan)

# L. Blended multi-horizon momentum (rank avg of 5/20/60 skip5)
mom5s = px.shift(5) / px.shift(10) - 1.0
mom20s = px.shift(5) / px.shift(25) - 1.0
mom60s = px.shift(5) / px.shift(65) - 1.0
blend = (mom5s.rank(axis=1, pct=True) + mom20s.rank(axis=1, pct=True) +
         mom60s.rank(axis=1, pct=True)) / 3.0
C["mom_blend_5_20_60"] = blend

# M. RSI(14) negated (mean reversion: overbought underperforms)
def rsi_series(x, w=14):
    d = x.diff()
    gain = d.clip(lower=0).rolling(w, min_periods=w).mean()
    loss = (-d.clip(upper=0)).rolling(w, min_periods=w).mean()
    rs_ = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs_)

rsi14 = pd.DataFrame({c: rsi_series(px[c], 14) for c in px.columns}, index=px.index)
C["rsi_14d_neg"] = -rsi14

# N. Intraday high-low range / close averaged 20d (OHLC vol proxy)
hl_range = (hi - lo) / px.replace(0, np.nan)
C["hl_range_20d"] = -hl_range.rolling(20, min_periods=mp(20)).mean()

# O. Volume trend 20d (participation)
vol_ma20 = vol.rolling(20, min_periods=mp(20)).mean()
vol_ma60 = vol.rolling(60, min_periods=mp(60)).mean()
C["volume_mom_20d"] = (vol_ma20 / vol_ma60.replace(0, np.nan) - 1.0)

# P. -VIX 10d slope (fear abating = risk-on)
vix_slope10 = (vix / vix.shift(10) - 1.0).reindex(ret.index)
C["vix_slope_10d_neg"] = pd.DataFrame({c: -vix_slope10 for c in px.columns}, index=px.index)

# Q. Crypto tilt: average of BTC/ETH 10d mom vs cross-sectional mean mom10
mom10 = px / px.shift(10) - 1.0
crypto_avg = mom10[["BTC", "ETH"]].mean(axis=1)
cs_mean = mom10.mean(axis=1)
C["crypto_tilt_10d"] = pd.DataFrame(
    {c: (crypto_avg - cs_mean).reindex(ret.index) for c in px.columns}, index=px.index)

# R. -60d skewness (longer-horizon tail asymmetry)
sk60 = ret.rolling(60, min_periods=mp(60)).skew()
C["skew_60d_neg"] = -sk60

# S. -max drawdown depth over 20d (recent resilience)
cummx20 = px.rolling(20, min_periods=mp(20)).max()
C["max_dd_20d_neg"] = -(px / cummx20 - 1.0)

# T. Beta to XAU (gold beta)
C["xau_beta_60d"] = beta_of(ret, xau_r, 60)

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
               "2029+": pd.Timestamp("2029-01-01"), "2030+": pd.Timestamp("2030-01-01")}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'cov':>5s}{'cov8':>5s}  {'2028+IC':>8s}{'2028+IR':>8s} {'2029+IC':>8s}{'2029+IR':>8s} {'2030+IC':>8s}{'2030+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'d10/d20':>11s}", flush=True)
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
    s30 = rec.get("2030+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{cov_ad:>5.2f}{cov_ge8:>5.2f}  "
          f"{(s28[0] if s28 else float('nan')):>8.4f}{(s28[1] if s28 else float('nan')):>8.3f} "
          f"{(s29[0] if s29 else float('nan')):>8.4f}{(s29[1] if s29 else float('nan')):>8.3f} "
          f"{(s30[0] if s30 else float('nan')):>8.4f}{(s30[1] if s30 else float('nan')):>8.3f} "
          f"{(srec[0] if srec else float('nan')):>9.4f}{(srec[1] if srec else float('nan')):>9.3f}  "
          f"{(d10 if d10 is not None else float('nan')):>6.4f}/{(d20 if d20 is not None else float('nan')):>6.4f}", flush=True)

print("\n=== PASS GATE (|IC|>=%.4f, |ICIR|>=%.3f, n>=%d, cov8>=%.2f, librho<0.5) ===" % (IC_TH, ICIR_TH, MIN_IC_DATES, 0.5), flush=True)
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["cov_ge8"] >= 0.5 and r["librho"] < 0.5:
        print(f"  PASS {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ge8={r['cov_ge8']:.2f}", flush=True)
    elif abs(r["ic"]) >= IC_TH or abs(r["icir"]) >= ICIR_TH:
        print(f"  NEAR {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

out = "scripts/miner_1_20300725_screen_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved {out} ({time.time()-t0:.1f}s)", flush=True)
