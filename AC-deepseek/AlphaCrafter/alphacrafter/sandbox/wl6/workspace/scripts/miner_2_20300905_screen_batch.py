"""
miner_2 batch screen + re-validation 2030-09-05 cycle (data visible through 2030-09-04).
Regime per memory: high-vol choppy sideways VIX 43.4 (off 54 peak), SPX 4803 +4.1%/20d recovery,
BTC +24.2%/20d strong rebound, WTI +22.1%/20d bounce, XAU +3.4%/20d strong, 000300.SH -2.2% weak,
US10Y -3.6%/20d (falling yields), NDX +1.4%, ETH +8.6%. High dispersion, risk-on rotation under high VIX.

New candidates this cycle (regime-tailored: crypto rebound, energy bounce, gold strength, rates):
  A. crypto_mom_20d        : avg(BTC,ETH) 20d momentum (crypto rebound regime)
  B. btc_mom_20d           : BTC-only 20d momentum (crypto leader)
  C. xau_breakout_40d      : XAU vs 40d rolling high (gold strength)
  D. wti_mom_20d           : WTI 20d momentum (energy bounce)
  E. us10y_mom_20d_neg     : negative US10Y 20d momentum (falling yield = bond bid)
  F. crypto_eq_rel_20d     : crypto avg 20d mom minus equity avg 20d mom (rotation)
  G. vix_retreat_10d_neg   : negative VIX 10d change (fear abating = risk-on)
  H. max_ret_20d_neg       : negative max daily return 20d (lottery reversal in high vol)
  I. down_up_vol_ratio_20d : downside vol / upside vol (asymmetry)
  J. corr_spx_60d_neg      : negative 60d corr to SPX (diversifier in high vol)
  K. mom20_sharpe          : 20d momentum / 20d vol (risk-adjusted momentum)
  L. eth_btc_rel_mom_20d   : ETH minus BTC 20d momentum (crypto internal rotation)
  M. range_pos_10d         : avg close position in 10d range (mean-reversion/trend)
  N. vol_ratio_10x60       : 10d vol / 60d vol (vol term structure)
  O. ndx_spx_rel_mom20     : NDX/SPX relative momentum (tech leadership)
  P. xau_copper_ratio_mom20: gold/copper ratio 20d momentum (risk-off vs cyclical)
  Q. skew_10d_neg          : short-horizon negative skewness (lottery)
  R. btc_beta_neg_20d      : negative beta to BTC returns 20d (crypto diversity)
  S. ret_auto_5d           : 5d return autocorrelation (trend persistence)

Also re-validate all 15 persisted library factors (drift check incl 2028+/2029+/2030+ sub-windows).
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), n_ic_dates >= 250, coverage dates>=8 >= 0.5.
Only data <= 2030-09-04 is loaded; nothing beyond the simulation date is touched.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2030-09-04"
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

# ---------------- new candidates (2030-09-05 cycle) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom20 = px / px.shift(20) - 1.0
mom10 = px / px.shift(10) - 1.0

# A. crypto avg momentum 20d
btc_mom20 = (px["BTC"] / px["BTC"].shift(20) - 1.0).reindex(ret.index)
eth_mom20 = (px["ETH"] / px["ETH"].shift(20) - 1.0).reindex(ret.index)
crypto_avg20 = ((btc_mom20 + eth_mom20) / 2.0)
C["crypto_mom_20d"] = pd.DataFrame({c: crypto_avg20 for c in px.columns}, index=px.index)

# B. BTC-only momentum 20d
C["btc_mom_20d"] = pd.DataFrame({c: btc_mom20 for c in px.columns}, index=px.index)

# C. XAU vs 40d rolling high (breakout strength)
hi40 = px.rolling(40, min_periods=mp(40)).max()
xau_brk = (px["XAU"] / hi40["XAU"] - 1.0).reindex(ret.index)
C["xau_breakout_40d"] = pd.DataFrame({c: xau_brk for c in px.columns}, index=px.index)

# D. WTI 20d momentum
wti_mom20 = (px["WTI"] / px["WTI"].shift(20) - 1.0).reindex(ret.index)
C["wti_mom_20d"] = pd.DataFrame({c: wti_mom20 for c in px.columns}, index=px.index)

# E. Negative US10Y 20d momentum (falling yield = bond bid)
us10y_mom20 = (us10y / us10y.shift(20) - 1.0).reindex(ret.index)
C["us10y_mom_20d_neg"] = pd.DataFrame({c: -us10y_mom20 for c in px.columns}, index=px.index)

# F. crypto avg minus equity avg 20d momentum (rotation)
eq_idx = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX']
eq_avg20 = mom20[eq_idx].mean(axis=1)
rot = (crypto_avg20 - eq_avg20).reindex(ret.index)
C["crypto_eq_rel_20d"] = pd.DataFrame({c: rot for c in px.columns}, index=px.index)

# G. negative VIX 10d change (fear abating)
vix_chg10 = (vix / vix.shift(10) - 1.0).reindex(ret.index)
C["vix_retreat_10d_neg"] = pd.DataFrame({c: -vix_chg10 for c in px.columns}, index=px.index)

# H. negative max daily return 20d (lottery reversal)
maxr20 = ret.rolling(20, min_periods=mp(20)).max()
C["max_ret_20d_neg"] = -maxr20

# I. downside vol / upside vol 20d (asymmetry)
up = ret.clip(lower=0)
down2 = (ret.clip(upper=0) * -1.0)
down_vol20 = rs(down2, 20)
up_vol20 = rs(up, 20)
C["down_up_vol_ratio_20d"] = (down_vol20 / up_vol20.replace(0, np.nan))

# J. negative 60d correlation to SPX (diversifier)
corr_spx = corr_of(ret, spx_r, 60)
C["corr_spx_60d_neg"] = -corr_spx

# K. 20d momentum / 20d vol (risk-adjusted momentum)
C["mom20_sharpe"] = mom20 / vol20.replace(0, np.nan)

# L. ETH minus BTC 20d momentum (crypto internal rotation)
eth_btc_rel = (eth_mom20 - btc_mom20).reindex(ret.index)
C["eth_btc_rel_mom_20d"] = pd.DataFrame({c: eth_btc_rel for c in px.columns}, index=px.index)

# M. avg close position in 10d range
hi10 = px.rolling(10, min_periods=mp(10)).max()
lo10 = px.rolling(10, min_periods=mp(10)).min()
rng_pos = ((px - lo10) / (hi10 - lo10).replace(0, np.nan)).rolling(10, min_periods=mp(10)).mean()
C["range_pos_10d"] = rng_pos

# N. 10d vol / 60d vol (vol term structure)
C["vol_ratio_10x60"] = vol10 / vol60.replace(0, np.nan)

# O. NDX/SPX relative momentum 20d (tech leadership)
ndx_spx_rel = (px["NDX"] / px["SPX"]).pct_change(20).reindex(ret.index)
C["ndx_spx_rel_mom20"] = pd.DataFrame({c: ndx_spx_rel for c in px.columns}, index=px.index)

# P. gold/copper ratio 20d momentum (risk-off vs cyclical)
xau_copper_r = (xau_copper_ratio / xau_copper_ratio.shift(20) - 1.0).reindex(ret.index)
C["xau_copper_ratio_mom20"] = pd.DataFrame({c: xau_copper_r for c in px.columns}, index=px.index)

# Q. short-horizon negative skewness (lottery)
sk10 = ret.rolling(10, min_periods=mp(10)).skew()
C["skew_10d_neg"] = -sk10

# R. negative beta to BTC returns 20d (crypto diversity)
C["btc_beta_neg_20d"] = -beta_of(ret, btc_r, 20)

# S. 5d return autocorrelation (trend persistence)
r5 = ret.rolling(5, min_periods=mp(5)).apply(
    lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 5 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan,
    raw=False)
C["ret_auto_5d"] = r5

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
      f"{'2028+IC':>8s}{'2028+IR':>8s} {'2029+IC':>8s}{'2029+IR':>8s} {'2030+IC':>8s}{'2030+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'d10/d20':>11s}", flush=True)
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
          f"{(s28[0] if s28 else float('nan')):>8.4f}{(s28[1] if s28 else float('nan')):>8.3f} "
          f"{(s29[0] if s29 else float('nan')):>8.4f}{(s29[1] if s29 else float('nan')):>8.3f} "
          f"{(s30[0] if s30 else float('nan')):>8.4f}{(s30[1] if s30 else float('nan')):>8.3f} "
          f"{(srec[0] if srec else float('nan')):>9.4f}{(srec[1] if srec else float('nan')):>9.3f}  "
          f"{(d10 if d10 is not None else float('nan')):>6.4f}/{(d20 if d20 is not None else float('nan')):>6.4f}", flush=True)

print("\n=== PASS GATE (|IC|>=%.4f, |ICIR|>=%.3f, n>=%d, cov>=%.2f) ===" % (IC_TH, ICIR_TH, MIN_IC_DATES, 0.5), flush=True)
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["cov_ge8"] >= 0.5:
        print(f"  PASS {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ge8={r['cov_ge8']:.2f}", flush=True)
    elif abs(r["ic"]) >= IC_TH or abs(r["icir"]) >= ICIR_TH:
        print(f"  NEAR {name}: ic={r['ic']:.4f} icir={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

out = "scripts/miner_2_20300905_screen_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved {out} ({time.time()-t0:.1f}s)", flush=True)
