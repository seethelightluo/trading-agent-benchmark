"""
miner_2 batch screen + re-validation 2029-07-26 cycle (data visible through 2029-07-25).
1) Re-validate all 15 persisted library factors (drift check across sub-windows).
2) Screen new candidate factors designed for high-vol sideways/choppy regime (VIX~66).
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), coverage dates>=8 >= 0.5, max_abs_library_correlation < 0.5.
"""
import json, time, hashlib, zlib, base64, io
import numpy as np
import pandas as pd

VISIBLE = "2029-07-25"
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
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change(); cn10y_r = cn10y.pct_change()
dxy = obs["DXY"]; dxy_r = dxy.pct_change()
spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change()
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

# ---------------- new candidates (high-vol choppy regime focus) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom20 = px.shift(5) / px.shift(25) - 1.0
mom60 = px.shift(5) / px.shift(65) - 1.0
mom120 = px.shift(5) / px.shift(125) - 1.0

# A. Trend efficiency (Kaufman): |net move| / path length over 20d and 60d
C["eff_ratio_20d"] = (px - px.shift(20)).abs() / ret.abs().rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)
C["eff_ratio_60d"] = (px - px.shift(60)).abs() / ret.abs().rolling(60, min_periods=mp(60)).sum().replace(0, np.nan)


# B. Signed trend quality: R2 of linear fit over 60d, signed by slope direction
def r2_trend(x, w=60):
    idx = np.arange(w)
    def f(s):
        y = np.asarray(s, dtype=float)
        if np.isnan(y).any() or np.std(y) == 0:
            return np.nan
        b, a = np.polyfit(idx, y, 1)
        yhat = a + b * idx
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return x.rolling(w, min_periods=mp(w)).apply(f, raw=True)


C["r2_trend_60d_signed"] = r2_trend(px, 60) * np.sign(px.diff(60).reindex(px.index))

# C. Drawdown depth from 60d/120d highs (deep-dd = mean-reversion / bounce candidate)
rollmax60 = px.rolling(60, min_periods=mp(60)).max()
rollmax120 = px.rolling(120, min_periods=mp(120)).max()
C["dd_60d_depth"] = px / rollmax60 - 1.0
C["dd_120d_depth"] = px / rollmax120 - 1.0

# D. Volatility structure: short/long vol ratio and compression
C["vol_ratio_5x60"] = vol10 / vol60.replace(0, np.nan) - 1.0
C["vol_compression_20x60"] = vol20 / vol60.replace(0, np.nan) - 1.0

# E. Cross-sectional dispersion regime (negative: high dispersion -> defensives win)
disp20 = ret.rolling(20, min_periods=mp(20)).std().median(axis=1)
C["dispersion_20d_neg"] = pd.DataFrame({c: -disp20 for c in ret.columns}, index=ret.index)

# F. Stress-conditional SPX beta (negative = defensive in high-VIX states)
vix_high = (vix > vix.rolling(60, min_periods=mp(60)).median()).astype(float)
spx_high_state = pd.DataFrame({c: vix_high for c in ret.columns}, index=ret.index)
C["vix_high_spx_beta_neg_60d"] = -beta_of(ret.mul(spx_high_state), spx_r.mul(vix_high), 60)

# G. Risk-off barometer betas (XAU/SPX ratio, WTI/XAU ratio)
xau_spx_ratio = px["XAU"] / px["SPX"]
C["xau_spx_ratio_beta_60d"] = beta_of(ret, xau_spx_ratio.pct_change(), 60)
wti_xau_ratio = px["WTI"] / px["XAU"]
C["wti_xau_ratio_beta_neg_60d"] = -beta_of(ret, wti_xau_ratio.pct_change(), 60)

# H. Range position & squeeze (Bollinger-style)
min20 = px.rolling(20, min_periods=mp(20)).min()
max20 = px.rolling(20, min_periods=mp(20)).max()
rng20 = (max20 - min20).replace(0, np.nan)
C["range_pos_20d"] = (px - min20) / rng20 - 0.5
sma20 = rm(px, 20)
bbw20 = 4 * rs(px, 20) / sma20.replace(0, np.nan)
C["bbw_squeeze_20d"] = -bbw20  # narrow bandwidth = squeeze

# I. Overnight trend (institutional flow proxy) 60d cumulative
overnight = op / px.shift(1) - 1.0
intraday = px / op - 1.0
C["overnight_trend_60d"] = overnight.rolling(60, min_periods=mp(60)).sum()
C["intraday_trend_60d"] = intraday.rolling(60, min_periods=mp(60)).sum()

# J. Crypto rotation: BTC/ETH ratio momentum (20d)
btc_eth = px["BTC"] / px["ETH"]
C["btc_eth_mom20"] = beta_of(ret, btc_eth.pct_change(), 20)

# K. Positive-day fraction (plain, direction test)
C["pos_day_frac_20d"] = (ret > 0).astype(float).rolling(20, min_periods=mp(20)).mean()

# L. Upside participation in SPX up-days (defensive quality)
spx_up = (spx_r > 0).astype(float)
C["spx_up_beta_60d"] = beta_of(ret.mul(pd.DataFrame({c: spx_up for c in ret.columns}, index=ret.index)),
                                spx_r.mul(spx_up), 60)

# M. Recent 5d reversal after 20d drawdown (bounce candidate)
C["dd_bounce_5d"] = (dd_60d_depth.diff(5)) * (dd_60d_depth < -0.05)

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
               "2028+": pd.Timestamp("2028-01-01"), "recent": pd.Timestamp("2028-04-01")}

results = {}
print(f"\n{'name':<28}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'2028+IC':>8s}{'2028+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'d10/d20':>11s}", flush=True)
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
    s27 = rec.get("2027+", (None, None))
    s28 = rec.get("2028+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<28}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{s28[0] if s28 else float('nan'):>8.4f}{s28[1] if s28 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}  "
          f"{d10 if d10 is not None else float('nan'):>6.4f}/{d20 if d20 is not None else float('nan'):>6.4f}", flush=True)

print("\n--- DRIFT CHECK (library factors) ---", flush=True)
drift_flags = []
for name in lib:
    r = results[name]
    full_ok = abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH
    rec = r["sub"].get("recent")
    s27 = r["sub"].get("2027+")
    flag = ""
    if not full_ok:
        flag += "FULL_FAIL "
    if rec is not None and (abs(rec[0]) < IC_TH or abs(rec[1]) < ICIR_TH):
        flag += "RECENT_WEAK "
    if s27 is not None and (abs(s27[0]) < IC_TH or abs(s27[1]) < ICIR_TH):
        flag += "2027_WEAK "
    if flag:
        drift_flags.append((name, flag))
    print(f"{name:<28} full={r['ic']:+.4f}/{r['icir']:+.3f} 2027+={s27} recent={rec} {flag}", flush=True)

passers = []
for name in C:
    r = results[name]
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < 0.5)
    if ok:
        passers.append(name)
    print(f"NEW {name:<28} PASS={ok} ic={r['ic']:+.4f} icir={r['icir']:+.3f} librho={r['librho']:.3f} "
          f"cov_ge8={r['cov_ge8']:.2f} recent={r['sub'].get('recent')}", flush=True)

print("\nPASSERS:", passers, flush=True)
print("DRIFT_FLAGS:", drift_flags, flush=True)

with open("scripts/miner_2_20290726_reval_screen_results.json", "w") as f:
    json.dump({"visible": VISIBLE, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": results, "passers": passers, "drift_flags": drift_flags}, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
