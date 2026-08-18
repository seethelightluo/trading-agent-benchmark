"""
miner_2 batch screen + re-validation 2029-09-20 cycle (data visible through 2029-09-19).
1) Re-validate all 15 persisted library factors (drift check across sub-windows incl 2029+).
2) Screen NEW candidate factors designed for high-vol sideways/choppy regime:
   - trend quality / whipsaw resistance (Kaufman efficiency ratio, trend R^2)
   - risk-adjusted short momentum, vol-scaled short reversal
   - downside beta (defensive), vol compression, variance-ratio trend persistence
   - volume-flow participation, drawdown depth, RSI mean reversion
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), n_ic_dates>=250, coverage dates>=8 >= 0.5, max_abs_library_correlation < 0.5.
Only data <= 2029-09-19 is loaded; nothing beyond the simulation date is touched.
"""
import json, time, hashlib, zlib, base64
import numpy as np
import pandas as pd

VISIBLE = "2029-09-19"
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
xau_r = px["XAU"].pct_change()
wti_r = px["WTI"].pct_change()


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

# ---------------- new candidates (2029-09-20 cycle) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom10 = px.shift(5) / px.shift(15) - 1.0
mom20 = px.shift(5) / px.shift(25) - 1.0
mom60 = px.shift(5) / px.shift(65) - 1.0
mom120 = px.shift(5) / px.shift(125) - 1.0
rollmax20 = px.rolling(20, min_periods=mp(20)).max()
rollmax60 = px.rolling(60, min_periods=mp(60)).max()
rollmin60 = px.rolling(60, min_periods=mp(60)).min()

# A. Kaufman efficiency ratio 20d/60d: |net move| / sum(|1d moves|) -> trend quality (whipsaw resistance)
def kaufman_er(w):
    net = (px - px.shift(w)).abs()
    path = ret.abs().rolling(w, min_periods=mp(w)).sum()
    return net / path.replace(0, np.nan)

C["eff_ratio_20d"] = kaufman_er(20)
C["eff_ratio_60d"] = kaufman_er(60)

# B. Trend R^2 over 60d (linear fit of log price): smooth-trend consistency
def trend_rsq(w=60):
    idx = np.arange(w)
    x = idx - idx.mean()
    denom = (x ** 2).sum()
    out = {}
    for c in px.columns:
        lp = np.log(px[c])
        s = lp.rolling(w, min_periods=mp(w))
        # rolling slope via covariance of lp with time
        slope = s.apply(lambda y: float(np.cov(np.arange(len(y)), y, ddof=1)[0, 1]) / denom, raw=True)
        # predicted = mean + slope * (t - tmean); R2 = 1 - SSres/SStot
        def r2calc(y):
            n = len(y)
            tt = np.arange(n)
            b = float(np.cov(tt, y, ddof=1)[0, 1]) / float(np.var(tt, ddof=1))
            a = y.mean() - b * tt.mean()
            pred = a + b * tt
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out[c] = s.apply(r2calc, raw=True)
    return pd.DataFrame(out)

print("building trend_rsq_60d...", flush=True)
C["trend_rsq_60d"] = trend_rsq(60)

# C. Risk-adjusted short momentum (10d mom / 10d vol) - short-horizon timing in chop
C["mom10_vol10_adj"] = mom10 / vol10.replace(0, np.nan)

# D. Vol-scaled short reversal (mean reversion in choppy market): -(5d ret)/vol20
C["rev5_vol_adj"] = -(px / px.shift(5) - 1.0) / vol20.replace(0, np.nan)

# E. Downside beta to SPX (defensive quality): beta computed on SPX-down days
spx_dn = (spx_r < 0)
spx_dn_r = spx_r.where(spx_dn)
C["downside_beta_spx_60d"] = beta_of(ret, spx_dn_r, 60)

# F. Vol compression 10x60 (squeeze): -vol10/vol60, high = compressed vol -> expansion candidate
C["vol_compress_10x60"] = -(vol10 / vol60.replace(0, np.nan))

# G. Variance-ratio trend persistence 20x1: var(20d ret)/(20*var(1d ret)); >1 trending, <1 mean-reverting
vr_num = ret.rolling(20, min_periods=mp(20)).apply(lambda y: np.nanvar(y), raw=True)
vr_den = ret.rolling(1, min_periods=1).apply(lambda y: np.nanvar(y), raw=True)
C["var_ratio_20x1"] = vr_num / (20.0 * vr_den.replace(0, np.nan))

# H. Drawdown depth from 60d high (oversold / reversal candidate in chop)
C["dd_from_high_60d"] = (px / rollmax60 - 1.0)

# I. Volume-flow imbalance 20d: (up-vol - down-vol)/total-vol (accumulation)
upv = vol.where(ret > 0, 0.0)
dnv = vol.where(ret < 0, 0.0)
C["vol_flow_imb_20d"] = (upv.rolling(20, min_periods=mp(20)).sum()
                         - dnv.rolling(20, min_periods=mp(20)).sum()) \
                        / vol.rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)

# J. Volume participation trend: 20d avg vol / 60d avg vol (rising participation)
C["vol_trend_20x60"] = vol.rolling(20, min_periods=mp(20)).mean() \
                       / vol.rolling(60, min_periods=mp(60)).mean().replace(0, np.nan) - 1.0

# K. RSI-like 14d oscillator (mean reversion): 100 - 100/(1+RS), use z-scored -RSI (oversold = high)
def rsi14(px_ser, w=14):
    d = px_ser.diff()
    up = d.clip(lower=0).rolling(w, min_periods=mp(w)).mean()
    dn = (-d.clip(upper=0)).rolling(w, min_periods=mp(w)).mean()
    rs_v = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs_v)

C["rsi14_neg"] = -pd.DataFrame({c: rsi14(px[c]) for c in px.columns})

# L. 60d momentum gated by Kaufman efficiency (only strong-trend momentum counts)
C["mom60_eff_cond"] = mom60 * (C["eff_ratio_60d"] > 0.25)

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
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
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
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s28[0] if s28 else float('nan'):>8.4f}{s28[1] if s28 else float('nan'):>8.3f} "
          f"{s29[0] if s29 else float('nan'):>8.4f}{s29[1] if s29 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}  "
          f"{d10 if d10 is not None else float('nan'):>6.4f}/{d20 if d20 is not None else float('nan'):>6.4f}", flush=True)

print("\n--- DRIFT CHECK (library factors) ---", flush=True)
drift_flags = []
for name in lib:
    r = results[name]
    full_ok = abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH
    rec = r["sub"].get("recent")
    s29 = r["sub"].get("2029+")
    flag = ""
    if not full_ok:
        flag += "FULL_FAIL "
    if rec is not None and (abs(rec[0]) < IC_TH or abs(rec[1]) < ICIR_TH):
        flag += "RECENT_WEAK "
    if s29 is not None and (abs(s29[0]) < IC_TH or abs(s29[1]) < ICIR_TH):
        flag += "2029_WEAK "
    if flag:
        drift_flags.append((name, flag))
    print(f"{name:<26} full={r['ic']:+.4f}/{r['icir']:+.3f} 2029+={s29} recent={rec} {flag}", flush=True)

passers = []
for name in C:
    r = results[name]
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < 0.5)
    if ok:
        passers.append(name)
    print(f"NEW {name:<26} PASS={ok} ic={r['ic']:+.4f} icir={r['icir']:+.3f} librho={r['librho']:.3f} "
          f"cov_ge8={r['cov_ge8']:.2f} recent={r['sub'].get('recent')} 2029+={r['sub'].get('2029+')}", flush=True)

print("\nPASSERS:", passers, flush=True)
print("DRIFT_FLAGS:", drift_flags, flush=True)

with open("scripts/miner_2_20290920_screen_results.json", "w") as f:
    json.dump({"visible": VISIBLE, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": results, "passers": passers, "drift_flags": drift_flags}, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
