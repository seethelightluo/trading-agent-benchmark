"""
miner_2 batch screen + re-validation 2029-10-04 cycle (data visible through 2029-10-03).
1) Re-validate all 15 persisted library factors (drift check across sub-windows incl 2029+).
2) Screen NEW candidate factors for high-vol sideways/choppy regime (VIX~66 off peak,
   short-horizon timing working per 20290920/20291004 trader feedback):
   - intraday vs overnight return decomposition (gap noise separation)
   - cross-asset rotation sensitivity (XAU/SPX ratio beta, crypto beta, avg pairwise corr)
   - drawdown depth / time-under-water (mean reversion & recovery candidates)
   - liquidity (Amihud), volume-price participation correlation
   - higher moments (skew, kurtosis), fast RSI(2), z-score mean reversion, range amplitude
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 (10d fwd,
daily cross-sectional rank IC), n_ic_dates>=250, coverage dates>=8 >= 0.5, max_abs_library_correlation < 0.5.
Only data <= 2029-10-03 is loaded; nothing beyond the simulation date is touched.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2029-10-03"
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
spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change()


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

# ---------------- new candidates (2029-10-04 cycle) ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
mom10 = px.shift(5) / px.shift(15) - 1.0
rollmax60 = px.rolling(60, min_periods=mp(60)).max()
rollmax120 = px.rolling(120, min_periods=mp(120)).max()

# intraday (close/open - 1) and overnight gap (open/prev_close - 1) decomposition
intraday = px / op - 1.0
gap = op / px.shift(1) - 1.0

# A. Intraday momentum 10d (sum of close/open-1, skips gap noise)
C["intraday_mom_10d"] = intraday.rolling(10, min_periods=mp(10)).sum()
# B. Overnight (gap) momentum 10d
C["overnight_mom_10d"] = gap.rolling(10, min_periods=mp(10)).sum()
# C. Gap reversal 5d (short-horizon gap fading in chop)
C["gap_rev_5d"] = -(gap.rolling(5, min_periods=mp(5)).mean())

# D. XAU/SPX ratio beta 60d (risk-off rotation sensitivity)
xau_spx_ratio = px["XAU"] / px["SPX"]
C["xau_spx_beta_60d"] = beta_of(ret, xau_spx_ratio.pct_change(), 60)
# E. Crypto beta 60d (correlation to BTC)
C["crypto_beta_60d"] = beta_of(ret, btc_r, 60)

# F. Avg pairwise correlation 60d (systemic dispersion; negative = low-corr diversification)
print("building avg pairwise corr 60d...", flush=True)
avg_corr = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
cols = ret.columns
for i, c1 in enumerate(cols):
    acc = []
    for j, c2 in enumerate(cols):
        if j <= i:
            continue
        cc = ret[c1].rolling(60, min_periods=mp(60)).corr(ret[c2])
        acc.append(cc)
    if acc:
        avg_corr[c1] = sum(acc) / len(acc)
for c1 in cols:
    acc = []
    for j, c2 in enumerate(cols):
        if j == c1:
            continue
        cc = ret[c1].rolling(60, min_periods=mp(60)).corr(ret[c2])
        acc.append(cc)
    avg_corr[c1] = sum(acc) / len(acc)
C["avg_corr_60d_neg"] = -avg_corr
# G. SPX correlation 60d negated (defensive decorrelation)
C["corr_spx_60d_neg"] = -corr_of(ret, spx_r, 60)

# H. Drawdown depth from 120d high (oversold / mean reversion)
C["dd_120d"] = (px / rollmax120 - 1.0)
# I. Time under water 120d (fraction of days below trailing 120d high)
cummax120 = px.rolling(120, min_periods=mp(120)).max()
below = (px < cummax120).astype(float)
C["tuw_120d"] = below.rolling(120, min_periods=mp(120)).mean()

# J. Amihud illiquidity 20d (|ret|/volume), high = illiquid
amihud = ret.abs() / vol.replace(0, np.nan)
C["amihud_20d"] = amihud.rolling(20, min_periods=mp(20)).mean()
# K. Volume-price participation correlation 20d (ret corr with volume change)
vchg = vol.pct_change()
C["vol_price_corr_20d"] = ret.rolling(20, min_periods=mp(20)).corr(vchg)

# L. Realized skew 60d negated (negative skew premium)
sk60 = ret.rolling(60, min_periods=mp(60)).skew()
C["skew_60d_neg"] = -sk60
# M. Kurtosis 20d negated (tail-risk aversion)
kurt20 = ret.rolling(20, min_periods=mp(20)).kurt()
C["kurt_20d_neg"] = -kurt20

# N. Fast RSI(2) negated (short-horizon mean reversion)
def rsi(px_ser, w=2):
    d = px_ser.diff()
    up = d.clip(lower=0).rolling(w, min_periods=mp(w)).mean()
    dn = (-d.clip(upper=0)).rolling(w, min_periods=mp(w)).mean()
    rsv = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rsv)

C["rsi2_neg"] = -pd.DataFrame({c: rsi(px[c], 2) for c in px.columns})

# O. Z-score from 20d mean negated (mean reversion distance)
sma20 = rm(px, 20)
C["zscore20_neg"] = -((px - sma20) / vol20.replace(0, np.nan))

# P. Range amplitude 20d (mean high-low range / close): volatility expansion
C["range_amp_20d"] = ((hi - lo) / px).rolling(20, min_periods=mp(20)).mean()

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
    print(f"NEW {name:<24} PASS={ok} ic={r['ic']:+.4f} icir={r['icir']:+.3f} librho={r['librho']:.3f} "
          f"cov_ge8={r['cov_ge8']:.2f} recent={r['sub'].get('recent')} 2029+={r['sub'].get('2029+')}", flush=True)

print("\nPASSERS:", passers, flush=True)
print("DRIFT_FLAGS:", drift_flags, flush=True)

with open("scripts/miner_2_20291004_screen_results.json", "w") as f:
    json.dump({"visible": VISIBLE, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": results, "passers": passers, "drift_flags": drift_flags}, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
