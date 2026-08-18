"""
miner_2 batch screen 2028-09-21 cycle (data visible through 2028-09-20).
Fresh candidate factors on the 15-instrument cross-asset universe.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (10d forward, daily cross-sectional rank IC).
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2028-09-20"
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
usdjpy = obs["USDJPY"]; usdjpy_r = usdjpy.pct_change()
eurusd = obs["EURUSD"]; eurusd_r = eurusd.pct_change()
usdcny = obs["USDCNY"]; usdcny_r = usdcny.pct_change()


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


# ---------------- library signals (15 persisted factors, recomputed) ----------------
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

# ---------------- new candidates ----------------
C = {}
vol20 = rs(ret, 20); vol60 = rs(ret, 60)
spx_r = px["SPX"].pct_change()
rng = (hi - lo) / px.replace(0, np.nan)          # daily range / close
oc = (op.shift(1) - px.shift(1)) / px.shift(1).replace(0, np.nan)  # overnight gap (open vs prev close)
co = (px - op) / op.replace(0, np.nan)           # intraday move (close vs open)

# 1) Parkinson range vol (20d)
C["parkinson_vol_20d"] = np.sqrt((np.log(hi / lo) ** 2).rolling(20, min_periods=mp(20)).mean() / (4 * np.log(2)))

# 2) Range-to-close vol ratio: intraday activity vs close-close vol
C["range_vol_ratio_20d"] = rs(rng, 20) / vol20.replace(0, np.nan)

# 3) Amihud illiquidity 20d (needs volume)
amihud = (ret.abs() / vol.replace(0, np.nan))
C["amihud_illiq_20d"] = np.log1p(amihud.rolling(20, min_periods=mp(20)).mean())

# 4) Volume trend 20d/60d ratio
C["vol_trend_20x60"] = rm(vol, 20) / rm(vol, 60).replace(0, np.nan) - 1.0

# 5) OBV momentum: slope of signed volume accumulation over 20d
obv = (np.sign(ret) * vol).fillna(0).cumsum()
C["obv_mom_20d"] = obv / obv.shift(20).replace(0, np.nan) - 1.0

# 6) Close position in daily range (20d avg): 1 = closes at high
C["close_pos_20d"] = ((px - lo) / (hi - lo).replace(0, np.nan)).rolling(20, min_periods=mp(20)).mean()

# 7) Upper shadow ratio (20d avg): sellers rejecting highs
upper_sh = (hi - np.maximum(op, px)) / (hi - lo).replace(0, np.nan)
C["upper_shadow_20d"] = upper_sh.rolling(20, min_periods=mp(20)).mean()

# 8) Drawdown depth vs 120d rolling high (negated = distance from peak)
C["dd_120d"] = (px / px.rolling(120, min_periods=mp(120)).max() - 1.0)

# 9) Kaufman efficiency ratio 60d (trend purity)
C["kaufman_eff_60d"] = (px - px.shift(60)).abs() / ret.abs().rolling(60, min_periods=mp(60)).sum().replace(0, np.nan)

# 10) Realized kurtosis 60d
C["kurt_60d"] = ret.rolling(60, min_periods=mp(60)).kurt()

# 11) Vol term structure: vol20 / vol60 - 1 (short vol premium)
C["vol_term_20x60"] = vol20 / vol60.replace(0, np.nan) - 1.0

# 12) Sharpe 60d (risk-adjusted return)
C["sharpe_60d"] = rm(ret, 60) / vol60.replace(0, np.nan)

# 13) Downside beta vs SPX (beta computed on SPX-down days only)
spx_down = spx_r < 0
spx_down_r = spx_r.where(spx_down)
C["downside_beta_spx_60d"] = beta_of(ret, spx_down_r.fillna(0), 60)

# 14) DXY beta 60d (FX sensitivity)
C["beta_dxy_60d"] = beta_of(ret, dxy_r, 60)

# 15) USDJPY beta 60d
C["beta_usdjpy_60d"] = beta_of(ret, usdjpy_r, 60)

# 16) EURUSD beta 60d
C["beta_eurusd_60d"] = beta_of(ret, eurusd_r, 60)

# 17) USDCNY beta 60d
C["beta_usdcny_60d"] = beta_of(ret, usdcny_r, 60)

# 18) WTI/COPPER ratio momentum (growth-commodity regime, 60d beta x 20d move)
wc_ratio = px["WTI"] / px["COPPER"]
C["wti_copper_cond_20d"] = beta_of(ret, wc_ratio.pct_change(), 60).mul(
    wc_ratio.pct_change(20).reindex(ret.index), axis=0)

# 19) SPX/XAU ratio momentum (risk appetite regime)
sx_ratio = px["SPX"] / px["XAU"]
C["spx_xau_cond_20d"] = beta_of(ret, sx_ratio.pct_change(), 60).mul(
    sx_ratio.pct_change(20).reindex(ret.index), axis=0)

# 20) Overnight gap momentum (20d sum of overnight gaps)
C["gap_mom_20d"] = oc.rolling(20, min_periods=mp(20)).sum()

# 21) Intraday reversal 5d: recent intraday move (close-open), mean-reversion flavor
C["intraday_rev_5d"] = -co.rolling(5, min_periods=mp(5)).mean()

# 22) Range breakout: current close vs 20d average high-low midpoint
mid20 = ((hi.rolling(20, min_periods=mp(20)).max() + lo.rolling(20, min_periods=mp(20)).min()) / 2)
C["mid_break_20d"] = px / mid20 - 1.0

# 23) VIX/SPX-vol ratio: fear premium per unit of realized vol (observation-based)
vix_reidx = vix.reindex(ret.index)
C["vix_fear_ratio_20d"] = (vix_reidx / vol20["SPX"].replace(0, np.nan)).to_frame("SPX") if False else \
    pd.DataFrame({c: (vix_reidx / vol20[c].replace(0, np.nan)) for c in ret.columns}, index=ret.index)

print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    common = factor.index.intersection(fwd.index)
    fr = factor.reindex(common).rank(axis=1, pct=True)
    rr = fwd.reindex(common).rank(axis=1, pct=True)
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
               "recent": pd.Timestamp("2027-06-01")}

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
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
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
    print(f"{name:<26} full={r['ic']:+.4f}/{r['icir']:+.3f} 2027+={s27} recent={rec} {flag}", flush=True)

passers = []
for name in C:
    r = results[name]
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < 0.5)
    rec = r["sub"].get("recent")
    s27 = r["sub"].get("2027+")
    stable = True
    if rec is not None:
        stable = stable and (abs(rec[0]) >= IC_TH * 0.7 and abs(rec[1]) >= ICIR_TH * 0.7)
    if s27 is not None:
        stable = stable and (abs(s27[0]) >= IC_TH * 0.5 and abs(s27[1]) >= ICIR_TH * 0.5)
    ok = ok and stable
    if ok:
        passers.append(name)
        print(f"*** PASSER: {name} IC={r['ic']:.4f} ICIR={r['icir']:.3f} librho={r['librho']:.3f} "
              f"n={r['n']} turn={r['turn']:.3f} 2027={s27} recent={rec}", flush=True)
    else:
        print(f"    fail: {name} IC={r['ic']:.4f} ICIR={r['icir']:.3f} librho={r['librho']:.3f} "
              f"n={r['n']} cov_ge8={r['cov_ge8']:.2f} 2027={s27} recent={rec}", flush=True)

print(f"\npassers={passers} drift_flags={[f[0] for f in drift_flags]}", flush=True)

with open("scripts/miner_2_20280921_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
