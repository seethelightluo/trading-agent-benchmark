"""
miner_2 batch screen 2028-12-28 cycle (data visible through 2028-12-27).
Fresh candidate factors on the 15-instrument cross-asset universe.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (10d forward, daily cross-sectional rank IC).
Re-validates the two 2028-09-21 passers (upper_shadow_20d, intraday_rev_5d) with 3 months of
new data, and screens a fresh batch from under-covered families (volume/flow, trend-quality,
tail-risk, rates, cross-asset dispersion, regime-conditional).
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2028-12-27"
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
vol20 = rs(ret, 20); vol60 = rs(ret, 60); vol10 = rs(ret, 10)
spx_r = px["SPX"].pct_change()
rng = (hi - lo) / px.replace(0, np.nan)
oc = (op.shift(1) - px.shift(1)) / px.shift(1).replace(0, np.nan)
co = (px - op) / op.replace(0, np.nan)

# ---- re-validation of 2028-09-21 passers ----
upper_sh = (hi - np.maximum(op, px)) / (hi - lo).replace(0, np.nan)
C["upper_shadow_20d"] = upper_sh.rolling(20, min_periods=mp(20)).mean()
C["intraday_rev_5d"] = -co.rolling(5, min_periods=mp(5)).mean()

# ---- A. volume / money-flow family ----
# 1) Money Flow Index 14d (classic overbought/oversold pressure)
tp = (hi + lo + px) / 3.0
mf = tp * vol
pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=mp(14)).sum()
neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=mp(14)).sum()
mfr = pos_mf / neg_mf.replace(0, np.nan)
C["mfi_14d"] = 100 - 100 / (1 + mfr)

# 2) Chaikin Money Flow 20d (accumulation/distribution)
mfv = ((px - lo) - (hi - px)) / (hi - lo).replace(0, np.nan)
cmf = (mfv * vol).rolling(20, min_periods=mp(20)).sum() / vol.rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)
C["cmf_20d"] = cmf

# 3) Volume z-score (20d): surge vs normal
volz = (vol - rm(vol, 20)) / vol.rolling(20, min_periods=mp(20)).std().replace(0, np.nan)
C["vol_zscore_20d"] = volz

# 4) Volume-price correlation 20d (trend confirmation)
C["vp_corr_20d"] = ret.rolling(20, min_periods=mp(20)).corr(vol.pct_change().replace(0, np.nan))

# 5) Volume concentration: 10d volume share of 60d (participation shift)
C["vol_share_10x60"] = rm(vol, 10) / rm(vol, 60).replace(0, np.nan)

# ---- B. trend-quality / technical family ----
# 6) RSI 14d (mean-reversion oscillator)
delta = ret
up = delta.clip(lower=0).rolling(14, min_periods=mp(14)).mean()
dn = (-delta.clip(upper=0)).rolling(14, min_periods=mp(14)).mean()
C["rsi_14d"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

# 7) Bollinger %B 20d/2std (position within bands)
mid20 = rm(px, 20)
sd20 = px.rolling(20, min_periods=mp(20)).std()
C["bb_pctb_20d"] = (px - (mid20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)

# 8) ADX-like trend strength: |close - close[10]| / ATR10
tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(), (lo - px.shift(1)).abs()], axis=1).max(axis=1)
C["trend_atr_10d"] = (px - px.shift(10)).abs() / tr.rolling(10, min_periods=mp(10)).mean().replace(0, np.nan)

# 9) Gain-loss asymmetry 20d: upside capture vs downside (quality of trend)
up_sum = delta.clip(lower=0).rolling(20, min_periods=mp(20)).sum()
dn_sum = (-delta.clip(upper=0)).rolling(20, min_periods=mp(20)).sum()
C["gain_loss_asym_20d"] = (up_sum - dn_sum) / (up_sum + dn_sum).replace(0, np.nan)

# ---- C. tail-risk / drawdown family ----
# 10) Max 20d drawdown (negated: deeper drawdown = lower score)
roll_max20 = px.rolling(20, min_periods=mp(20)).max()
C["dd_20d_neg"] = (px / roll_max20 - 1.0)

# 11) VaR 95% over 60d (negated: riskier = lower)
C["var95_60d_neg"] = -ret.rolling(60, min_periods=mp(60)).quantile(0.05)

# 12) Max daily loss 20d (negated)
C["maxloss_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).min()

# 13) Time-under-water proxy: fraction of days below 60d SMA
C["below_sma60_frac"] = (px < rm(px, 60)).rolling(60, min_periods=mp(60)).mean()

# ---- D. rates / carry family ----
# 14) US10Y vs CN10Y spread beta (rate differential sensitivity)
uscn_spread = us10y - cn10y
C["beta_uscn_spread_60d"] = beta_of(ret, uscn_spread.pct_change(), 60)

# 15) US10Y momentum (yield trend, sign-flipped = falling yields = risk-on)
C["us10y_mom_20d"] = -us10y.pct_change(20).reindex(ret.index).to_frame().apply(
    lambda s: s, axis=0) if False else pd.DataFrame(
    {c: (-us10y.pct_change(20)).reindex(ret.index) for c in ret.columns}, index=ret.index)

# 16) CN10Y momentum (sign-flipped)
C["cn10y_mom_20d"] = pd.DataFrame(
    {c: (-cn10y.pct_change(20)).reindex(ret.index) for c in ret.columns}, index=ret.index)

# ---- E. cross-asset dispersion / relative family ----
# 17) Cross-sectional dispersion of 10d returns (regime breadth)
cs_disp = ret.rolling(10, min_periods=mp(10)).std().mean(axis=1)
C["cs_dispersion_10d"] = pd.DataFrame({c: cs_disp for c in ret.columns}, index=ret.index)

# 18) SPX vs NDX ratio momentum (broad vs tech leadership)
spx_ndx = px["SPX"] / px["NDX"]
C["spx_ndx_ratio_20d"] = beta_of(ret, spx_ndx.pct_change(), 60).mul(
    spx_ndx.pct_change(20).reindex(ret.index), axis=0)

# 19) 000300 vs HSI ratio momentum (A-share vs HK leadership)
chi_hsi = px["000300.SH"] / px["HSI"]
C["chi_hsi_ratio_20d"] = beta_of(ret, chi_hsi.pct_change(), 60).mul(
    chi_hsi.pct_change(20).reindex(ret.index), axis=0)

# 20) BTC vs ETH relative momentum (crypto internal rotation)
btc_eth = px["BTC"] / px["ETH"]
C["btc_eth_ratio_20d"] = beta_of(ret, btc_eth.pct_change(), 60).mul(
    btc_eth.pct_change(20).reindex(ret.index), axis=0)

# ---- F. regime-conditional family ----
# 21) Momentum gated by VIX level (trend-follow only in calm regime)
vix_level = vix.reindex(ret.index)
C["mom60_vixcalm"] = ((px.shift(5) / px.shift(65) - 1.0)).mul(
    (vix_level < vix_level.rolling(120, min_periods=mp(120)).median()).astype(float).to_frame(
        "g").apply(lambda s: s, axis=0) if False else
    pd.DataFrame({c: (vix_level < vix_level.rolling(120, min_periods=mp(120)).median()).astype(float)
                 for c in ret.columns}, index=ret.index))

# 22) Momentum gated by DXY trend (risk-on USD weakness)
dxy_trend = np.sign(dxy.pct_change(20).reindex(ret.index))
C["mom60_dxyweak"] = ((px.shift(5) / px.shift(65) - 1.0)).mul(
    pd.DataFrame({c: (dxy_trend < 0).astype(float) for c in ret.columns}, index=ret.index))

# 23) Vol-scaled momentum 60d (risk-adjusted trend)
C["sharpe_60d_alt"] = (px.shift(5) / px.shift(65) - 1.0) / vol20.replace(0, np.nan)

# 24) Serial correlation 20d (trend persistence vs reversal)
C["autocorr_1_20d"] = ret.rolling(20, min_periods=mp(20)).apply(
    lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else np.nan, raw=False)

# 25) Overnight gap persistence 20d (institutional flow proxy)
C["gap_persist_20d"] = oc.rolling(20, min_periods=mp(20)).mean()

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
               "recent": pd.Timestamp("2027-09-01")}

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

with open("scripts/miner_2_20281228_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
