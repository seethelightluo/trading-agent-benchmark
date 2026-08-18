"""
miner_2 batch screen 2027-12-02 cycle (data visible through 2027-12-01).

Context: live ensemble beta_vix_60d_neg(0.36)/mom_120d_skip5(0.30)/vol_of_vol20x60(0.18)/
low_vol_20d(0.16,dir=-1). Last block 2027-11-18..12-02 was -1.98%, regime flipped to BEAR
(trend -1.29). mom_120d_skip5 WTI call reversed 3 blocks in a row (WTI -27% pnl rate this
block). Feedback: demote/filter momentum, defensive rotation winning.
Prior batches already tested (avoid re-testing): volume-flow, OHLC vol, drawdown/ulcer,
trend t-stat, RSI, eff ratio, skew/kurt (20/60/120), cvar 20/60, beta_dxy/ndx/chi/hsi/
copper/crypto/btc/xau/usdjpy/sprd10y, mom gating on vol, vix_prem_beta, gain_loss_asym,
dd_speed, down_beta_ew, gap_share, corr_ew, park vol 20d, up/down ratios, autocorr 1/2/5.

Goal:
 1) Re-validate the 9 library factors (drift check full / online / 2027+ / last-250d).
 2) Discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
    15-instrument universe (>=250 IC dates, >=8 valid instruments/date, max abs library
    correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
 3) Update library persistence records (last_validated + fresh metrics) for factors that
    still pass; flag/deprecate on drift.

NEW candidate families this cycle:
  A) Regime-conditional momentum (address WTI/energy whipsaw):
     mom120_gated_trend (sign of EW 120d trend), mom60_gated_trend, mom120_ew_filt
     (momentum active only when EW trend > 0), mom120_gated_sma (EW above 120d SMA)
  B) Risk-off / defensive betas:
     vix_up_beta_60d_neg (beta on VIX-up days, negated), beta_dxy_up_60d_neg,
     beta_eurusd_60d (risk-on), beta_us10y_60d (rate beta), rate_cond_beta_60d
     (us10y beta * sign(us10y 60d move)), beta_wti_60d (energy co-movement)
  C) Short-term reversal (whipsaw regime):
     ret5_rev_neg, ret20_rev_neg, ret10_voladj_rev
  D) Long-horizon trend anchors: sma_dist_200d, sma_dist_100d, ema_cross_20x60
  E) Downside risk: down_capture_120d_neg, up_beta_ew_60d, cvar10_q10_neg, maxdd_20d
  F) Price-volume / distribution: vol_ret_corr_20d, vol_ret_corr_60d, parkinson_60d_neg,
     vol_ratio_10x60, kurt_120d_neg, autocorr10_60d, trend_consistency_60d, tail_ratio_60d

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2027-12-01"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
CORR_TH = 0.5
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
dxy = obs["DXY"]; dxyr = dxy.pct_change()
eur = obs["EURUSD"]; eurr = eur.pct_change()
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()


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


# ---------------- library signals (9 persisted factors, recomputed) ----------------
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

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5); vol10 = rs(ret, 10); vol20 = rs(ret, 20)
vol60 = rs(ret, 60); vol120 = rs(ret, 120)
ret5 = px.pct_change(5); ret10 = px.pct_change(10); ret20 = px.pct_change(20)
ret60 = px.pct_change(60); ret120 = px.pct_change(120)
ew_px = px.mean(axis=1)
ew_ret = ew_px.pct_change()
ew_trend60 = (ew_px / ew_px.shift(60) - 1.0)
ew_trend120 = (ew_px / ew_px.shift(120) - 1.0)
mom120 = lib["mom_120d_skip5"]

# A) regime-conditional momentum
C["mom120_gated_trend"] = mom120.mul(ew_trend120.reindex(ret.index).apply(np.sign), axis=0)
C["mom60_gated_trend"] = ret60.mul(ew_trend60.reindex(ret.index).apply(np.sign), axis=0)
C["mom120_ew_filt"] = mom120.where(ew_trend120.reindex(ret.index) > 0)
ew_sma120 = ew_px.rolling(120, min_periods=mp(120)).mean()
C["mom120_gated_sma"] = mom120.mul((ew_px > ew_sma120).astype(float).reindex(ret.index), axis=0)

# B) risk-off / defensive betas
vix_up = vixr.where(vixr > 0)
C["vix_up_beta_60d_neg"] = -beta_of(ret, vix_up, 60)
dxy_up = dxyr.where(dxyr > 0)
C["beta_dxy_up_60d_neg"] = -beta_of(ret, dxy_up, 60)
C["beta_eurusd_60d"] = beta_of(ret, eurr, 60)
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)
us10y_ret60 = us10y / us10y.shift(60) - 1.0
C["rate_cond_beta_60d"] = C["beta_us10y_60d"].mul(us10y_ret60.reindex(ret.index).apply(np.sign), axis=0)
C["beta_wti_60d"] = beta_of(ret, px["WTI"].pct_change(), 60)

# C) short-term reversal
C["ret5_rev_neg"] = -ret5
C["ret20_rev_neg"] = -ret20
C["ret10_voladj_rev"] = -ret10 / vol10.replace(0, np.nan)

# D) long-horizon trend anchors
C["sma_dist_200d"] = px / px.rolling(200, min_periods=mp(200)).mean() - 1.0
C["sma_dist_100d"] = px / px.rolling(100, min_periods=mp(100)).mean() - 1.0
ema20 = px.ewm(span=20, min_periods=mp(20)).mean()
ema60 = px.ewm(span=60, min_periods=mp(60)).mean()
C["ema_cross_20x60"] = ema20 / ema60 - 1.0

# E) downside risk
ew_down = ew_ret.where(ew_ret < 0)
dn_asset = ret.where(ew_ret < 0)
up_asset = ret.where(ew_ret > 0)
C["down_capture_120d_neg"] = -(dn_asset.rolling(120, min_periods=mp(120)).mean() /
                              ew_down.rolling(120, min_periods=mp(120)).mean().replace(0, np.nan))
ew_up = ew_ret.where(ew_ret > 0)
C["up_beta_ew_60d"] = beta_of(up_asset, ew_up, 60)
C["cvar10_q10_neg"] = -ret.rolling(10, min_periods=mp(10)).quantile(0.10)
C["maxdd_20d"] = (px / px.rolling(20, min_periods=mp(20)).max() - 1.0).rolling(20, min_periods=mp(20)).min()

# F) price-volume / distribution
vol_chg = vol.pct_change()
C["vol_ret_corr_20d"] = ret.rolling(20, min_periods=mp(20)).corr(vol_chg)
C["vol_ret_corr_60d"] = ret.rolling(60, min_periods=mp(60)).corr(vol_chg)
C["parkinson_60d_neg"] = -np.sqrt((np.log(hi / lo).clip(lower=1e-12) ** 2).rolling(60, min_periods=mp(60)).mean() / (4 * np.log(2)))
C["vol_ratio_10x60"] = vol10 / vol60.replace(0, np.nan) - 1.0
C["kurt_120d_neg"] = -ret.rolling(120, min_periods=mp(120)).kurt()
C["autocorr10_60d"] = ret.rolling(60, min_periods=mp(60)).corr(ret.shift(10))
C["trend_consistency_60d"] = (np.sign(ret) == np.sign(ret60)).rolling(60, min_periods=mp(60)).mean()
C["tail_ratio_60d"] = ret.rolling(60, min_periods=mp(60)).quantile(0.95) / \
                     ret.rolling(60, min_periods=mp(60)).quantile(0.05).abs().replace(0, np.nan)

print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    common = factor.index.intersection(fwd.index)
    fr = factor.reindex(common).rank(axis=1, pct=True)
    rr = fwd.reindex(common).rank(axis=1, pct=True)
    mask = fr.isna().values | rr.isna().values
    nvalid = (~mask).sum(axis=1)
    F = np.ma.array(fr.values, mask=mask)
    R = np.ma.array(rr.values, mask=mask)
    Fm = F - F.mean(axis=1, keepdims=True)
    Rm = R - R.mean(axis=1, keepdims=True)
    num = (Fm * Rm).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Rm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic = np.ma.filled(ic, np.nan)
    ic[nvalid < min_valid] = np.nan
    return pd.Series(ic, index=common)


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
               "recent": pd.Timestamp("2026-11-01"), "last250": px.index[-1] - pd.Timedelta(days=400)}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'onlineIC':>9s}{'onlineIR':>9s}  {'decay10/20':>11s}", flush=True)
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
    d10 = (dec.get(10) or (None, None))[0]
    d20 = (dec.get(20) or (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    son = rec.get("online", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{son[0] if son else float('nan'):>9.4f}{son[1] if son else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<{CORR_TH}) ---", flush=True)
passers = []
for name, r in results.items():
    if name in lib:
        continue
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < CORR_TH:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ad={r['cov_ad']:.3f} cov_ge8={r['cov_ge8']:.3f} sub={r['sub']}", flush=True)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+/online/recent/last250 |IC|<{IC_TH} or sign flip) ---", flush=True)
lib_flags = {}
for name in lib:
    r = results[name]
    flag = []
    for wname in ("2027+", "online", "recent", "last250"):
        s = r["sub"].get(wname)
        if s and (abs(s[0]) < IC_TH or (s[0] * r["ic"] < 0)):
            flag.append(f"{wname} IC={s[0]:.4f} ICIR={s[1]:.3f}")
    lib_flags[name] = flag
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} -> {'FLAG ' + '; '.join(flag) if flag else 'ok'}", flush=True)

with open("scripts/miner_2_20271202_screen_results.json", "w") as fh:
    json.dump({"results": results, "passers": passers, "lib_flags": lib_flags}, fh, indent=1, default=str)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers}", flush=True)
