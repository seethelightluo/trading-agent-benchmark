"""
miner_3 batch screen 2028-08-10 cycle (data visible through 2028-08-09).

Context: 15-factor library; live ensemble (root factor_ensemble.json):
mom_120d_skip5(0.26)/beta_vix_60d_neg(0.24)/vol_beta_spx_60d(0.18)/
sign_ewma_60d(0.16)/down_vol_ratio_20x120(0.10)/low_vol_20d(0.06,dir=-1).

Goals:
 (1) re-validate 15 library factors (drift check incl. 2027+/recent windows);
 (2) test NEW orthogonal candidate families, |IC|>=0.0070 & |ICIR|>=0.0840
     at H=10 on 15-instrument universe (>=250 IC dates, >=8 valid/date,
     max abs library corr < 0.5, stable in recent/2027+ sub-windows);
 (3) persist new passers + refresh passing library factors WITH
     base64:zlib:csv signal artifacts so the post-Miner gate can recover them;
 (4) print drift flags and deprecation candidates.

Already tested in prior batches (do NOT re-test): ATR trend (trend_atr_20x60),
dip_mom60_20d, rel_mom_120d, mom_comp_skip5, xs_win_freq_60d, ref-cond betas
(btc/xau/wti/dxy/yield/usdjpy/yc), safe_haven_rot_20d, vix_high_gate_negbeta,
ret5_rev, cmf_20d, vmm_20d, vol_imb_10/20d, trend_eff_20d, mom20_gated_lowvol,
lowvol_vix_high_cond, resid_mom_60d, range_pos_20d, dist_high_120d, maxdd_20d,
vol_adj_mom_60d, vol_confirm_mom_20d, amihud_20d, skew_neg_60d, kurt_60d,
vol_ratio_20x60, updown_beta_60d, dxy_beta_60d, intraday_mom_20d, ew_beta_60d,
breadth_mom_120d, btc_spill_20d, yc_slope_beta_20d, overnight_mom_20d,
intraday_rev_20d, donchian_pos_20d, rsi_14d, eff_ratio_20d, autocorr_20d,
bb_pos_20d, vol_ratio_10x60, wti_beta_cond_60x20, downside_beta_60d,
corr_chg_20d, gap_freq_20d, volume_ratio_20x60, yield_mom_neg_20d.

NEW candidate families this cycle (16, all distinct from prior tests):
  A) momentum acceleration (20d mom minus its 20d-prior value)  mom_accel_20d
  B) MA60/MA20 trend cross                                       ma_cross_60x20
  C) trend consistency (sign agreement with 20d return)          trend_consistency_20d
  D) breakout frequency (new 20d-high touches)                   breakout_freq_20d
  E) volatility clustering (autocorr of |ret|)                   vol_autocorr_20d
  F) up/down vol asymmetry                                       up_down_vol_ratio_60d
  G) VIX level z-score (fear regime)                             vix_level_z_60d
  H) VIX 20d change (fear easing)                                vix_chg_neg_20d
  I) CN10Y yield momentum                                        cn10y_mom_20d
  J) yield-curve spread (CN10Y-US10Y) momentum                   yc_spread_mom_20d
  K) commodity basket beta (WTI/COPPER/XAU ew)                   commodity_beta_60d
  L) crypto basket beta (BTC/ETH ew)                             crypto_beta_60d
  M) gold beta (plain XAU beta)                                  gold_beta_60d
  N) HSI beta                                                    hsi_beta_60d
  O) lottery premium (max daily ret over 20d)                    lottery_max_ret_20d
  P) crash sensitivity (min daily ret over 20d)                  min_ret_20d

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2028-08-09"
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
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change(); cn10y_r = cn10y.pct_change()


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
wti_r = px["WTI"].pct_change()
xau_r = px["XAU"].pct_change()
hsi_r = px["HSI"].pct_change()

# A) momentum acceleration: 20d momentum minus its value 20d earlier
mom20 = px.pct_change(20)
C["mom_accel_20d"] = mom20 - mom20.shift(20)

# B) MA60/MA20 trend cross
C["ma_cross_60x20"] = rm(px, 60) / rm(px, 20) - 1.0

# C) trend consistency: fraction of last-20d days with same sign as 20d return
C["trend_consistency_20d"] = (np.sign(ret).mul(np.sign(mom20), axis=0)
                              .rolling(20, min_periods=mp(20)).mean())

# D) breakout frequency: fraction of last-20d days closing above prior 20d high
prior_hi = hi.shift(1).rolling(20, min_periods=mp(20)).max()
C["breakout_freq_20d"] = (px > prior_hi).rolling(20, min_periods=mp(20)).mean()

# E) volatility clustering: 20d autocorrelation of |ret|
aret = ret.abs()
C["vol_autocorr_20d"] = aret.rolling(20, min_periods=mp(20)).corr(aret.shift(1))

# F) up/down vol asymmetry (60d)
up_ret = ret.where(ret > 0)
dn_ret = ret.where(ret < 0)
C["up_down_vol_ratio_60d"] = up_ret.rolling(60, min_periods=mp(60)).std() / \
    dn_ret.rolling(60, min_periods=mp(60)).std().replace(0, np.nan)

# G) VIX level z-score vs its 60d mean
vix_mean = vix.rolling(60, min_periods=mp(60)).mean()
vix_std = vix.rolling(60, min_periods=mp(60)).std()
C["vix_level_z_60d"] = ((vix - vix_mean) / vix_std.replace(0, np.nan)).reindex(px.index)

# H) VIX 20d change, negated (fear easing -> risk-on)
C["vix_chg_neg_20d"] = (-vix.pct_change(20)).reindex(px.index)

# I) CN10Y yield momentum
C["cn10y_mom_20d"] = cn10y.pct_change(20).reindex(px.index)

# J) yield-curve spread (CN10Y - US10Y) 20d change
yc_spread = cn10y - us10y
C["yc_spread_mom_20d"] = yc_spread.diff(20).reindex(px.index)

# K) commodity basket beta (WTI/COPPER/XAU equal-weight returns)
comm_r = (wti_r + px["COPPER"].pct_change() + xau_r) / 3.0
C["commodity_beta_60d"] = beta_of(ret, comm_r, 60)

# L) crypto basket beta (BTC/ETH equal-weight returns)
cry_r = (px["BTC"].pct_change() + px["ETH"].pct_change()) / 2.0
C["crypto_beta_60d"] = beta_of(ret, cry_r, 60)

# M) gold beta (plain)
C["gold_beta_60d"] = beta_of(ret, xau_r, 60)

# N) HSI beta (plain)
C["hsi_beta_60d"] = beta_of(ret, hsi_r, 60)

# O) lottery premium: max daily return over 20d
C["lottery_max_ret_20d"] = ret.rolling(20, min_periods=mp(20)).max()

# P) crash sensitivity: min daily return over 20d
C["min_ret_20d"] = ret.rolling(20, min_periods=mp(20)).min()

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
    rec_ok = rec is not None and abs(rec[0]) >= IC_TH and abs(rec[1]) >= ICIR_TH
    s27 = r["sub"].get("2027+")
    s27_ok = s27 is not None and abs(s27[0]) >= IC_TH and abs(s27[1]) >= ICIR_TH
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

with open("scripts/miner_3_20280810_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
