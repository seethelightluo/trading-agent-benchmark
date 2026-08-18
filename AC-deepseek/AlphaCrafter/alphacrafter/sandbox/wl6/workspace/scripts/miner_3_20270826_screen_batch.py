"""
miner_3 batch screen 2027-08-26 cycle (data visible through 2027-08-25).

Context: live ensemble (beta_vix_60d_neg 0.40 / vol_of_vol20x60 0.24 / mom_120d_skip5 0.20 dir=+1 /
low_vol_20d 0.16 dir=-1). Two consecutive shallow negative blocks (20270729-20270812 -0.44%,
20270812-20270826 -0.19%); regime shifted bull->sideways (trend -0.37 from bull); risk rotation:
WTI/SPX/N225/SX5E strong, SOX/BTC/ETH corrected. Screener feedback: momentum neutral-to-positive,
anchor beta_vix_60d_neg inert (VIX pinned) -> keep watching, consider further anchor trim.

Goal: discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
15-instrument tradable universe, >=250 IC dates, >=8 valid instruments/date, max abs library
correlation < 0.5; PERSIST gate-passers with signal artifacts (base64:zlib:csv).
Also re-validate the 8 library factors for drift (2027+ / online sub-windows).

New candidate families (avoiding prior tested-and-failed ideas):
  A) intraday position quality: CLV (close-low)/(high-low), Parkinson vol, range ratio
  B) vol term-structure slope: vol5/vol60, vol10/vol120, vol20/vol120 (total, not downside)
  C) trend efficiency: Kaufman efficiency ratio 40/60d (20d variant was evicted for librho)
  D) mean reversion / overextension: RSI-14, 3d reversal, 20d z-scored move reversal
  E) cross-sectional relative strength vs equal-weight market (20/60d)
  F) cross-asset factor betas: WTI, COPPER, NDX, BTC, US10Y-change
  G) return quality: AR(1) autocorr, max loss/gain, consecutive up-streak, profit factor
  H) liquidity (volume exists for 9/15): volume trend 10x60, Amihud illiquidity 20d

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-08-25"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
WARM_END = pd.Timestamp("2026-07-15")
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

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

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
vix_move20 = (vix / vix.shift(20) - 1.0)


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


# ---------------- library signals (8 persisted factors, recomputed) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5)
vol10 = rs(ret, 10)
vol20 = rs(ret, 20)
vol60 = rs(ret, 60)
vol120 = rs(ret, 120)
ret5 = px.pct_change(5)
ret10 = px.pct_change(10)
ret20 = px.pct_change(20)
ret60 = px.pct_change(60)

# A) intraday position / range quality
rng = (hi - lo) / px  # daily range ratio
C["clv_10d"] = ((px - lo) / (hi - lo).replace(0, np.nan)).rolling(10, min_periods=mp(10)).mean()
C["clv_20d"] = ((px - lo) / (hi - lo).replace(0, np.nan)).rolling(20, min_periods=mp(20)).mean()
park = np.log(hi / lo).rolling(20, min_periods=mp(20)).std()
C["park_vol_20d_neg"] = -park
C["range_ratio_20d_neg"] = -rng.rolling(20, min_periods=mp(20)).mean()

# B) vol term-structure slope
C["vol_slope_5x60"] = vol5 / vol60.replace(0, np.nan) - 1.0
C["vol_slope_10x120"] = vol10 / vol120.replace(0, np.nan) - 1.0
C["vol_ratio_20x120"] = vol20 / vol120.replace(0, np.nan)

# C) trend efficiency (Kaufman)
C["eff_ratio_40d"] = (px - px.shift(40)).abs() / ret.abs().rolling(40, min_periods=mp(40)).sum().replace(0, np.nan)
C["eff_ratio_60d"] = (px - px.shift(60)).abs() / ret.abs().rolling(60, min_periods=mp(60)).sum().replace(0, np.nan)

# D) mean reversion / overextension
def rsi(pxx, w=14):
    d = pxx.diff()
    up = d.clip(lower=0).rolling(w, min_periods=mp(w)).mean()
    dn = (-d.clip(upper=0)).rolling(w, min_periods=mp(w)).mean()
    rs_ = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs_)


rsi14 = rsi(px, 14)
C["rsi_14_neg"] = -(rsi14 - 50.0) / 50.0
C["rev_3d"] = -(px.pct_change(3))
C["overext_20d_neg"] = -(ret20 / (vol20 * np.sqrt(20)).replace(0, np.nan))

# E) cross-sectional relative strength vs equal-weight market
ew_ret20 = ret20.mean(axis=1)
ew_ret60 = ret60.mean(axis=1)
C["xs_alpha_20d"] = ret20.sub(ew_ret20, axis=0)
C["xs_alpha_60d"] = ret60.sub(ew_ret60, axis=0)

# F) cross-asset factor betas
wti_r = px["WTI"].pct_change()
copper_r = px["COPPER"].pct_change()
ndx_r = px["NDX"].pct_change()
btc_r = px["BTC"].pct_change()
us10y_r = px["US10Y"].pct_change()
C["beta_wti_60d"] = beta_of(ret, wti_r, 60)
C["beta_copper_60d"] = beta_of(ret, copper_r, 60)
C["beta_ndx_60d"] = beta_of(ret, ndx_r, 60)
C["beta_btc_60d"] = beta_of(ret, btc_r, 60)
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)

# G) return quality
C["autocorr_20d"] = ret.rolling(20, min_periods=mp(20)).apply(
    lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)
C["max_loss_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).min()
C["max_gain_20d"] = ret.rolling(20, min_periods=mp(20)).max()
pos = ret.clip(lower=0)
neg_ = (-ret.clip(upper=0))
C["profit_factor_60d"] = pos.rolling(60, min_periods=mp(60)).sum() / neg_.rolling(60, min_periods=mp(60)).sum().replace(0, np.nan)
# consecutive up-day streak (max over last 60d)
upb = (ret > 0).astype(int)
streak = pd.DataFrame(0, index=upb.index, columns=upb.columns)
for c in upb.columns:
    s = upb[c]
    cnt = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    streak[c] = cnt
C["streak_up_60d"] = streak.rolling(60, min_periods=mp(60)).max()

# H) liquidity (volume present for 9/15)
vol10m = vol.rolling(10, min_periods=5).mean()
vol60m = vol.rolling(60, min_periods=30).mean()
C["vol_trend_10x60"] = vol10m / vol60m.replace(0, np.nan)
C["amihud_20d_neg"] = -((ret.abs() / vol.replace(0, np.nan)).rolling(20, min_periods=mp(20)).mean())

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


fwd10 = px.shift(-H_ADMIT) / px - 1.0
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
sub_windows = {"full": None, "warm": WARM_END, "2024+": pd.Timestamp("2024-01-01"),
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01"),
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01")}

results = {}
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+':>10s} {'online':>10s} {'fullwin':>14s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic if wname == "full" else ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    dec = {}
    for h, fh in fwd_all.items():
        ich = fast_ic_series(f, fh)
        mm, ii, _, _ = ic_summary(ich)
        dec[str(h)] = round(mm, 4)
    ranks = f.rank(axis=1, pct=True)
    turn = float(ranks.diff(10).abs().mean().mean()) if len(f) > 20 else np.nan
    cov_ad = float((~f.isna()).sum().sum() / (f.shape[0] * f.shape[1]))
    cov_d8 = float((f.notna().sum(axis=1) >= MIN_INSTR).mean())
    ok = abs(m) >= IC_TH and abs(icir) >= ICIR_TH and n >= MIN_IC_DATES and lc < 0.5
    results[name] = {"ic": round(m, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
                     "n_ic_dates": n, "lib_corr": round(lc, 3), "lib_corr_detail": det,
                     "turnover_10d_rank": round(turn, 3), "coverage_asset_days": round(cov_ad, 3),
                     "coverage_dates_ge8": round(cov_d8, 3), "decay": dec, "recent": rec,
                     "gate_pass": bool(ok)}
    wstr = ""
    if rec.get("2027+"):
        wstr = f"{rec['2027+'][0]:.4f}/{rec['2027+'][1]:.4f}"
    onstr = ""
    if rec.get("online"):
        onstr = f"{rec['online'][0]:.4f}/{rec['online'][1]:.4f}"
    fw = f"{rec['full'][0]:.4f}/{rec['full'][1]:.4f}" if rec.get("full") else "-"
    flag = "  <== PASS" if ok else ""
    print(f"{name:<24}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {turn:>6.3f}  "
          f"{wstr:>10s} {onstr:>10s} {fw:>14s}{flag}", flush=True)

print("\n=== GATE PASSERS (|IC|>=0.0070 & |ICIR|>=0.0840 & n>=250 & librho<0.5, full window) ===", flush=True)
gate_pass = [k for k, r in results.items() if r["gate_pass"]]
for name in gate_pass:
    r = results[name]
    print(f"  {name}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} librho={r['lib_corr']} turn={r['turnover_10d_rank']} "
          f"cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']} decay10={r['decay']['10']} "
          f"recent={ {k: v for k, v in r['recent'].items() if k != 'full'} }", flush=True)

out = {k: {kk: vv for kk, vv in v.items() if kk != "signal"} for k, v in results.items()}
with open("scripts/miner_3_20270826_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved to scripts/miner_3_20270826_screen_results.json", flush=True)
