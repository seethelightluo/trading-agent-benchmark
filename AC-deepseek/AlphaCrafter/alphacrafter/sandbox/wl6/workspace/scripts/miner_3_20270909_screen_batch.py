"""
miner_3 batch screen 2027-09-09 cycle (data visible through 2027-09-08).

Context: live ensemble beta_vix_60d_neg(0.40)/vol_of_vol20x60(0.24)/mom_120d_skip5(0.20,dir=+1)/
low_vol_20d(0.16,dir=-1). Last block 20270826-20270909 +1.73% (strongest since July): ETH/BTC/US10Y
wins, SOX/XAU/WTI losses; mom_120d 3-for-3 positive; vol pair kept adding SOX/NDX losers; anchor
finally positive on SPX/US10Y. Regime sideways at block end (trend 0.64).

Goal: discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
15-instrument tradable universe (>=250 IC dates, >=8 valid instruments/date, max abs library
correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
Also re-validate the 8 library factors for drift (full / online / 2027+ sub-windows) and update
their persistence records (EFFECTIVE or DEPRECATED).

New candidate families (avoiding 2027-08-26 tested: CLV/park/range-ratio, vol term-structure slope,
Kaufman eff, RSI-14, 3d reversal, overext z, xs alpha vs EW, WTI/COPPER/NDX/BTC/US10Y betas,
autocorr lag1, max loss/gain, profit factor, streak, vol trend, Amihud):
  A) risk-adjusted momentum (Sharpe 20/60/120, vol-scaled 10d mom)
  B) drawdown dynamics (distance from rolling high 20/120)
  C) idiosyncratic vol vs SPX (60d) and realized skewness 20/60
  D) downside co-movement sensitivity vs market down-days (60d)
  E) up-day ratio (drift consistency) 20/60
  F) skip-lag (5d) autocorrelation 60d
  G) vol-of-vol coefficient of variation 20x60
  H) conditional momentum: 20d mom gated by low/high vol regime
  I) lead-asset momentum gap (own 20d ret - leader 20d ret; leader map below)
  J) VIX-level correlation (level, not return beta)
  K) DXY beta 60d (observation signal)
  L) US10Y-CN10Y spread-change beta 60d

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-09-08"
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
vix = obs["VIX"]; vixr = vix.pct_change(); vix_move20 = (vix / vix.shift(20) - 1.0)
dxy_r = obs["DXY"].pct_change()
us10y = px["US10Y"]; cn10y = px["CN10Y"]; spr_chg = (us10y - cn10y).diff()


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
lib["beta_cn10y_60d"] = beta_of(ret, cn10y.pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5); vol10 = rs(ret, 10); vol20 = rs(ret, 20)
vol60 = rs(ret, 60); vol120 = rs(ret, 120)
ret5 = px.pct_change(5); ret10 = px.pct_change(10); ret20 = px.pct_change(20); ret60 = px.pct_change(60)
spx_r = px["SPX"].pct_change()

# A) risk-adjusted momentum
for w in (20, 60, 120):
    C[f"sharpe_{w}d"] = rm(ret, w) / rs(ret, w).replace(0, np.nan)
C["mom10_voladj"] = ret10 / vol10.replace(0, np.nan)

# B) drawdown dynamics
C["dd_20d"] = px / px.rolling(20, min_periods=mp(20)).max() - 1.0
C["dd_120d"] = px / px.rolling(120, min_periods=mp(120)).max() - 1.0

# C) idiosyncratic vol vs SPX + realized skewness
rho_spx = ret.rolling(60, min_periods=mp(60, 2)).corr(pd.DataFrame({c: spx_r for c in ret.columns}, index=ret.index))
ivol60 = rs(ret, 60) * np.sqrt((1.0 - rho_spx.clip(-1, 1) ** 2).clip(lower=0))
C["ivol_60d_neg"] = -ivol60
C["skew_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).skew()
C["skew_60d_neg"] = -ret.rolling(60, min_periods=mp(60)).skew()

# D) downside co-movement sensitivity (market down-day covariance share)
mkt_down = spx_r * (spx_r < 0).astype(float)
mkt_down = pd.DataFrame({c: mkt_down for c in ret.columns}, index=ret.index)
cov_down = ret.rolling(60, min_periods=mp(60, 2)).cov(mkt_down)
norm = rs(ret, 60) * pd.DataFrame({c: rs(spx_r, 60) for c in ret.columns}, index=ret.index).replace(0, np.nan)
C["down_cov_60d_neg"] = -(cov_down / norm.reindex(ret.index))
cov_all = ret.rolling(60, min_periods=mp(60, 2)).cov(pd.DataFrame({c: spx_r for c in ret.columns}, index=ret.index))
C["down_share_60d"] = cov_down / cov_all.replace(0, np.nan)

# E) up-day ratio
C["up_ratio_20d"] = (ret > 0).rolling(20, min_periods=mp(20)).mean()
C["up_ratio_60d"] = (ret > 0).rolling(60, min_periods=mp(60)).mean()

# F) skip-lag (5d) autocorrelation 60d
C["autocorr5_60d"] = ret.rolling(60, min_periods=mp(60, 2)).corr(ret.shift(5))

# G) vol-of-vol coefficient of variation
C["vol_cv_20x60"] = lib["vol_of_vol20x60"].abs() / vol20.replace(0, np.nan)

# H) conditional momentum (20d mom gated by vol regime)
vol20_med = vol20.rolling(120, min_periods=mp(120)).median()
C["mom20_lowvol"] = ret20 * (vol20 <= vol20_med).astype(float)
C["mom20_highvol"] = ret20 * (vol20 > vol20_med).astype(float)

# I) lead-asset momentum gap (own 20d ret - leader 20d ret)
LEADER = {'SOX': 'NDX', 'ETH': 'BTC', 'HSI': 'SPX', 'N225': 'SPX', 'SX5E': 'SPX',
          '000300.SH': 'HSI', '000688.SH': 'HSI', 'XAU': 'WTI', 'COPPER': 'WTI',
          'WTI': 'COPPER', 'BTC': 'NDX', 'NDX': 'SPX', 'SPX': 'NDX',
          'US10Y': 'SPX', 'CN10Y': 'US10Y'}
lead_gap = pd.DataFrame(index=ret20.index, columns=ret20.columns, dtype=float)
for c in ret20.columns:
    lead_gap[c] = ret20[c] - ret20[LEADER[c]]
C["lead_gap_20d"] = lead_gap

# J) VIX-level correlation (level sensitivity, not return beta)
vixl = vix.reindex(ret.index)
vixl_df = pd.DataFrame({c: vixl for c in ret.columns}, index=ret.index)
C["vix_level_corr_60d"] = ret.rolling(60, min_periods=mp(60, 2)).corr(vixl_df)

# K) DXY beta 60d
C["beta_dxy_60d"] = beta_of(ret, dxy_r, 60)

# L) US10Y-CN10Y spread-change beta 60d
C["beta_sprchg_60d"] = beta_of(ret, spr_chg, 60)

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
    chg = rk.diff(10).abs().mean(axis=1).mean()
    return float(chg) if np.isfinite(chg) else np.nan


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
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'onlineIC':>9s}{'onlineIR':>9s}  {'decay10/20':>11s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    turn = turnover_10d(f)
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
                     "turn": turn, "sub": rec, "decay": dec, "det": det}
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    son = rec.get("online", (None, None))
    print(f"{name:<24}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{son[0] if son else float('nan'):>9.4f}{son[1] if son else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<0.5) ---", flush=True)
passers = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < 0.5:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} sub={r['sub']}", flush=True)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+ or online |IC|<{IC_TH} or sign flip) ---", flush=True)
for name in lib:
    r = results[name]
    s27 = r["sub"].get("2027+")
    son = r["sub"].get("online")
    flag = []
    if s27 and (abs(s27[0]) < IC_TH or (s27[0] * r["ic"] < 0)):
        flag.append(f"2027+ IC={s27[0]:.4f} ICIR={s27[1]:.3f}")
    if son and (abs(son[0]) < IC_TH or (son[0] * r["ic"] < 0)):
        flag.append(f"online IC={son[0]:.4f} ICIR={son[1]:.3f}")
    if r["librho"] >= 0.5:
        flag.append(f"librho={r['librho']:.3f}")
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} -> {'FLAG ' + '; '.join(flag) if flag else 'ok'}", flush=True)

print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers}", flush=True)
