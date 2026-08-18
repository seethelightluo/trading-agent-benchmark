"""
miner_2 batch screen 2027-09-23 cycle (data visible through 2027-09-22).

Context: live ensemble beta_vix_60d_neg(0.40)/vol_of_vol20x60(0.24)/mom_120d_skip5(0.20,dir=+1)/
low_vol_20d(0.16,dir=-1). Recent blocks: 20270826-20270909 +1.73% (ETH/BTC/US10Y wins), mom_120d
3-for-3 positive; vol pair mixed; VIX anchor inert since Feb; regime sideways.

Goal:
 1) Re-validate the 8 library factors (drift check across full / online / 2027+ windows).
 2) Discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
    15-instrument tradable universe (>=250 IC dates, >=8 valid instruments/date, max abs library
    correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
 3) Explicitly report # dates and # instruments used (small cross-asset universe is intentional).

New candidate families (avoiding 2027-08-26 tested: CLV/park/range-ratio, vol term-structure slope,
Kaufman eff, RSI-14, 3d reversal, overext z, xs alpha vs EW, single-asset betas, autocorr lag1,
max loss/gain, profit factor, streak, vol trend, Amihud; and 2027-09-09 tested: sharpe 20/60/120,
mom10_voladj, dd_20d/120d, ivol_60d_neg, skew_20d/60d_neg, down_cov/down_share, up_ratio, autocorr5,
vol_cv, mom20_low/highvol, lead_gap_20d, vix_level_corr, beta_dxy_60d, beta_sprchg_60d):
  A) volume-flow: corr(ret, dvol, 20d); volume z-score 20d vs 60d baseline
  B) MA trend structure: MA20/MA60 - 1; close/MA60 - 1; close/MA120 - 1
  C) overnight vs intraday momentum (gap vs session), vol-normalized 20d
  D) gain-loss asymmetry: up-vol/down-vol 20d
  E) momentum acceleration: ret60/ret120 - 1 (rate of change of trend)
  F) conditional rate beta: beta to US10Y chg * sign(20d US10Y move) (rate-regime sensitivity)
  G) implied/realized premium beta: beta to d(VIX/SPX_rvol20)
  H) drawdown speed: 60d drawdown scaled by its own duration (how fast the drop happened)
  I) 250d high proximity: close/rolling_max(close,250) - 1 (long-cycle trend)
  J) up-path efficiency: |ret20| / sum(abs(daily ret)) gated on up moves (trend smoothness up)
  K) dual-momentum blend: 0.5*mom120 + 0.5*(MA20/MA60-1) (combo)

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2027-09-22"
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
vix = obs["VIX"]; vixr = vix.pct_change(); vix_move20 = (vix / vix.shift(20) - 1.0)
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
us10y_m20 = us10y / us10y.shift(20) - 1.0


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
vol_chg = vol.pct_change()

# A) volume-flow
vol_chg_clean = vol_chg.replace([np.inf, -np.inf], np.nan)
C["volflow_corr_20d"] = ret.rolling(20, min_periods=mp(20, 2)).corr(vol_chg_clean)
vol_base = vol.rolling(60, min_periods=mp(60)).mean().replace(0, np.nan)
C["vol_z_20d"] = vol / vol_base - 1.0

# B) MA trend structure
ma20 = rm(px, 20); ma60 = rm(px, 60); ma120 = rm(px, 120)
C["ma_cross_20x60"] = ma20 / ma60 - 1.0
C["dist_ma60"] = px / ma60 - 1.0
C["dist_ma120"] = px / ma120 - 1.0

# C) overnight vs intraday momentum (vol-normalized 20d cumulative)
gap = (op / px.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
intra = (px / op - 1.0).replace([np.inf, -np.inf], np.nan)
C["gap_mom_20d"] = gap.rolling(20, min_periods=mp(20)).sum() / vol20.replace(0, np.nan)
C["intra_mom_20d"] = intra.rolling(20, min_periods=mp(20)).sum() / vol20.replace(0, np.nan)

# D) gain-loss asymmetry 20d (up-vol / down-vol)
up = ret.clip(lower=0); dn = (ret.clip(upper=0) * -1.0)
C["gain_loss_asym_20d"] = rs(up, 20) / rs(dn, 20).replace(0, np.nan)

# E) momentum acceleration: ret60/ret120 - 1 (signed ratio)
ret120 = px.pct_change(120)
C["mom_accel_60x120"] = (ret60 / ret120.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

# F) conditional rate beta: beta to US10Y daily chg * sign(20d US10Y move)
b10 = beta_of(ret, us10y_r, 60)
C["rate_beta_cond_60x20"] = b10.mul(us10y_m20.reindex(ret.index).apply(np.sign), axis=0)

# G) implied/realized premium beta: beta to d(VIX / SPX rvol20)
spx_rvol20 = rs(px["SPX"].pct_change(), 20)
prem = (vix / spx_rvol20.reindex(vix.index).replace(0, np.nan)).diff()
C["vix_prem_beta_60d"] = beta_of(ret, prem, 60)

# H) drawdown speed: 60d drawdown scaled by its own duration (drop depth per day in drawdown)
hh60 = px.rolling(60, min_periods=mp(60)).max()
dd60 = px / hh60 - 1.0
days_since_high = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for c in px.columns:
    ch = px[c]
    run = pd.Series(np.nan, index=ch.index)
    cur = np.nan
    for i in range(len(ch)):
        v = ch.iloc[i]
        if np.isnan(v):
            cur = np.nan
        else:
            cur = 0 if (np.isnan(cur) or v >= hh60[c].iloc[i] * 0.9999) else cur + 1
        run.iloc[i] = cur
    days_since_high[c] = run
C["dd_speed_60d"] = (dd60 / days_since_high.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# I) 250d high proximity
C["high_prox_250d"] = px / px.rolling(250, min_periods=mp(250)).max() - 1.0

# J) up-path efficiency (trend smoothness on up-moves)
absret = ret.abs()
C["up_eff_20d"] = ret20.abs() / absret.rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)

# K) dual-momentum blend: 0.5*mom120_skip5 + 0.5*ma_cross_20x60
C["dual_mom_120x60"] = 0.5 * lib["mom_120d_skip5"] + 0.5 * C["ma_cross_20x60"]

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


def coverage(f):
    fv = f.notna()
    cov_asset_days = float(fv.values.mean()) if fv.size else np.nan
    n_valid = fv.sum(axis=1)
    cov_dates_ge8 = float((n_valid >= 8).mean())
    return cov_asset_days, cov_dates_ge8


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
    ca, cd = coverage(f)
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
                     "turn": turn, "cov_asset_days": ca, "cov_dates_ge8": cd,
                     "sub": rec, "decay": dec, "det": det}
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    son = rec.get("online", (None, None))
    print(f"{name:<24}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{son[0] if son else float('nan'):>9.4f}{son[1] if son else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<{CORR_TH}) ---", flush=True)
passers = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < CORR_TH:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov={r['cov_asset_days']:.3f}/{r['cov_dates_ge8']:.3f} sub={r['sub']}", flush=True)
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
    if r["librho"] >= CORR_TH:
        flag.append(f"librho={r['librho']:.3f}")
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} -> {'FLAG ' + '; '.join(flag) if flag else 'ok'}", flush=True)

with open(f"scripts/miner_2_20270923_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers} (results saved)", flush=True)
