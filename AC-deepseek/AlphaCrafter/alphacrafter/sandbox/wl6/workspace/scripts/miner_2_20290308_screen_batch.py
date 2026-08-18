"""
miner_2 batch screen 2029-03-08 cycle (data visible through 2029-03-07).
Fresh candidate factors on the 15-instrument cross-asset universe.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (10d forward, daily cross-sectional rank IC).
Families: higher moments, trend quality (R2), range/OHLC information, overnight/intraday split,
conditional macro betas (DXY/WTI/XAU/US10Y), recovery/consistency dynamics, volume trend,
plus drift re-validation of the 14 persisted library factors and prior near-passers.
"""
import json, time
import numpy as np
import pandas as pd

VISIBLE = "2029-03-07"
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


# ---------------- library signals (14 persisted factors, recomputed) ----------------
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
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
spx_r = px["SPX"].pct_change()
mom60 = px.shift(5) / px.shift(65) - 1.0
mom20 = px.shift(5) / px.shift(25) - 1.0

# ---- A. higher moments / distribution shape ----
k60 = ret.rolling(60, min_periods=mp(60)).kurt()
C["kurt_60d_neg"] = -k60                                  # negative excess kurtosis (stable distributions)
sk60 = ret.rolling(60, min_periods=mp(60)).skew()
C["skew_chg_20x60"] = sk20 - sk60                          # skewness regime shift (freshness of downside risk)

# ---- B. trend quality ----
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

C["r2_trend_60d"] = r2_trend(px, 60)                      # linear-trend quality (R^2)
rollmax60 = px.rolling(60, min_periods=mp(60)).max()
dd60 = px / rollmax60 - 1.0
C["dd_recovery_5d"] = -(dd60.diff(5))                      # drawdown recovery speed (positive = improving)
sign_cons = (ret > 0).astype(float).rolling(20, min_periods=mp(20)).mean()
C["sign_consistency_20d"] = (sign_cons - 0.5).abs()        # directional consistency

# ---- C. range / OHLC information ----
hl = (hi - lo) / px.replace(0, np.nan)
C["range_vol_ratio_20d"] = hl.rolling(20, min_periods=mp(20)).std() / vol20.replace(0, np.nan) - 1.0
range20 = hl.rolling(20, min_periods=mp(20)).mean()
C["range_exp_5d"] = hl / range20.replace(0, np.nan) - 1.0  # recent range expansion vs own 20d mean
C["upper_shadow_20d"] = ((hi - np.maximum(op, px)) / (hi - lo).replace(0, np.nan)).rolling(
    20, min_periods=mp(20)).mean()                          # prior near-passer drift check

# ---- D. overnight / intraday split ----
overnight = op / px.shift(1) - 1.0
intraday = px / op - 1.0
C["overnight_mom_20d"] = overnight.rolling(20, min_periods=mp(20)).sum()   # cumulative overnight drift
C["intraday_mom_20d"] = intraday.rolling(20, min_periods=mp(20)).sum()     # cumulative intraday drift
C["overnight_vol_ratio_20d"] = rs(overnight, 20) / rs(intraday, 20).replace(0, np.nan)
C["gap_cont_5d"] = corr_of(overnight, intraday, 5)          # gap continuation (+) vs reversal (-)

# ---- E. conditional macro betas ----
dxy_down = (dxy_r < 0).astype(float)
C["dxy_beta_cond_60d"] = beta_of(ret, dxy_r, 60).mul(pd.DataFrame({c: dxy_down for c in ret.columns}, index=ret.index))
wti_up = (wti_r > 0).astype(float)
C["wti_beta_cond_60d"] = beta_of(ret, wti_r, 60).mul(pd.DataFrame({c: wti_up for c in ret.columns}, index=ret.index))
xau_up = (xau_r > 0).astype(float)
C["xau_beta_cond_60d"] = beta_of(ret, xau_r, 60).mul(pd.DataFrame({c: xau_up for c in ret.columns}, index=ret.index))
C["us10y_beta_neg_60d"] = -beta_of(ret, us10y_r, 60)        # rate sensitivity (US yield, distinct from CN10Y)
btc_eth = px["BTC"] / px["ETH"]
C["btc_eth_beta_20d"] = beta_of(ret, btc_eth.pct_change(), 20)   # crypto-rotation sensitivity

# ---- F. regime-conditional momentum ----
yld_up = (us10y.pct_change(20) > 0).astype(float)
C["yield_cond_mom60"] = mom60.mul(pd.DataFrame({c: yld_up for c in ret.columns}, index=ret.index))
dxy_fall20 = (dxy.pct_change(20) < 0).astype(float)
C["dxy_fall_mom20"] = mom20.mul(pd.DataFrame({c: dxy_fall20 for c in ret.columns}, index=ret.index))

# ---- G. relative / cross-sectional ----
C["rel_vol_20d"] = vol20 / vol20.median(axis=1).replace(0, np.nan)   # vol vs cross-sectional median
up20 = ret.clip(lower=0).rolling(20, min_periods=mp(20)).std()
dn20 = (ret.clip(upper=0) * -1.0).rolling(20, min_periods=mp(20)).std()
C["updown_vol_ratio_20d"] = up20 / dn20.replace(0, np.nan) - 1.0     # upside/downside vol asymmetry
C["volume_trend_20d"] = rm(vol, 20) / rm(vol, 60).replace(0, np.nan) - 1.0  # volume trend

# ---- H. prior near-passers (drift check from previous cycles) ----
C["sharpe_60d"] = mom60 / vol60.replace(0, np.nan)
ewma60 = px.ewm(span=60, adjust=False).mean()
C["trend_dist_60d"] = px / ewma60 - 1.0
rollmax20 = px.rolling(20, min_periods=mp(20)).max()
C["dd20_neg"] = px / rollmax20 - 1.0
co = (px - op) / op.replace(0, np.nan)
C["intraday_rev_5d"] = -co.rolling(5, min_periods=mp(5)).mean()
downside = ret.where(ret < 0, 0.0)
C["downside_beta_spx_60d"] = beta_of(downside, spx_r.where(spx_r < 0, 0.0), 60)

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
               "recent": pd.Timestamp("2028-04-01")}

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
              f"n={r['n']} turn={r['turn']:.3f} cov_ge8={r['cov_ge8']:.2f} 2027={s27} recent={rec}", flush=True)
    else:
        print(f"    fail: {name} IC={r['ic']:.4f} ICIR={r['icir']:.3f} librho={r['librho']:.3f} "
              f"n={r['n']} cov_ge8={r['cov_ge8']:.2f} 2027={s27} recent={rec}", flush=True)

print(f"\npassers={passers} drift_flags={[f[0] for f in drift_flags]}", flush=True)

with open("scripts/miner_2_20290308_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
