"""
miner_1 exploration 2029-03-08 cycle (data visible through 2029-03-07).

Context: live ensemble cs_7f (beta_vix_60d_neg/beta_cn10y_60d/vol_beta_spx_60d/
mom_120d_skip5/vol_of_vol20x60/sign_ewma_60d/down_vol_ratio_20x120). Most
library factors last validated 2026-2028; drift re-check needed at 2029-03.

Goals:
 (1) re-validate library factors (drift check, recent window);
 (2) test NEW candidate families, |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
     15-instrument universe (>=250 IC dates, >=8 valid/date, max abs library
     corr < 0.5, stable in recent sub-windows);
 (3) persist passers with base64:zlib:csv signal artifacts.

Prior-tested (do NOT re-test): see miner_3_20280810_screen_batch.py header list.
This cycle's candidates are distinct from that list.

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2029-03-07"
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

# ---------------- new candidates (distinct from prior batches) ----------------
C = {}
vol5 = rs(ret, 5); vol20 = rs(ret, 20); vol60 = rs(ret, 60)
spx_r = px["SPX"].pct_change()
mom20 = px.pct_change(20)
mom60 = px.pct_change(60)

# A) candle body position: mean of (close-low)/(high-low) over 20d (intraday buying pressure)
rng = (hi - lo).replace(0, np.nan)
C["candle_body_pos_20d"] = ((px - lo) / rng).rolling(20, min_periods=mp(20)).mean()

# B) plain SPX return beta 60d (equity sensitivity)
C["spx_beta_60d"] = beta_of(ret, spx_r, 60)

# C) short relative momentum: 20d return minus cross-sectional mean (demeaned by date)
C["rel_mom_20d"] = mom20.sub(mom20.mean(axis=1), axis=0)

# D) momentum term-structure: mom60 - mom20 (slope of momentum curve)
C["mom_term_spread_60x20"] = mom60 - mom20

# E) short vol spike: vol5 / vol60
C["vol_spike_5x60"] = vol5 / vol60.replace(0, np.nan)

# F) down-day frequency over 60d (fraction of negative daily returns)
C["down_freq_60d"] = (ret < 0).rolling(60, min_periods=mp(60)).mean()

# G) signed streak of same-direction daily closes (5d cap)
def signed_streak(x):
    out = np.zeros(len(x)); s = 0
    for i in range(1, len(x)):
        r = x.iloc[i]
        if np.isnan(r):
            out[i] = 0.0; s = 0
        elif r > 0:
            s = s + 1 if s >= 0 else 1
            out[i] = min(s, 5)
        elif r < 0:
            s = s - 1 if s <= 0 else -1
            out[i] = max(s, -5)
        else:
            out[i] = 0.0
    return pd.Series(out, index=x.index)

C["streak_5d"] = ret.apply(signed_streak)

# H) SPX lead-lag: lagged 1d SPX return (SPX leads cross-section next day)
C["lead_lag_spx_1d"] = pd.DataFrame({c: spx_r.shift(1) for c in px.columns}, index=px.index)

# I) VIX lead-lag, negated: -lagged 1d VIX change (fear spike -> risk-off next day)
C["lead_lag_vix_neg_1d"] = pd.DataFrame({c: (-vixr.shift(1)).reindex(px.index) for c in px.columns}, index=px.index)

# J) rolling Sharpe (60d): mean daily ret / std daily ret
C["sharpe_60d"] = ret.rolling(60, min_periods=mp(60)).mean() / vol60.replace(0, np.nan)

# K) distance from 60d high (negative of drawdown from recent high)
C["dist_high_60d"] = px / px.rolling(60, min_periods=mp(60)).max() - 1.0

# L) correlation with BTC returns 60d (crypto linkage)
C["corr_btc_60d"] = corr_of(ret, px["BTC"].pct_change(), 60)

# M) body ratio: mean of (open-close)/(high-low) over 20d (gap/body direction)
C["body_ratio_20d"] = ((op - px) / rng).rolling(20, min_periods=mp(20)).mean()

# N) volume-return divergence: 20d cum return * sign of 20d volume trend (price/volume confirm)
vol20m = vol.rolling(20, min_periods=mp(20)).mean()
C["vol_price_confirm_20d"] = mom20 * np.sign(vol20m.diff(5))

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
               "recent": pd.Timestamp("2027-06-01"), "last12m": pd.Timestamp("2028-03-08")}

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
        dec[h] = (round(mm, 4), round(ii, 4)) if (mm is not None and np.isfinite(mm)) else None
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
    l12 = r["sub"].get("last12m")
    l12_ok = l12 is not None and abs(l12[0]) >= IC_TH and abs(l12[1]) >= ICIR_TH
    flag = ""
    if not full_ok:
        flag += "FULL_FAIL "
    if rec is not None and not rec_ok:
        flag += "RECENT_WEAK "
    if s27 is not None and not s27_ok:
        flag += "2027_WEAK "
    if l12 is not None and not l12_ok:
        flag += "L12M_WEAK "
    if flag:
        drift_flags.append((name, flag))
    print(f"{name:<26} full={r['ic']:+.4f}/{r['icir']:+.3f} 2027+={s27} recent={rec} last12m={l12} {flag}", flush=True)

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

with open("scripts/miner_1_20290308_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s", flush=True)
