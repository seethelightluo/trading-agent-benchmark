"""
miner_3 batch screen 2027-06-17 cycle (data visible through 2027-06-16).

Context: live ensemble (beta_vix_60d_neg 0.46 / vol_of_vol20x60 0.28 / low_vol_20d 0.26 dir=-1)
posted 3 consecutive negative blocks (20270520-20270617); risk-off regime since Feb 2027:
defensives (US10Y, WTI, XAU, 000300) win; high-vol losers (ETH, BTC, SOX, NDX, SX5E, N225, COPPER) drag.
Screener feedback: demote vol_of_vol20x60, admit a DEFENSIVE factor.

Goal: discover NEW orthogonal defensive factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10
on the 15-instrument tradable universe, >=250 IC dates, >=8 valid instruments/date,
max abs library correlation preferably < 0.5; PERSIST gate-passers with signal artifacts
(base64:zlib:csv). Also re-validate the 8 library factors for drift.

Candidate families (avoiding previously tested-and-failed ideas from prior cycles):
  A) drawdown severity: maxdd 60/120d, dd ratio 20x120, ulcer index, dd per unit vol
  B) consistency/quality: win rate, above-MA ratio, sharpe, sortino, downside vol, skew, kurtosis
  C) downside/conditional market beta: downside beta 60/120d, -upside beta, idiosyncratic vol
  D) regime-gated defensives: low-vol gated by VIX level/change, defensive momentum gated by VIX
  E) trend proximity & cross-asset risk appetite: dist to 20d high, ETH/BTC ratio beta,
     vol acceleration (recent stress)

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-06-16"
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
vix_lev = vix / vix.rolling(60, min_periods=30).median()


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
spx_ret = px["SPX"].pct_change()
spx_down = spx_ret.where(spx_ret < 0)
spx_up = spx_ret.where(spx_ret > 0)
vol20 = rs(ret, 20)
vol60 = rs(ret, 60)
vol5 = rs(ret, 5)

# A) drawdown severity
def maxdd(pxx, w):
    rollmax = pxx.rolling(w, min_periods=mp(w)).max()
    dd = pxx / rollmax - 1.0
    return dd.rolling(w, min_periods=mp(w)).min()


dd60 = maxdd(px, 60)
dd120 = maxdd(px, 120)
C["maxdd_60d_neg"] = -dd60
C["maxdd_120d_neg"] = -dd120
C["dd_ratio_20x120_neg"] = -(maxdd(px, 20) / dd120.replace(0, np.nan))
C["ulcer_60d_neg"] = -(dd60.pow(2).rolling(60, min_periods=mp(60)).mean().pow(0.5))
C["dd_vol_ratio_60d_neg"] = -(dd60.abs() / vol60.replace(0, np.nan))

# B) consistency / quality
C["win_rate_60d"] = (ret > 0).rolling(60, min_periods=mp(60)).mean()
ma20 = rm(px, 20)
C["above_ma_ratio_60d"] = (px > ma20).rolling(60, min_periods=mp(60)).mean()
C["sharpe_60d"] = rm(ret, 60) / vol60.replace(0, np.nan)
ddn = down
ddn20 = rs(ddn, 20)
C["sortino_20d"] = rm(ret, 20) / ddn20.replace(0, np.nan)
C["downside_vol_20d_neg"] = -ddn20
C["skew_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).skew()
C["kurt_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).kurt()

# C) downside / conditional market beta + idiosyncratic vol
C["downside_beta_60d"] = beta_of(ret, spx_down.fillna(0.0), 60)
C["downside_beta_120d"] = beta_of(ret, spx_down.fillna(0.0), 120)
C["upside_beta_60d_neg"] = -beta_of(ret, spx_up.fillna(0.0), 60)
resid = ret - beta_of(ret, spx_ret, 60).mul(spx_ret.reindex(ret.index), axis=0)
C["ivol_60d_neg"] = -resid.rolling(60, min_periods=mp(60)).std()

# D) regime-gated defensives (VIX level/change gates)
C["vix_gate_lowvol_20"] = -vol20 * vix_lev.clip(lower=1.0).reindex(ret.index)
C["vix_gate_lowvol_chg"] = -vol20 * np.maximum(vix_move20, 0.0).reindex(ret.index)
C["vix_gate_mom_60"] = (px.shift(5) / px.shift(65) - 1.0) * (vix_lev > 1.0).reindex(ret.index)

# E) trend proximity + risk appetite + vol acceleration
rollmax20 = px.rolling(20, min_periods=mp(20)).max()
C["dist_high_20d"] = -(px / rollmax20 - 1.0)
eth_btc = px["ETH"] / px["BTC"]
C["eth_btc_beta_30d"] = beta_of(ret, eth_btc.pct_change(), 30)
C["vol_accel_5x20_neg"] = -(vol5 / vol20.replace(0, np.nan))
# haven-ratio (XAU/US10Y) beta: assets that track the haven pair in risk-off
haven_ratio = px["XAU"] / px["US10Y"]
C["haven_ratio_beta_40d"] = beta_of(ret, haven_ratio.pct_change(), 40)

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
with open("scripts/miner_3_20270617_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved to scripts/miner_3_20270617_screen_results.json", flush=True)
