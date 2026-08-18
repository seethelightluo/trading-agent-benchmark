"""
miner_3 batch screen 2027-02-25 cycle (data visible through 2027-02-24).

FIX (from prior turn): DataFrame*Series alignment blow-up. `df * s` aligns the
Series against DataFrame columns and yields an all-NaN (n, n+len(s)) frame.
All row-wise broadcasts now use explicit `.mul(s.reindex(df.index), axis=0)`.

Context: live ensemble (beta_vix_60d_neg/beta_cn10y_60d/vol_of_vol/low_vol)
posted -2.48% for 20270211-20270225 block (sideways, crypto/commodity whipsaw:
ETH -19.6%, WTI -10.0%). Trader feedback: recheck vol_of_vol and crypto/commodity
factor exposure. Goal: find NEW orthogonal factors passing |IC|>=0.0070 &
|ICIR|>=0.0840 at H=10 with max abs library rho < 0.5, then PERSIST them with
embedded signal artifacts (base64:zlib:csv) so the post-miner gate can recover
the signal and recompute pairwise correlation.

Gate (H=10, 15-instrument tradable universe): |IC|>=0.0070, |ICIR|>=0.0840,
>=250 IC dates, >=8 valid instruments/date. Library correlation threshold 0.5.
Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2027-02-24"
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
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
dxy_r = dxy.pct_change()
usdcny = load_close("USDCNY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdcny_r = usdcny.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
    """beta of each col of a (DataFrame) vs series m. Explicit axis alignment."""
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
# FIX: explicit row-wise broadcast instead of DataFrame*Series
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
copper_r = px["COPPER"].pct_change()
xau_r = px["XAU"].pct_change()
wti_r = px["WTI"].pct_change()
ndx_r = px["NDX"].pct_change()
spx_r = px["SPX"].pct_change()
btc_r = px["BTC"].pct_change()
eth_r = px["ETH"].pct_change()
gc = (px["XAU"] / px["COPPER"]).pct_change()
cn10y_r = px["CN10Y"].pct_change()
us10y_r = px["US10Y"].pct_change()

# A) re-validation of prior strong candidates
C["copper_beta_45d"] = beta_of(ret, copper_r, 45)
C["copper_beta_60d"] = beta_of(ret, copper_r, 60)
C["copper_beta_75d"] = beta_of(ret, copper_r, 75)
C["gc_ratio_beta_60d"] = beta_of(ret, gc, 60)
bvix60 = -beta_of(ret, vixr, 60)
bc_cn10y = beta_of(ret, px["CN10Y"].pct_change(), 60)
C["bvix_x_cn10y"] = bvix60 * np.sign(bc_cn10y.replace(0, np.nan))
C["skew_20d"] = ret.rolling(20, min_periods=mp(20)).skew()

# B) new orthogonal ideas
C["dxy_beta_60d"] = beta_of(ret, dxy_r, 60)
C["corr_spx_60d"] = ret.rolling(60, min_periods=mp(60)).corr(spx_r.reindex(ret.index))
C["beta_ndx_60d"] = beta_of(ret, ndx_r, 60)
C["mom_60d_skip5"] = (px.shift(5) / px.shift(65) - 1.0)
hlr = (hi - lo) / px.replace(0, np.nan)
C["hilo_pos_20d"] = rm(((px - lo) / (hi - lo).replace(0, np.nan)), 20)
C["kurt_60d"] = ret.rolling(60, min_periods=mp(60)).kurt()
C["range_vol_ratio_20d"] = rm(hlr, 20) / rs(ret, 20).replace(0, np.nan)
C["volume_trend_20x60"] = rm(vol, 20) / rm(vol, 60).replace(0, np.nan)
rollmax20 = px.rolling(20, min_periods=mp(20)).max()
rollmax120 = px.rolling(120, min_periods=mp(120)).max()
dd20 = (px / rollmax20 - 1.0)
dd120 = (px / rollmax120 - 1.0)
C["dd_ratio_20x120"] = dd20 / dd120.replace(0, np.nan)
C["ndxspx_beta_60d"] = beta_of(ret, ndx_r - spx_r, 60)
C["cnus10y_beta_60d"] = beta_of(ret, cn10y_r - us10y_r, 60)
C["upday_frac_60d"] = (ret > 0).astype(float).rolling(60, min_periods=mp(60)).mean()
C["xau_beta_60d"] = beta_of(ret, xau_r, 60)
C["wti_beta_60d"] = beta_of(ret, wti_r, 60)
C["usdcny_beta_60d"] = beta_of(ret, usdcny_r, 60)
C["btc_beta_60d"] = beta_of(ret, btc_r, 60)
# C) crypto/commodity exposure (trader feedback): avoid whipsaw-heavy buckets
C["eth_beta_60d"] = beta_of(ret, eth_r, 60)
C["crypto_avg_beta_60d"] = (beta_of(ret, btc_r, 60) + beta_of(ret, eth_r, 60)) / 2.0
C["comm_avg_beta_60d"] = (beta_of(ret, copper_r, 60) + beta_of(ret, wti_r, 60)) / 2.0
C["gc_ratio_mom_20d"] = (gc.rolling(20, min_periods=mp(20)).mean())
C["xau_wti_ratio_beta_60d"] = beta_of(ret, (px["XAU"] / px["WTI"]).pct_change(), 60)

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
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01")}

results = {}
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'warm':>12s} {'2024+':>12s} {'2025+':>12s} {'2026+':>12s}", flush=True)
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
    valid = f.notna()
    cov_ad = round(float(valid.sum().sum()) / float(f.shape[0] * f.shape[1]), 3)
    cov_d8 = round(float((valid.sum(axis=1) >= MIN_INSTR).mean()), 3)
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_corr": round(lc, 3),
                     "lib_det": det, "recent": rec, "decay": dec,
                     "turnover_10d_rank": round(turn, 3), "coverage_asset_days": cov_ad,
                     "coverage_dates_ge8": cov_d8, "signal": f, "is_library": name in lib}
    ok = abs(m) >= IC_TH and abs(icir) >= ICIR_TH and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v and k != "full")
    w = rec.get("warm")
    wstr = f"{w[0]}/{w[1]}" if w else "-"
    flag = "  <== PASS" if ok else ""
    print(f"{name:<24}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {wstr:>12s} {rstr}{flag}", flush=True)

print("\n=== GATE PASSERS (|IC|>=0.0070 & |ICIR|>=0.0840 & n>=250, full window) ===", flush=True)
gate_pass = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES:
        gate_pass.append(name)
        r["gate_pass"] = True
        print(f"  {name}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} librho={r['lib_corr']:.3f} "
              f"turn={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']} "
              f"decay10={r['decay']['10']} recent={ {k: v for k, v in r['recent'].items() if k != 'full'} }", flush=True)
    else:
        r["gate_pass"] = False

out = {}
for name, r in results.items():
    out[name] = {k: v for k, v in r.items() if k != "signal"}
with open("scripts/miner_3_20270225_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved", flush=True)
