"""
miner_2 batch screen 2027-04-22 cycle (data visible through 2027-04-21).

Context: live ensemble (beta_vix_60d_neg 0.46 / vol_of_vol20x60 0.28 / low_vol_20d 0.26 dir=-1)
posted -0.33% for 20270408-20270422 block. Screener feedback: anchor beta_vix_60d_neg likely
inert (VIX flat since Feb), shift weight toward vol family or a NEW ADMITTED REVERSAL factor.
Prior cycle ret3_rev IC=0.0251 ICIR=0.0698 (missed ICIR 0.084 gate); trend_consist_20d passed
ICIR but librho 0.529; crypto_beta_60d passed but librho 0.640.

Goal: discover NEW factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the 15-instrument
tradable universe with max abs library rho < 0.5 (orthogonality) and robust recent sub-window
IC (especially 2026+/2027+), then PERSIST gate-passers with embedded signal artifacts.

Focus this cycle:
  A) REVERSAL family (screener request): 5d/10d reversal w/ skips, vol-scaled reversal,
     residual (market-orthogonal) reversal, regime-conditional reversal (vol state),
     drawdown-distance reversal, 20d zscore-negated reversal, overnight/gap reversal proxy
  B) Cross-asset flow / ratio momentum: BTC/NDX beta, WTI beta, XAU-COPPER ratio mom,
     XAU/US10Y (haven) ratio mom, BTC/ETH ratio mom, SPX momentum 5d (risk appetite)
  C) Vol-structure continuation (vol family weight shift): vol term structure 20/60,
     vol_of_vol10x30, down/up vol ratio, vol skew, EWMA vol ratio 5/40
  D) Trend-quality: 20d efficiency, MA20/MA60 cross, high-watermark consistency
  E) Macro regime: DXY momentum 20d, USDCNY momentum, USDJPY momentum, VIX level zscore

Gate (H=10, 15-instrument tradable universe): |IC|>=0.0070, |ICIR|>=0.0840,
>=250 IC dates, >=8 valid instruments/date. Library correlation threshold 0.5 for
admission preference (rho<0.5). Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-04-21"
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
usdjpy = load_close("USDJPY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdjpy_r = usdjpy.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    var_m = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / var_m


# ---------------- library signals (8 persisted factors, recomputed) -----------
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
vol20 = rs(ret, 20)
vol60 = rs(ret, 60)
vol10 = rs(ret, 10)

# A) REVERSAL family -----------------------------------------------------------
C["ret5_rev_skip1"] = -(px.shift(1) / px.shift(6) - 1.0)          # fade 5d winners, skip 1
C["ret10_rev_skip2"] = -(px.shift(2) / px.shift(12) - 1.0)        # fade 10d winners, skip 2
C["ret3_rev_skip1"] = -(px.shift(1) / px.shift(4) - 1.0)          # fade 3d winners, skip 1
C["ret5_rev_voladj"] = -(px.shift(1) / px.shift(6) - 1.0) / vol20.replace(0, np.nan)  # vol-scaled
C["ret10_rev_voladj"] = -(px.shift(2) / px.shift(12) - 1.0) / vol20.replace(0, np.nan)

# residual reversal: asset return orthogonal to market, then 5d sum negated
mkt_ret = ret.mean(axis=1)
mkt_beta5 = beta_of(ret, mkt_ret, 60)
resid = ret - mkt_beta5 * mkt_ret.reindex(ret.index)
C["resid_rev5"] = -resid.rolling(5, min_periods=3).sum()

# regime-conditional reversal: reversal only when vol-of-vol is elevated
vov = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
vov_hi = (vov > vov.rolling(120, min_periods=mp(120)).median()).astype(float)
C["rev5_cond_vov"] = -(px.shift(1) / px.shift(6) - 1.0) * vov_hi

# drawdown-distance reversal: fade assets far below 60d high (deep drawdown snapback)
rollmax60 = px.rolling(60, min_periods=mp(60)).max()
C["ddist_60d_rev"] = -(px / rollmax60 - 1.0) * 1.0   # negative distance to high = contrarian long

# 20d zscore negated (mean reversion of price vs its own mean)
C["zscore20_rev"] = -(px - rm(px, 20)) / rs(px, 20).replace(0, np.nan)

# gap/overnight reversal proxy: open-to-close relative to prior close-to-open
C["gap_rev5"] = -(op / px.shift(1) - 1.0).rolling(5, min_periods=3).sum()

# B) Cross-asset flow / ratio momentum -----------------------------------------
btc_r = px["BTC"].pct_change()
eth_r = px["ETH"].pct_change()
xau_r = px["XAU"].pct_change()
copper_r = px["COPPER"].pct_change()
wti_r = px["WTI"].pct_change()
ndx_r = px["NDX"].pct_change()
spx_r = px["SPX"].pct_change()
us10y_r = px["US10Y"].pct_change()
cn300_r = px["000300.SH"].pct_change()

C["btc_ndx_beta_40d"] = beta_of(ret, ndx_r, 40)          # crypto-equity linkage
C["wti_beta_60d"] = beta_of(ret, wti_r, 60)              # oil sensitivity
C["copper_beta_60d"] = beta_of(ret, copper_r, 60)        # growth/cyclical beta
C["xau_copper_ratio_mom20"] = (px["XAU"] / px["COPPER"]).pct_change(20)
C["xau_us10y_ratio_mom20"] = (px["XAU"] / px["US10Y"]).pct_change(20)
C["btc_eth_ratio_mom20"] = (px["BTC"] / px["ETH"]).pct_change(20)
C["spx_mom5"] = (px["SPX"].shift(1) / px["SPX"].shift(6) - 1.0)  # risk appetite broadcast
C["cn300_mom10"] = (px["000300.SH"].shift(1) / px["000300.SH"].shift(11) - 1.0)

# C) Vol-structure continuation -------------------------------------------------
C["vol_ts_20x60"] = vol20 / vol60.replace(0, np.nan)     # term structure (contango)
C["vol_of_vol10x30"] = rs(ret, 10).rolling(30, min_periods=mp(30)).std()
up = ret.clip(lower=0)
dn = (-ret).clip(lower=0)
C["down_up_vol_ratio20"] = rs(dn, 20) / rs(up, 20).replace(0, np.nan)
C["vol_skew20"] = (rs(dn, 20) - rs(up, 20)) / vol20.replace(0, np.nan)
ewma_v5 = ret.pow(2).ewm(span=5).mean().pow(0.5)
ewma_v40 = ret.pow(2).ewm(span=40).mean().pow(0.5)
C["ewma_vol_ratio_5x40"] = ewma_v5 / ewma_v40.replace(0, np.nan)

# D) Trend-quality --------------------------------------------------------------
C["efficiency_20d"] = (px / px.shift(20) - 1.0).abs() / \
                      ret.abs().rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)
ma20 = rm(px, 20)
ma60 = rm(px, 60)
C["ma_cross_20x60"] = (ma20 / ma60 - 1.0)
C["hwm_consist_40d"] = (px / px.rolling(40, min_periods=mp(40)).max())  # near-high persistence

# E) Macro regime ----------------------------------------------------------------
C["dxy_mom20"] = (dxy / dxy.shift(20) - 1.0).reindex(ret.index)
C["usdcny_mom20"] = (usdcny / usdcny.shift(20) - 1.0).reindex(ret.index)
C["usdjpy_mom20"] = (usdjpy / usdjpy.shift(20) - 1.0).reindex(ret.index)
C["vix_level_z"] = ((vix - rm(vix, 60)) / rs(vix, 60).replace(0, np.nan)).reindex(ret.index)

print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    if isinstance(factor, pd.Series):
        factor = pd.DataFrame({c: factor for c in fwd.columns}, index=factor.index)
    if isinstance(fwd, pd.Series):
        fwd = pd.DataFrame({c: fwd for c in factor.columns}, index=fwd.index)
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
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'warm':>12s} {'2025+':>12s} {'2026+':>12s} {'2027+':>10s}", flush=True)
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
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v and k not in ("full", "warm"))
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
with open("scripts/miner_2_20270422_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved", flush=True)
