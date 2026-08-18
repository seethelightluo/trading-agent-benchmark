"""
miner_2 batch screen 2027-05-06 cycle (data visible through 2027-05-05).

Context: live ensemble (beta_vix_60d_neg 0.46 / vol_of_vol20x60 0.28 / low_vol_20d 0.26 dir=-1)
posted -0.33% for 20270408-20270422 block. Screener feedback: anchor beta_vix_60d_neg likely
inert (VIX flat since Feb); shift weight toward vol family or NEW ADMITTED REVERSAL factor.
Prior cycle (2027-04-22): ret3_rev IC=0.0251 ICIR=0.0698 (missed ICIR 0.084 gate);
trend_consist_20d passed ICIR but librho 0.529; crypto_beta_60d passed but librho 0.640.

Goal: discover NEW factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the 15-instrument
tradable universe with max abs library rho < 0.5 (orthogonality preference) and robust recent
sub-window IC (2026+/2027+), then PERSIST gate-passers with embedded signal artifacts.
Also re-validate current effective factors (beta_vix_60d_neg, vol_of_vol20x60, low_vol_20d).

Focus this cycle:
  A) REVERSAL (refined): longer-skip 10d/20d reversal, vol-conditional reversal,
     deep-drawdown (120d) reversal, 10d zscore reversal, residual reversal 10d
  B) HAVEN/DEFENSIVE (XAU floor concern): XAU beta, XAU/SPX ratio mom, US10Y mom
     broadcast, US10Y-CN10Y spread mom, XAU/CN300 ratio mom
  C) MACRO FX BETA: DXY beta (neg), USDJPY beta (carry), USDCNY mom, DXY mom10 broadcast
  D) TREND QUALITY: up-day consistency 10d/40d, vol-adj momentum 60d, volume-confirmed mom
  E) VOL STRUCTURE variants: vol_ts 10x60, vol_of_vol 10x60, down-vol ratio 10x60,
     ewm vol ratio 10x60

Gate (H=10, 15-instrument tradable universe): |IC|>=0.0070, |ICIR|>=0.0840,
>=250 IC dates, >=8 valid instruments/date. Library correlation threshold 0.5 for
admission preference (rho<0.5). Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-05-05"
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
vol10 = rs(ret, 10)
vol20 = rs(ret, 20)
vol60 = rs(ret, 60)
vol120 = rs(ret, 120)

# A) REVERSAL family (refined) -------------------------------------------------
C["ret10_rev_skip3"] = -(px.shift(3) / px.shift(13) - 1.0)          # fade 10d winners, skip 3
C["ret20_rev_skip5"] = -(px.shift(5) / px.shift(25) - 1.0)          # 20d reversal, skip 5
C["ret5_rev_volcond"] = -(px.shift(1) / px.shift(6) - 1.0) * (vol20 > vol60).astype(float)  # reversal in turbulent regime
C["ddist_120d_rev"] = -(px / px.rolling(120, min_periods=mp(120)).max() - 1.0)  # deep drawdown snapback
C["zscore10_rev"] = -(px - rm(px, 10)) / rs(px, 10).replace(0, np.nan)
mkt_ret = ret.mean(axis=1)
mkt_beta60 = beta_of(ret, mkt_ret, 60)
resid = ret - mkt_beta60 * mkt_ret.reindex(ret.index)
C["resid_rev10"] = -resid.rolling(10, min_periods=5).sum()
# 3d reversal (continuity from prior cycle)
C["ret3_rev_skip1"] = -(px.shift(1) / px.shift(4) - 1.0)

# B) HAVEN / DEFENSIVE ----------------------------------------------------------
xau_r = px["XAU"].pct_change()
spx_r = px["SPX"].pct_change()
us10y_r = px["US10Y"].pct_change()
cn10y_r = px["CN10Y"].pct_change()
cn300_r = px["000300.SH"].pct_change()
C["xau_beta_60d"] = beta_of(ret, xau_r, 60)                            # gold linkage
C["xau_spx_ratio_mom20"] = (px["XAU"] / px["SPX"]).pct_change(20)      # haven demand broadcast
C["us10y_mom20_bcast"] = (px["US10Y"].shift(1) / px["US10Y"].shift(21) - 1.0)  # rates direction
C["us10y_cn10y_spread_mom20"] = (px["US10Y"] / px["CN10Y"]).pct_change(20)
C["xau_cn300_ratio_mom20"] = (px["XAU"] / px["000300.SH"]).pct_change(20)
C["us10y_mom60_bcast"] = (px["US10Y"].shift(5) / px["US10Y"].shift(65) - 1.0)

# C) MACRO FX BETA -------------------------------------------------------------
C["dxy_beta_60d_neg"] = -beta_of(ret, dxy_r, 60)                       # dollar-weak beneficiaries
C["usdjpy_beta_60d"] = beta_of(ret, usdjpy_r, 60)                      # carry/risk proxy
C["usdcny_mom20_bcast"] = (usdcny / usdcny.shift(20) - 1.0).reindex(ret.index)
C["dxy_mom10_bcast"] = (dxy / dxy.shift(10) - 1.0).reindex(ret.index)
C["dxy_mom60_bcast"] = (dxy.shift(5) / dxy.shift(65) - 1.0).reindex(ret.index)

# D) TREND QUALITY -------------------------------------------------------------
up_day = (ret > 0).astype(float)
C["upday_consist_10d"] = up_day.rolling(10, min_periods=5).mean()
C["upday_consist_40d"] = up_day.rolling(40, min_periods=20).mean()
mom60s = (px.shift(5) / px.shift(65) - 1.0)
C["vol_adj_mom60"] = mom60s / vol20.replace(0, np.nan)                # risk-adjusted momentum
volz = (vol - vol.rolling(60, min_periods=mp(60)).mean()) / vol.rolling(60, min_periods=mp(60)).std().replace(0, np.nan)
C["vol_conf_mom20"] = (px.shift(1) / px.shift(21) - 1.0) * volz       # volume-confirmed momentum

# E) VOL STRUCTURE variants ----------------------------------------------------
C["vol_ts_10x60"] = vol10 / vol60.replace(0, np.nan)
C["vol_of_vol10x60"] = rs(ret, 10).rolling(60, min_periods=mp(60)).std()
down10 = (ret.clip(upper=0) * -1.0)
C["down_vol_ratio_10x60"] = -(rs(down10, 10) / rs(down10, 60).replace(0, np.nan))
ewma_v10 = ret.pow(2).ewm(span=10).mean().pow(0.5)
ewma_v60 = ret.pow(2).ewm(span=60).mean().pow(0.5)
C["ewma_vol_ratio_10x60"] = ewma_v10 / ewma_v60.replace(0, np.nan)
# vol-of-vol relative to level (CV of vol) - curvature
C["vov_cv_20x120"] = rs(ret, 20).rolling(120, min_periods=mp(120)).std() / vol20.replace(0, np.nan)

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
with open("scripts/miner_2_20270506_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved", flush=True)
