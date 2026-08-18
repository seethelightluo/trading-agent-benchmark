import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-06-16"          # previous completed trading day before current date 2027-06-17
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
vix_move60 = (vix.shift(5) / vix.shift(65) - 1.0)
vix_ma60 = vix.rolling(60, min_periods=30).mean()
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
dxy_r = dxy.pct_change()
usdcny = load_close("USDCNY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdcny_r = usdcny.pct_change()
usdjpy = load_close("USDJPY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdjpy_r = usdjpy.pct_change()
eurusd = load_close("EURUSD", VISIBLE, INDEX_DIR)["close"].astype(float)
eurusd_r = eurusd.pct_change()


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


# ---------------- library signals (8 persisted effective factors) -------------
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

# ---------------- new candidates (defensive / tail-risk / quality) ------------
C = {}

# ---- A) DRAWDOWN / PAIN family ----
roll_max120 = px.rolling(120, min_periods=mp(120)).max()
dd_120 = px / roll_max120 - 1.0
C["ddepth_120d"] = dd_120                              # closeness to 120d high (drawdown depth)
C["ddepth_60d"] = px / px.rolling(60, min_periods=mp(60)).max() - 1.0
# Pain index: average drawdown over past 60d (area under drawdown curve)
def pain_index(pxx, w):
    rmax = pxx.rolling(w, min_periods=mp(w)).max()
    dds = pxx / rmax - 1.0
    return dds.rolling(w, min_periods=mp(w)).mean()
C["pain_60d_neg"] = -pain_index(px, 60)                # high = shallow avg drawdown (defensive)
C["pain_120d_neg"] = -pain_index(px, 120)
# max drawdown depth over window (worst peak-to-trough)
def max_dd(pxx, w):
    rmax = pxx.rolling(w, min_periods=mp(w)).max()
    return (pxx / rmax - 1.0).rolling(w, min_periods=mp(w)).min()
C["maxdd_60d_neg"] = -max_dd(px, 60)                   # high = small worst drawdown
C["maxdd_120d_neg"] = -max_dd(px, 120)
# ulcer index (sqrt of mean squared drawdown)
def ulcer(pxx, w):
    rmax = pxx.rolling(w, min_periods=mp(w)).max()
    dds = (pxx / rmax - 1.0)
    return (dds.pow(2)).rolling(w, min_periods=mp(w)).mean().pow(0.5)
C["ulcer_60d_neg"] = -ulcer(px, 60)
C["ulcer_120d_neg"] = -ulcer(px, 120)

# ---- B) TAIL / DISTRIBUTION family ----
C["skew_60d_neg"] = -ret.rolling(60, min_periods=mp(60)).skew()   # high = positive skew (defensive)
C["skew_120d_neg"] = -ret.rolling(120, min_periods=mp(120)).skew()
C["kurt_60d_neg"] = -ret.rolling(60, min_periods=mp(60)).kurt()   # low kurtosis preferred
# VaR95 negative (small loss at 95% VaR = defensive)
def var95(x, w):
    return x.rolling(w, min_periods=mp(w)).quantile(0.05)
C["var95_60d_neg"] = -var95(ret, 60)                   # high = less negative 5% tail
C["var95_120d_neg"] = -var95(ret, 120)
# semi-deviation (downside risk) negative
downs = ret.clip(upper=0)
def semi_dev(x, w):
    return (x.clip(upper=0).pow(2)).rolling(w, min_periods=mp(w)).mean().pow(0.5)
C["semi_dev_60d_neg"] = -semi_dev(ret, 60)
C["semi_dev_120d_neg"] = -semi_dev(ret, 120)
# upside/downside capture asymmetry: mean up-day vs mean down-day magnitude
def updown_asym(x, w):
    up = x.clip(lower=0)
    dn = x.clip(upper=0)
    upm = up.replace(0, np.nan).rolling(w, min_periods=mp(w)).mean()
    dnm = dn.replace(0, np.nan).rolling(w, min_periods=mp(w)).mean()
    return upm / dnm.abs().replace(0, np.nan)
C["updown_asym_60d"] = updown_asym(ret, 60)
C["updown_asym_120d"] = updown_asym(ret, 120)
# positive-day consistency (fraction of up days)
upday = (ret > 0).astype(float)
C["upday_frac_60d"] = upday.rolling(60, min_periods=mp(60)).mean()
C["upday_frac_120d"] = upday.rolling(120, min_periods=mp(120)).mean()

# ---- C) RISK-ADJUSTED RETURN / QUALITY family ----
mom60 = px.shift(5) / px.shift(65) - 1.0
mom120 = px.shift(5) / px.shift(125) - 1.0
C["sharpe_60d"] = mom60 / (rs(ret, 60) * np.sqrt(60)).replace(0, np.nan)
C["sharpe_120d"] = mom120 / (rs(ret, 120) * np.sqrt(120)).replace(0, np.nan)
# Sortino: momentum / downside dev
C["sortino_60d"] = mom60 / semi_dev(ret, 60).replace(0, np.nan)
C["sortino_120d"] = mom120 / semi_dev(ret, 120).replace(0, np.nan)
# Calmar: momentum / maxdd
C["calmar_120d"] = mom120 / max_dd(px, 120).abs().replace(0, np.nan)
# efficiency ratio (trend strength): |net move| / sum(|moves|)
def eff_ratio(x, w):
    num = (x.shift(1) - x.shift(w + 1)).abs()
    den = x.diff().abs().rolling(w, min_periods=mp(w)).sum()
    return num / den.replace(0, np.nan)
C["eff_ratio_20d"] = eff_ratio(px, 20)
C["eff_ratio_60d"] = eff_ratio(px, 60)

# ---- D) VOL STRUCTURE / REGIME-CONDITIONED DEFENSIVE ----
vol10 = rs(ret, 10)
vol20 = rs(ret, 20)
vol60 = rs(ret, 60)
C["vol_ratio_5x120"] = -rs(ret, 5) / rs(ret, 120).replace(0, np.nan)   # high = calm recently (defensive)
C["vol_ratio_10x120"] = -vol10 / rs(ret, 120).replace(0, np.nan)
# low-vol conditioned on VIX elevated (defensive when stress)
vix_high = (vix > vix_ma60).astype(float).reindex(ret.index)
C["lowvol_cond_vixhigh"] = (-vol20).mul(vix_high, axis=0)
# gold beta (haven linkage)
C["xau_beta_60d"] = beta_of(ret, px["XAU"].pct_change(), 60)
# low beta to equal-weight market (systematic risk avoidance)
mkt_ret = ret.mean(axis=1)
C["mkt_beta_60d_neg"] = -beta_of(ret, mkt_ret, 60)
C["mkt_beta_20d_neg"] = -beta_of(ret, mkt_ret, 20)
# residual vol (idiosyncratic risk) negative
beta60 = beta_of(ret, mkt_ret, 60)
resid = ret - beta60 * mkt_ret.reindex(ret.index)
C["resid_vol_60d_neg"] = -rs(resid, 60)
# downside vol ratio short/long (from library but re-parameterized)
C["down_vol_ratio_10x120"] = -(rs(downs, 10) / rs(downs, 120).replace(0, np.nan))

# ---- E) MACRO-CONDITIONED MOMENTUM (regime gates) ----
mom20 = px.shift(1) / px.shift(21) - 1.0
# momentum only when VIX falling (risk-on confirmation)
vix_fall = (vix_move20 < 0).astype(float).reindex(ret.index)
C["mom20_vixfall"] = mom20 * vix_fall
# momentum only when DXY falling (liquidity expansion)
dxy_fall = (dxy.pct_change(20) < 0).astype(float).reindex(ret.index)
C["mom20_dxyfall"] = mom20 * dxy_fall
# defensive: negative momentum when VIX rising (risk-off flight) -> short risk assets
vix_rise = (vix_move20 > 0).astype(float).reindex(ret.index)
C["negmom20_vixrise"] = (-mom20) * vix_rise
# carry proxy: high yield = US10Y rising? Broadcast rate-momentum
C["us10y_mom20_bcast"] = (px["US10Y"].shift(1) / px["US10Y"].shift(21) - 1.0).reindex(ret.index)
C["us10y_mom60_bcast"] = (px["US10Y"].shift(5) / px["US10Y"].shift(65) - 1.0).reindex(ret.index)
# USDJPY carry proxy beta (JPY weakness = risk-on)
C["usdjpy_beta_60d"] = beta_of(ret, usdjpy_r, 60)
C["usdjpy_mom20_bcast"] = (usdjpy.shift(1) / usdjpy.shift(21) - 1.0).reindex(ret.index)

# ---- F) RANGE / INTRADAY QUALITY ----
rng = (hi - lo) / px.replace(0, np.nan)
C["range_ratio_10x60_neg"] = -rng.rolling(10, min_periods=5).mean() / rng.rolling(60, min_periods=30).mean().replace(0, np.nan)
# close location within day range (buying pressure)
cl_loc = (px - lo) / (hi - lo).replace(0, np.nan)
C["close_loc_20d"] = cl_loc.rolling(20, min_periods=10).mean()
# volume trend (participation)
C["vol_mom20_bcast"] = (vol / vol.rolling(60, min_periods=30).mean()).reindex(ret.index)

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
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'warm':>12s} {'2024+':>12s} {'2026+':>12s} {'2027+':>10s}", flush=True)
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
    print(f"{name:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {wstr:>12s} {rstr}{flag}", flush=True)

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
with open("scripts/miner_2_20270617_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s; results saved", flush=True)
