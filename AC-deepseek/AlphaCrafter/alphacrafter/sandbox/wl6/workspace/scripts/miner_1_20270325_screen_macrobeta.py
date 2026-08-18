"""miner_1 cycle 2027-03-25: batch screen of NEW candidate factors.

Focus: macro-beta factors using observation-only signals (DXY, USDJPY, EURUSD)
that the current library does NOT use, plus vol-trend / skew / drawdown / range
factors. All data truncated at VISIBLE (2027-03-24, previous completed day).

Gates (15-instrument universe): |daily paper IC| >= 0.0070 AND |ICIR| >= 0.0840
at 10d admission horizon. Library max-abs correlation reported for provenance.
"""
import json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2027-03-24"
H_ADMIT = 10
MIN_IC_DATES = 200
MIN_INSTR = 8

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDJPY', 'EURUSD', 'VIX']


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes = {}
    highs = {}
    lows = {}
    opens = {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        opens[s] = df["open"].astype(float)
    px = pd.DataFrame(closes).dropna(how="all")
    hi = pd.DataFrame(highs).reindex(px.index)
    lo = pd.DataFrame(lows).reindex(px.index)
    op = pd.DataFrame(opens).reindex(px.index)
    return px, hi, lo, op


t0 = time.time()
px, hi, lo, op = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

macro = {m: load_close(m, VISIBLE, INDEX_DIR)["close"].astype(float) for m in MACRO}
for m, s in macro.items():
    print(f"{m}: last={s.iloc[-1]:.2f} n={len(s)}", flush=True)


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def rsum(x, w):
    return x.rolling(w, min_periods=mp(w)).sum()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    var_m = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / var_m


def corr_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w, 2)).corr(mdf)


dxy_r = macro["DXY"].pct_change()
jpy_r = macro["USDJPY"].pct_change()
eur_r = macro["EURUSD"].pct_change()
vix_r = macro["VIX"].pct_change()
us10y_r = px["US10Y"].pct_change()

down = ret.clip(upper=0) * -1
up = ret.clip(lower=0)

C = {}
# --- macro-beta family (DXY / USDJPY / EURUSD / US10Y) ---
C["beta_dxy_60d_neg"] = -beta_of(ret, dxy_r, 60)
C["beta_usdjpy_60d"] = beta_of(ret, jpy_r, 60)
C["beta_eurusd_60d"] = beta_of(ret, eur_r, 60)
C["corr_dxy_60d_neg"] = -corr_of(ret, dxy_r, 60)
C["beta_us10y_60d_neg"] = -beta_of(ret, us10y_r, 60)
# conditional dollar move: -beta_dxy * recent DXY move (fear/risk-on conditioning)
dxy_mom20 = (macro["DXY"] / macro["DXY"].shift(20) - 1.0).reindex(px.index)
jpy_mom20 = (macro["USDJPY"] / macro["USDJPY"].shift(20) - 1.0).reindex(px.index)
vix_mom20 = (macro["VIX"] / macro["VIX"].shift(20) - 1.0).reindex(px.index)
C["dxy_beta_cond_60x20"] = -beta_of(ret, dxy_r, 60) * dxy_mom20
C["jpy_beta_cond_60x20"] = beta_of(ret, jpy_r, 60) * jpy_mom20
# --- vol trend family ---
C["vol_term_10x60"] = rs(ret, 10) / rs(ret, 60).replace(0, np.nan)
C["vol_term_20x120"] = rs(ret, 20) / rs(ret, 120).replace(0, np.nan)
# --- asymmetry / distribution shape ---
C["skew_20d"] = ret.rolling(20, min_periods=12).skew()
C["updown_mean_asym_20d"] = up.rolling(20, min_periods=12).mean() / down.rolling(20, min_periods=12).mean().replace(0, np.nan)
C["updown_vol_asym_20d"] = up.rolling(20, min_periods=12).std() / down.rolling(20, min_periods=12).std().replace(0, np.nan)
# --- short-term autocorrelation (mean-reversion speed) ---
C["ar1_20d"] = ret.rolling(20, min_periods=12).apply(lambda x: x.autocorr() if len(x) > 3 else np.nan, raw=False)
# --- drawdown / range ---
C["dd_from_high_60d"] = px / px.rolling(60, min_periods=mp(60)).max() - 1.0
C["dd_from_high_120d"] = px / px.rolling(120, min_periods=mp(120)).max() - 1.0
C["range_pos_60d"] = (px - lo.rolling(60, min_periods=mp(60)).min()) / (hi.rolling(60, min_periods=mp(60)).max() - lo.rolling(60, min_periods=mp(60)).min()).replace(0, np.nan)
C["hilo_pos_5d"] = (px - lo.rolling(5, min_periods=3).min()) / (hi.rolling(5, min_periods=3).max() - lo.rolling(5, min_periods=3).min()).replace(0, np.nan)
# --- squeeze: recent range compression vs longer vol ---
C["vol_squeeze_20x60"] = ((hi - lo) / px).rolling(20, min_periods=12).mean() / rs(ret, 60).replace(0, np.nan)
# --- gap intensity ---
C["gap_ratio_10d"] = (op / px.shift(1) - 1.0).abs().rolling(10, min_periods=8).mean()

C = {k: v.reindex(px.index) for k, v in C.items()}
print(f"candidates built: {len(C)} in {time.time()-t0:.1f}s", flush=True)

# --- library signal reconstruction (8 effective factors) ---
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret, vix_r, 60) * vix_mom20,
    "down_vol_ratio_20x120": down.rolling(20).std() / down.rolling(120).std(),
    "beta_vix_60d_neg": -beta_of(ret, vix_r, 60),
    "beta_cn10y_60d": beta_of(ret, px["CN10Y"].pct_change(), 60),
    "low_vol_20d": -ret.rolling(20).std(),
}
lib = {k: v.reindex(px.index) for k, v in lib.items()}


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
        return np.nan, np.nan, np.nan, 0
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = m / s if s and math.isfinite(s) and s > 0 else 0.0
    hit = float((ic > 0).mean()) if len(ic) else np.nan
    return m, icir, hit, int(len(ic))


def max_lib_rho(fv, lib_sigs):
    best, arg = 0.0, None
    for name, lsig in lib_sigs.items():
        ic = fast_ic_series(fv, lsig, min_valid=5).dropna()
        if len(ic) < 50:
            continue
        r = float(abs(ic.mean()))
        if r > best:
            best, arg = r, name
    return best, arg


fwd10 = px.shift(-H_ADMIT) / px - 1.0
sub_windows = {
    "full": px.index.min(),
    "2024+": pd.Timestamp("2024-01-01"),
    "2025+": pd.Timestamp("2025-01-01"),
    "2026+": pd.Timestamp("2026-01-01"),
    "2027+": pd.Timestamp("2027-01-01"),
}

print(f"\n{'factor':<26}{'ic':>8}{'icir':>7}{'hit':>6}{'n':>6} {'librho':>7} {'libarg':<20} | 2024+ 2025+ 2026+ 2027+ (ic/icir)", flush=True)
results = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lr, larg = max_lib_rho(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "librho": lr, "libarg": larg, "sub": rec}
    flag = "  <<< GATE" if (abs(m) >= 0.007 and abs(icir) >= 0.084 and n >= MIN_IC_DATES) else ""
    subs = " ".join(f"{wn}:{rec[wn][0]}/{rec[wn][1]}" if rec.get(wn) else f"{wn}:-" for wn in ["2024+", "2025+", "2026+", "2027+"])
    print(f"{name:<26}{m:>8.4f}{icir:>7.3f}{hit:>6.2f}{n:>6d} {lr:>7.3f} {str(larg):<20} | {subs}{flag}", flush=True)

print(f"\nTOTAL {time.time()-t0:.1f}s", flush=True)
json.dump({k: {"ic": v["ic"], "icir": v["icir"], "hit": v["hit"], "n": v["n"],
               "librho": v["librho"], "libarg": v["libarg"],
               "sub": {kk: vv for kk, vv in v["sub"].items()}} for k, v in results.items()},
          open("scripts/miner_1_20270325_screen_macrobeta_results.json", "w"), indent=1)
