"""miner_3 screen (2026-11-19 cycle): trend-quality / reversal / defensive / carry-like candidates.

Regime context: existing ensemble is momentum-heavy (mom120/vol_of_vol/mom10/vix_beta) and
memory shows persistent momentum decay (block_return -2.66pct). Screen themes:
  1) trend quality (Kaufman efficiency) - smooth trends persist
  2) short/medium-term reversal (contrarian regime)
  3) defensive resilience (distance from highs, downside vol)
  4) momentum term-structure / carry proxy
  5) vol-managed momentum
  6) volume expansion

Validation uses closes visible through 2026-11-04 (worldline current_date 2026-11-05).
Gate: |IC|>=0.0070, |ICIR|>=0.0840 at H=10, >=250 IC dates, on the 15-instrument universe.
Also reports recent sub-window IC/ICIR and max abs correlation vs library.
Vectorized implementation (no per-date python loops).
"""
import sys, json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2026-11-04"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes, vols = {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    vol = pd.DataFrame(vols)
    return px, vol


t0 = time.time()
px, vol = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
dxy_r = dxy.pct_change()


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
    return a.rolling(w, min_periods=mp(w, 2)).cov(m) / m.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)


C = {}
# --- 1) trend quality: Kaufman efficiency ratio (net move / total path) ---
for w in (20, 60, 120):
    path = ret.abs().rolling(w, min_periods=mp(w)).sum().replace(0, np.nan)
    C[f"trend_eff_{w}d"] = (px / px.shift(w) - 1.0).abs() / path
for w in (60, 120):
    path = ret.abs().rolling(w, min_periods=mp(w)).sum().replace(0, np.nan)
    C[f"trend_eff_signed_{w}d"] = (px / px.shift(w) - 1.0) / path

# --- 2) reversal ---
C["rev_3d"] = -(px / px.shift(3) - 1.0)
C["rev_5d_skip1"] = -(px.shift(1) / px.shift(6) - 1.0)
C["zrev_20d"] = -(px / rm(px, 20) - 1.0) / rs(ret, 20).replace(0, np.nan)
C["zrev_60d"] = -(px / rm(px, 60) - 1.0) / rs(ret, 60).replace(0, np.nan)
C["rev_10d_skip5"] = -(px.shift(5) / px.shift(15) - 1.0)

# --- 3) defensive / resilience ---
C["dd_20d"] = px / px.rolling(20, min_periods=mp(20)).max() - 1.0
C["dd_60d"] = px / px.rolling(60, min_periods=mp(60)).max() - 1.0
C["dd_120d"] = px / px.rolling(120, min_periods=mp(120)).max() - 1.0
C["low_vol_20d"] = -rs(ret, 20)
C["down_vol_20d"] = -((-ret.clip(upper=0)).rolling(20, min_periods=mp(20)).std())
C["down_vol_ratio_20x120"] = -(rs(ret.clip(upper=0) * -1, 20) / rs(ret.clip(upper=0) * -1, 120).replace(0, np.nan))
C["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)

# --- 4) momentum term structure / carry proxy ---
C["ts_mom_10x120"] = px.pct_change(10) - px.pct_change(120)
C["ts_mom_20x120"] = px.pct_change(20) - px.pct_change(120)
C["ts_mom_60x120"] = px.pct_change(60) - px.pct_change(120)

# --- 5) vol-managed momentum ---
C["vmm_20d"] = px.pct_change(20) / rs(ret, 20).replace(0, np.nan)
C["vmm_60d"] = px.pct_change(60) / rs(ret, 60).replace(0, np.nan)
C["vmm_120d"] = px.pct_change(120) / rs(ret, 120).replace(0, np.nan)

# --- 6) volume expansion (flow) ---
v20 = rsum(vol, 20)
v120 = rsum(vol, 120)
C["vol_exp_20x120"] = v20 / v120.replace(0, np.nan)
upday = (ret > 0).astype(float)
up_vol = (vol * upday).rolling(20, min_periods=mp(20)).sum()
dn_vol = (vol * (1 - upday)).rolling(20, min_periods=mp(20)).sum()
C["vol_imb_20d"] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)

# --- 7) risk-off / DXY linkage ---
C["beta_dxy_120d"] = beta_of(ret, dxy_r, 120)

print(f"candidates: {len(C)} built in {time.time()-t0:.1f}s", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    """Vectorized cross-sectional rank IC per date (Spearman via rank pct + pearson on ranks)."""
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
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = m / s if s and math.isfinite(s) and s > 0 else 0.0
    hit = float((ic > 0).mean()) if len(ic) else np.nan
    return m, icir, hit, int(len(ic))


# library signals for correlation (vectorized: rank-IC of candidate vs library signal)
ret_l = px.pct_change()
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret_l.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret_l, vixr, 60).mul(vix / vix.shift(20) - 1.0, axis=0),
}


def max_lib_corr(fv, lib_sigs):
    best = 0.0
    for fid, lsig in lib_sigs.items():
        ic = fast_ic_series(fv, lsig, min_valid=MIN_INSTR).dropna()
        if len(ic):
            best = max(best, abs(float(ic.mean())))
    return best


fwd10 = px.shift(-H_ADMIT) / px - 1.0
sub_windows = {
    "full": px.index.min(),
    "2024+": pd.Timestamp("2024-01-01"),
    "2025+": pd.Timestamp("2025-01-01"),
    "2026+": pd.Timestamp("2026-01-01"),
}

print(f"\n{'factor':<26}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  libcorr  | recent ic/icir", flush=True)
results = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc = max_lib_corr(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_corr": round(lc, 3), "recent": rec, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v)
    print(f"{name:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {'PASS' if ok else '':<4} {rstr}", flush=True)

# decay + turnover + coverage for gate-passing candidates
print("\n=== DETAIL (gate-passing) ===")
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
for name, r in results.items():
    if abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840 and r["n"] >= MIN_IC_DATES:
        f = r["signal"]
        dec = {}
        for h, fr_ in fwd_all.items():
            ic = fast_ic_series(f, fr_)
            mm, _, _, nn = ic_summary(ic)
            dec[str(h)] = round(mm, 4) if nn > 0 else None
        r["decay"] = dec
        ranks = f.rank(axis=1, pct=True)
        r["turnover_10d_rank"] = round(float(ranks.diff(10).abs().mean().mean()), 3)
        valid = f.notna()
        r["coverage_asset_days"] = round(float(valid.sum().sum()) / float(f.shape[0] * f.shape[1]), 3)
        r["coverage_dates_ge8"] = round(float((valid.sum(axis=1) >= 8).mean()), 3)
        print(f"  {name:<24} decay={dec} turnover={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)

with open("scripts/miner_3_20261119_screen_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
