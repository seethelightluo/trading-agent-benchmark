"""miner_3 screen (2026-11-19 cycle, visible through 2026-11-18): regime-conditional defensive tilts,
cross-asset rate/carry sensitivity via 10Y yield series, volume-imbalance window variants,
downside-vol-managed momentum.

Regime context: live ensemble (mom120/vol_of_vol/mom10/vix_beta) keeps losing (block_return -2.51pct,
memory 20261119); beta_vix_60d_neg was the strongest new factor last cycle (IC 0.0696/ICIR 0.162).
Plans from last cycle: (1) combine beta_vix_60d_neg with a momentum filter (regime-conditional
defensive tilt), (2) carry proxies from US10Y/CN10Y yield series, (3) volume-imbalance variants
at other windows to reduce library correlation.

Gate: |IC|>=0.0070, |ICIR|>=0.0840 at H=10, >=250 IC dates, >=8 valid instruments per date,
15-instrument tradable universe. Also reports max |spearman-rho| vs FULL active library (8 factors),
which approximates the deterministic post-Miner pairwise-correlation gate (threshold 0.5).
"""
import sys, json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2026-11-18"
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
vix_ma60 = vix.rolling(60, min_periods=30).mean()
vix_hi = (vix > vix_ma60).astype(float).reindex(px.index, fill_value=np.nan)
dxy = load_close("DXY", VISIBLE, INDEX_DIR)["close"].astype(float)
dxy_r = dxy.pct_change()
us10y = px["US10Y"]
cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
cn10y_r = cn10y.pct_change()
print(f"us10y 20d chg recent: {(us10y/us10y.shift(20)-1).dropna().tail(3).round(4).tolist()}", flush=True)
print(f"cn10y nonzero chg days: {int((cn10y_r.dropna()!=0).sum())} / {int(cn10y_r.dropna().shape[0])}", flush=True)


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


down = ret.clip(upper=0) * -1
mom120 = px.shift(5) / px.shift(125) - 1.0
mom20 = px.pct_change(20)
bvix60 = -beta_of(ret, vixr, 60)          # base defensive factor from last cycle

C = {}
# --- 1) regime-conditional defensive tilts (interaction beta_vix x regime) ---
C["bvix_mom120_nan"] = bvix60.where(mom120 < 0)
C["bvix_mom120_zero"] = bvix60.where(mom120 < 0, 0.0)
C["bvix_vixhi_nan"] = bvix60.where(vix_hi > 0)
C["bvix_vixhi_zero"] = bvix60.where(vix_hi > 0, 0.0)
C["bvix_mom120_sign"] = bvix60 * np.sign(mom120)          # defensive tilt scaled by momentum sign
# --- 2) diversifier: low correlation to SPX ---
C["div_corr_spx_60d_neg"] = -corr_of(ret, px["SPX"], 60)
# --- 3) rate / carry sensitivity (10Y yield series) ---
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)
C["beta_us10y_neg_60d"] = -beta_of(ret, us10y_r, 60)
us10y_20d_chg = us10y / us10y.shift(20) - 1.0
C["rate_cond_rise_20"] = beta_of(ret, us10y_r, 60).where(us10y_20d_chg > 0)
C["rate_cond_fall_20"] = beta_of(ret, us10y_r, 60).where(us10y_20d_chg < 0)
C["beta_cn10y_60d"] = beta_of(ret, cn10y_r, 60)
# --- 4) downside-vol-managed momentum ---
C["vmm_down_20d"] = mom20 / rs(down, 20).replace(0, np.nan)
C["vmm_down_60d"] = px.pct_change(60) / rs(down, 60).replace(0, np.nan)
# --- 5) volume-imbalance window variants ---
upday = (ret > 0).astype(float)
for w in (10, 60):
    up_vol = (vol * upday).rolling(w, min_periods=mp(w)).sum()
    dn_vol = (vol * (1 - upday)).rolling(w, min_periods=mp(w)).sum()
    C[f"vol_imb_{w}d"] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)
# --- 6) downside vol ratio variant 60x120 ---
C["down_vol_ratio_60x120"] = -(rs(down, 60) / rs(down, 120).replace(0, np.nan))
# --- 7) DXY linkage conditional on risk-off regime ---
C["bdxy_vixhi_nan"] = beta_of(ret, dxy_r, 120).where(vix_hi > 0)

print(f"candidates: {len(C)} built in {time.time()-t0:.1f}s", flush=True)


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
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = m / s if s and math.isfinite(s) and s > 0 else 0.0
    hit = float((ic > 0).mean()) if len(ic) else np.nan
    return m, icir, hit, int(len(ic))


# FULL active library (8 factors currently in factors/ root, incl. last cycle's survivors)
ret_l = px.pct_change()
down_l = ret_l.clip(upper=0) * -1
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret_l.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret_l, vixr, 60).mul((vix / vix.shift(20) - 1.0).reindex(px.index), axis=0),
    "beta_vix_60d_neg": -beta_of(ret_l, vixr, 60),
    "down_vol_ratio_20x120": -(rs(down_l, 20) / rs(down_l, 120).replace(0, np.nan)),
    "low_vol_20d": -rs(ret_l, 20),
}
upday_l = (ret_l > 0).astype(float)
up_vol_l = (vol * upday_l).rolling(20, min_periods=mp(20)).sum()
dn_vol_l = (vol * (1 - upday_l)).rolling(20, min_periods=mp(20)).sum()
lib["vol_imb_20d"] = (up_vol_l - dn_vol_l) / (up_vol_l + dn_vol_l).replace(0, np.nan)


def max_lib_rho(fv, lib_sigs):
    """mean |spearman rho| vs each library signal (mean over dates), return max across library."""
    best = 0.0
    for fid, lsig in lib_sigs.items():
        lsig = lsig.reindex(index=fv.index, columns=fv.columns)
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

print(f"\n{'factor':<24}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  librho  | recent ic/icir", flush=True)
results = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lr = max_lib_rho(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_rho": round(lr, 3), "recent": rec, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v)
    print(f"{name:<24}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lr:>6.3f}  {'PASS' if ok else '':<4} {rstr}", flush=True)

print("\n=== DETAIL (gate-passing) ===", flush=True)
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

# --- re-validation of currently effective library factors (drift check on extended window) ---
print("\n=== LIBRARY RE-VALIDATION (window ..2026-11-18) ===", flush=True)
for fid, lsig in lib.items():
    lsig = lsig.reindex(px.index)
    ic = fast_ic_series(lsig, fwd10)
    m, icir, hit, n = ic_summary(ic)
    rec26 = {}
    ic26 = ic[ic.index >= pd.Timestamp("2026-01-01")]
    mm, ii, _, nn = ic_summary(ic26)
    rec26 = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    print(f"{fid:<24}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  2026+:{rec26}", flush=True)

with open("scripts/miner_3_20261119_screen_regime_carry_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
