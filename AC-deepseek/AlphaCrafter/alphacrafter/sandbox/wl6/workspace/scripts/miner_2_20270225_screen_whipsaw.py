import json, math, time
import numpy as np
import pandas as pd

VISIBLE = "2027-02-24"
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
    closes = {s: load_close(s, cutoff)["close"].astype(float) for s in TRADABLE}
    px = pd.DataFrame(closes).dropna(how="all")
    return px


t0 = time.time()
px = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
print(f"vix_last={vix.iloc[-1]:.1f} vix_ma60_last={vix.rolling(60,min_periods=30).mean().iloc[-1]:.1f}", flush=True)


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
up = ret.clip(lower=0)
sign = np.sign(ret)

crypto = (px["BTC"] + px["ETH"]) / 2.0
crypto_r = crypto.pct_change()
comm = (px["XAU"] + px["COPPER"] + px["WTI"]) / 3.0
comm_r = comm.pct_change()
eq = px[["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"]].mean(axis=1)
eq_r = eq.pct_change()

C = {}
# 1) whipsaw: fraction of days with sign change vs 5d ago (reversal frequency)
C["whipsaw_10d"] = (sign != sign.shift(5)).rolling(10, min_periods=8).mean()
C["whipsaw_20d"] = (sign != sign.shift(5)).rolling(20, min_periods=14).mean()
# 2) trend efficiency (net move / gross move)
C["eff_ratio_10d"] = (px - px.shift(10)).abs() / rsum(ret.abs(), 10).replace(0, np.nan)
C["eff_ratio_20d"] = (px - px.shift(20)).abs() / rsum(ret.abs(), 20).replace(0, np.nan)
# 3) position within recent range
def range_pos(px, w):
    hi = px.rolling(w, min_periods=mp(w)).max()
    lo = px.rolling(w, min_periods=mp(w)).min()
    return (px - lo) / (hi - lo).replace(0, np.nan)
C["range_pos_10d"] = range_pos(px, 10)
C["range_pos_20d"] = range_pos(px, 20)
# 4) crypto sensitivity (60d beta to BTC/ETH basket)
C["crypto_beta_60d"] = beta_of(ret, crypto_r, 60)
C["crypto_corr_60d"] = corr_of(ret, crypto_r, 60)
# 5) commodity sensitivity
C["comm_beta_60d"] = beta_of(ret, comm_r, 60)
# 6) equity beta minus commodity beta (rotation tilt)
C["eqm_comm_beta_spread_60d"] = beta_of(ret, eq_r, 60) - beta_of(ret, comm_r, 60)
# 7) short/long vol ratio (vol clustering)
C["vol_ratio_5x60"] = rs(ret, 5) / rs(ret, 60).replace(0, np.nan)
# 8) z-scored short reversal (20d return scaled by 20d vol)
C["rev20_z"] = -(px - px.shift(20)) / (rs(ret, 20) * px.shift(20)).replace(0, np.nan)
# 9) down-day ratio and max down-run length (per column)
downflag = (ret < 0).astype(float)
C["down_days_ratio_20d"] = downflag.rolling(20, min_periods=14).mean()
# 10) dispersion of recent daily returns (kurtosis-like concentration)
C["kurt_20d"] = ret.rolling(20, min_periods=12).kurt()
# 11) conditional crypto whipsaw: crypto beta * whipsaw
C["crypto_beta_x_whipsaw20"] = C["crypto_beta_60d"] * C["whipsaw_20d"]
# 12) VIX-sensitivity of daily range (fear persistence)
C["vix_beta_60d"] = beta_of(ret, vixr, 60)
# 13) gap between 20d and 60d trend (acceleration)
C["mom_accel_20x60"] = (px.shift(5) / px.shift(25) - 1.0) - (px.shift(5) / px.shift(65) - 1.0)
# 14) smoothness of cumulative return path (normalized abs deviation from straight line)
def straightness(px, w):
    start = px.shift(w)
    net = px - start
    ideal = start + net * (np.arange(len(px))[:, None] / w)
    dev = (px - ideal).abs().rolling(w, min_periods=mp(w)).mean()
    return dev / net.abs().replace(0, np.nan)
C["straightness_20d"] = straightness(px, 20)

C = {k: v for k, v in C.items() if v is not None}
print(f"candidates built: {len(C)} in {time.time()-t0:.1f}s", flush=True)


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


ret_l = ret
down_l = ret_l.clip(upper=0) * -1
lib = {
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret_l.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -beta_of(ret_l, vixr, 60) * (vix / vix.shift(20) - 1.0),
    "down_vol_ratio_20x120": down_l.rolling(20).std() / down_l.rolling(120).std(),
    "beta_vix_60d_neg": -beta_of(ret_l, vixr, 60),
    "beta_cn10y_60d": beta_of(ret_l, px["CN10Y"].pct_change(), 60),
    "low_vol_20d": -ret_l.rolling(20).std(),
}
lib = {k: v.reindex(px.index) for k, v in lib.items()}


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

print(f"\n{'factor':<28}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  librho  vs  | 2024+ 2025+ 2026+ 2027+ (ic/icir)", flush=True)
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
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_rho": round(lr, 3),
                     "lib_arg": larg, "recent": rec, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v)
    print(f"{name:<28}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lr:>6.3f} {str(larg):<18} {'PASS' if ok else '':<4} {rstr}", flush=True)

print("\n=== DETAIL (gate-passing candidates) ===", flush=True)
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
        print(f"  {name:<28} decay={dec} turnover={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)

print("\n=== LIBRARY RE-VALIDATION (..2027-02-24) ===", flush=True)
for fid, lsig in lib.items():
    lsig = lsig.reindex(px.index)
    ic = fast_ic_series(lsig, fwd10)
    m, icir, hit, n = ic_summary(ic)
    rec = {}
    for wname, wstart in sub_windows.items():
        if wname == "full":
            continue
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    print(f"{fid:<26}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {rec}", flush=True)

with open("scripts/miner_2_20270225_screen_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in results.items()},
              fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
