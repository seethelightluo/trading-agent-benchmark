"""miner_3 focused re-check (2026-12-03): faithful gate simulation.

The deterministic post-Miner gate computes pairwise spearman rho from the REAL
signal artifacts in factors/*.json. Current artifact-bearing library members:
beta_cn10y_60d, beta_vix_60d_neg, down_vol_ratio_20x120, low_vol_20d
(mom_*, vol_of_vol, vix_beta_cond were quarantined: no recoverable artifacts;
vol_imb_* were evicted). Candidates must show |mean cross-sectional spearman
rho| < 0.5 vs EVERY artifact-bearing library member to survive, plus the
IC/ICIR admission gate (|IC|>=0.007, |ICIR|>=0.084, n>=250, >=8 instruments).
"""
import json, math, time, zlib, base64
import numpy as np
import pandas as pd

VISIBLE = "2026-12-02"
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
    return pd.DataFrame(closes).dropna(how="all"), pd.DataFrame(vols)


def decode_artifact(path):
    d = json.load(open(path))
    a = d["validation"]["signal_artifact"]
    raw = base64.b64decode(a["data"])
    csv = zlib.decompress(raw).decode()
    df = pd.read_csv(pd.io.common.StringIO(csv), index_col=0, parse_dates=True)
    return df, a


t0 = time.time()
px, vol = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
us10y = px["US10Y"]
cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
cn10y_r = cn10y.pct_change()
usdjpy = load_close("USDJPY", VISIBLE, INDEX_DIR)["close"].astype(float)
usdjpy_r = usdjpy.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    var_m = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / var_m


# ---------- library artifacts (faithful, gate-style) ----------
lib_art = {}
for fid in ["beta_cn10y_60d", "beta_vix_60d_neg", "down_vol_ratio_20x120", "low_vol_20d"]:
    df, a = decode_artifact(f"factors/{fid}.json")
    df.columns = [str(c) for c in df.columns]
    lib_art[fid] = df
    print(f"artifact {fid}: {df.shape} {df.index.min().date()}..{df.index.max().date()}", flush=True)

# ---------- candidate signals ----------
C = {}
C["copper_beta_60d"] = beta_of(ret, px["COPPER"].pct_change(), 60)
C["copper_beta_120d"] = beta_of(ret, px["COPPER"].pct_change(), 120)
cw = (px["COPPER"] / px["WTI"]).pct_change()
C["cpwti_ratio_beta_60d"] = beta_of(ret, cw, 60)
gc = (px["XAU"] / px["COPPER"]).pct_change()
C["gc_ratio_beta_60d"] = beta_of(ret, gc, 60)
be = (px["BTC"] / px["ETH"]).pct_change()
C["btceth_ratio_beta_60d"] = beta_of(ret, be, 60)
C["skew_20d"] = -ret.rolling(20, min_periods=mp(20)).skew()
bvix60 = -beta_of(ret, vixr, 60)
bc_cn = beta_of(ret, cn10y_r, 60)
C["bvix_x_cn10y"] = bvix60 * np.sign(bc_cn.replace(0, np.nan))
# US10Y conditional beta variants (rate factor, currently ACTIVE while CN10Y is flat)
u10_chg20 = us10y / us10y.shift(20) - 1.0
C["us10y_beta_cond_rise20"] = beta_of(ret, us10y_r, 60).where(u10_chg20 > 0)
C["us10y_beta_cond_fall20"] = beta_of(ret, us10y_r, 60).where(u10_chg20 < 0)
C["us10y_beta_60d"] = beta_of(ret, us10y_r, 60)
# copper-beta window variants (to widen rho margin vs beta_vix_60d_neg)
for w in (45, 60, 75, 90):
    C[f"copper_beta_{w}d"] = beta_of(ret, px["COPPER"].pct_change(), w)
# gold/copper ratio beta window variants
gc = (px["XAU"] / px["COPPER"]).pct_change()
for w in (60, 90, 120):
    C[f"gc_ratio_beta_{w}d"] = beta_of(ret, gc, w)
# vol term structure (EWMA-ish blend)
C["vol_term_ewma_10x60"] = ret.ewm(span=10, min_periods=mp(10)).std() / ret.ewm(span=60, min_periods=mp(60)).std().replace(0, np.nan)

print(f"candidates: {list(C.keys())}", flush=True)

# ---------- faithful pairwise rho vs artifacts ----------
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


def faithful_rho(fv, lib_df):
    """mean over dates of cross-sectional spearman corr between candidate and lib artifact."""
    common = fv.index.intersection(lib_df.index)
    fr = fv.reindex(common).rank(axis=1, pct=True)
    lr = lib_df.reindex(common).rank(axis=1, pct=True)
    mask = fr.isna().values | lr.isna().values
    nvalid = (~mask).sum(axis=1)
    F = np.ma.array(fr.values, mask=mask)
    L = np.ma.array(lr.values, mask=mask)
    Fm = F - F.mean(axis=1, keepdims=True)
    Lm = L - L.mean(axis=1, keepdims=True)
    num = (Fm * Lm).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Lm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    rho = np.ma.filled(rho, np.nan)
    rho[nvalid < MIN_INSTR] = np.nan
    rho = rho[~np.isnan(rho)]
    return abs(float(rho.mean())) if len(rho) else np.nan


fwd10 = px.shift(-H_ADMIT) / px - 1.0
sub_windows = {"2024+": pd.Timestamp("2024-01-01"), "2025+": pd.Timestamp("2025-01-01"),
               "2026+": pd.Timestamp("2026-01-01")}
print(f"\n{'factor':<28}{'ic':>8}{'icir':>8}{'hit':>6}{'n':>6}  | rho vs beta_cn10y/beta_vix/downvol/lowvol | recent", flush=True)
res = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    rhos = {fid: faithful_rho(f, lib_df) for fid, lib_df in lib_art.items()}
    rmax = max(rhos.values())
    rstr = "/".join(f"{v:.2f}" for v in rhos.values())
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    res[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "rho": rhos, "rmax": rmax, "recent": rec, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES and rmax < 0.5
    rstr2 = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v)
    print(f"{name:<28}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {rstr}  {'PASS' if ok else '':<4} {rstr2}", flush=True)

print("\n=== DETAIL (would-survive candidates) ===", flush=True)
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
for name, r in res.items():
    if abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840 and r["n"] >= MIN_IC_DATES and r["rmax"] < 0.5:
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

with open("scripts/miner_3_20261203_faithful_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "signal"} for n, r in res.items()}, fh, indent=1, default=str)
print(f"\ndone {time.time()-t0:.1f}s", flush=True)
