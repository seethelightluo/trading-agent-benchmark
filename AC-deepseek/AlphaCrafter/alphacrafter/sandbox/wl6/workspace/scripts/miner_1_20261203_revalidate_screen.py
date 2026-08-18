"""miner_1 comprehensive re-validation + new-candidate screen (2026-12-03 cycle).

Data visible through 2026-12-02 (date.json visible_through). Re-validates the 8
currently-effective library factors with fresh data and screens the full new-
candidate family (previous weekday-grid batch + union-calendar candidates) on
the UNION trading calendar for maximum coverage.

Gate (shared, H=10 on the 15-instrument universe): |IC|>=0.0070, |ICIR|>=0.0840.
Reports warm-up (2020-01..2026-07-15), recent-400d, 2024+/2025+/2026+ windows,
decay, coverage, turnover, and max abs library correlation. Saves signal
artifacts for gate-passing candidates (persistence provenance). Pure research:
no backtest/step/account mutation.
"""
import json, time, hashlib, base64, zlib, io
import numpy as np
import pandas as pd

VISIBLE = "2026-12-02"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
WARM_END = pd.Timestamp("2026-07-15")

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
usdjpy = load_close("USDJPY", VISIBLE, INDEX_DIR)["close"].astype(float)
jpy_r = usdjpy.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)


# ---------------- library factors (persisted formulas) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = ret.rolling(20, min_periods=mp(20)).std().rolling(60, min_periods=mp(60)).std()
vix_move = (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = -beta_of(ret, vixr, 60) * vix_move
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
upday = (ret > 0).astype(float)
up_vol = (vol * upday).rolling(20, min_periods=mp(20)).sum()
dn_vol = (vol * (1 - upday)).rolling(20, min_periods=mp(20)).sum()
lib["vol_imb_20d"] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)

# ---------------- new candidates (union calendar) ----------------
C = {}
# crash-risk / fat-tail
C["skew_60d"] = ret.rolling(60, min_periods=mp(60)).skew()
C["kurt_60d"] = ret.rolling(60, min_periods=mp(60)).kurt()
# trend / SMA structure
C["sma_cross_20_60"] = px.rolling(20, min_periods=mp(20)).mean() / px.rolling(60, min_periods=mp(60)).mean() - 1.0
C["zscore_20d"] = (px - rm(px, 20)) / rs(ret, 20).replace(0, np.nan)
C["dist_high_20d"] = px / px.rolling(20, min_periods=mp(20)).max() - 1.0
C["dist_high_60d"] = px / px.rolling(60, min_periods=mp(60)).max() - 1.0
C["dist_low_20d"] = px / px.rolling(20, min_periods=mp(20)).min() - 1.0
# short vs long vol regime
C["vol_ratio_5_60"] = rs(ret, 5) / rs(ret, 60).replace(0, np.nan)
C["parkinson_20d"] = (np.log(px * 0 + 1))  # placeholder replaced below
# MACD
ema12 = px.ewm(span=12, adjust=False).mean()
ema26 = px.ewm(span=26, adjust=False).mean()
C["macd_12_26"] = (ema12 - ema26) / px
# reversal
C["rev_5d"] = -(px / px.shift(5) - 1.0)
C["rev_20d_skip5"] = -(px.shift(5) / px.shift(25) - 1.0)
# vol-managed momentum
C["ts_mom_60x120"] = px.pct_change(60) - px.pct_change(120)
C["ts_mom120_voladj"] = (px.shift(5) / px.shift(125) - 1.0) / rs(ret, 20).replace(0, np.nan)
C["vmm_60d"] = px.pct_change(60) / rs(ret, 60).replace(0, np.nan)
# close-location value & candle structure (NEW)
Hh = pd.DataFrame({s: load_close(s, VISIBLE)["high"].astype(float) for s in TRADABLE}).reindex(px.index)
Ll = pd.DataFrame({s: load_close(s, VISIBLE)["low"].astype(float) for s in TRADABLE}).reindex(px.index)
Op = pd.DataFrame({s: load_close(s, VISIBLE)["open"].astype(float) for s in TRADABLE}).reindex(px.index)
rng = (Hh - Ll).replace(0, np.nan)
clv = (px - Ll) / rng
C["clv_20d"] = clv.rolling(20, min_periods=mp(20)).mean()
upper_sh = ((Hh - np.maximum(Op, px)) / rng)
C["upper_shadow_20d"] = upper_sh.rolling(20, min_periods=mp(20)).mean()
C["winrate_20d"] = (ret > 0).astype(float).rolling(20, min_periods=mp(20)).mean()
# liquidity / volume structure
amihud = (ret.abs() / vol.replace(0, np.nan))
C["amihud_neg_20d"] = -amihud.rolling(20, min_periods=mp(20)).mean()
v20m = vol.rolling(20, min_periods=mp(20)).mean()
v20s = vol.rolling(20, min_periods=mp(20)).std()
C["vol_z_20d"] = (vol - v20m) / v20s.replace(0, np.nan)
# macro beta factors
C["beta_jpy_60d"] = beta_of(ret, jpy_r, 60)
C["dxy_beta_60x20"] = -beta_of(ret, dxy_r, 60) * (dxy / dxy.shift(20) - 1.0)
C["vix_beta_raw_60"] = beta_of(ret, vixr, 60)
# range/parkinson realised vol
C["range_ratio_20d"] = ((Hh - Ll) / px).rolling(20, min_periods=mp(20)).mean()
park = (np.log(Hh / Ll) ** 2).rolling(20, min_periods=mp(20)).mean() / (4 * np.log(2))
C["parkinson_20d"] = park
# market beta & residual structure
mkt = ret.mean(axis=1, skipna=True)
beta_mkt = ret.rolling(60, min_periods=mp(60, 2)).cov(mkt) / mkt.rolling(60, min_periods=mp(60, 2)).var().replace(0, np.nan)
res_ret = ret - beta_mkt * mkt
C["beta_60d"] = beta_mkt
C["res_mom_20d"] = res_ret.rolling(20, min_periods=mp(20)).sum()
C["idio_vol_20d"] = res_ret.rolling(20, min_periods=mp(20)).std()
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
for name, f in {**lib, **C}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic if wname == "full" else ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "lib_corr": round(lc, 3),
                     "lib_det": det, "recent": rec, "signal": f}
    ok = abs(m) >= IC_TH and abs(icir) >= ICIR_TH and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v and k != "full")
    w = rec.get("warm")
    wstr = f"{w[0]}/{w[1]}" if w else "-"
    print(f"{name:<24}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {wstr:>12s} {rstr}", flush=True)

print("\n=== GATE PASSERS (|IC|>=0.0070 & |ICIR|>=0.0840 & n>=250, full window) ===", flush=True)
gate_pass = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES:
        gate_pass.append(name)
        print(f"  PASS {name:<24} ic={r['ic']:.4f} icir={r['icir']:.4f} n={r['n']} librho={r['lib_corr']}", flush=True)

print("\n=== DETAIL (gate-passing NEW candidates) ===", flush=True)
for name in gate_pass:
    if name in lib:
        continue
    r = results[name]
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
    print(f"  {name:<24} decay={dec} turn10={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)

# save results + signal artifacts for gate passers (provenance)
def sig_artifact(f):
    f = f.reindex(px.index)
    df = f.copy()
    df.index = df.index.strftime("%Y-%m-%d")
    csv = df.to_csv().encode()
    blob = base64.b64encode(zlib.compress(csv, 6)).decode()
    sha = hashlib.sha256(csv).hexdigest()
    return {"format": "base64:zlib:csv", "description": f"Factor signal panel: rows = dates, cols = assets. Shape {df.shape}",
            "columns": list(df.columns), "shape": list(df.shape),
            "n_valid_values": int(df.notna().sum().sum()), "sha256": sha, "data": blob}

out = {}
for name, r in results.items():
    out[name] = {k: v for k, v in r.items() if k != "signal"}
    if name in gate_pass:
        out[name]["signal_artifact"] = sig_artifact(r["signal"])
with open("scripts/miner_1_20261203_revalidate_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s; results saved", flush=True)
