"""
miner_3 batch screen 2027-01-14 cycle (data visible through 2027-01-13).

Context: live ensemble (beta_vix_60d_neg/beta_cn10y_60d/vol_of_vol/low_vol)
posted a -4.61% block in Dec-2026 (commodity/crypto selloff) and the trader has
not advanced since; both 2026-12-31 and 2027-01-14 were safety advances. New
factor families are needed that (a) pass |IC|>=0.0070 & |ICIR|>=0.0840 at H=10,
(b) keep max abs Spearman rho < 0.5 vs the 8 persisted library factors.

A) Re-validation (strong past candidates from 2026-12-03 faithful check, now
   with ~6 more weeks of data):
   - copper_beta_45d / copper_beta_60d : beta of asset returns to COPPER returns
     (global-growth linkage)
   - gc_ratio_beta_60d                 : beta to XAU/COPPER ratio returns
     (safe-haven vs growth rotation)
   - bvix_x_cn10y                      : stacked defensive composite
     (-VIX-beta * sign(CN10Y beta))

B) New orthogonal ideas not previously screened:
   - kurt_60d            : rolling kurtosis of daily returns (tail heaviness)
   - range_vol_ratio_20d : mean daily (high-low)/close vs close-close vol
   - volume_trend_20x60  : 20d/60d mean volume ratio (participation trend)
   - dd_ratio_20x120     : 20d max-drawdown / 120d max-drawdown (crash speed)
   - ndxspx_beta_60d     : beta to NDX-SPX relative return (tech leadership)
   - cnus10y_beta_60d    : beta to CN10Y-US10Y yield-change differential
   - upday_frac_60d      : fraction of up days over 60d (momentum consistency)

Gate (H=10, 15-instrument tradable universe): |IC|>=0.0070, |ICIR|>=0.0840,
>=250 IC dates, >=8 valid instruments/date. Library correlation threshold 0.5.
Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-01-13"
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


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
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
lib["vix_beta_cond_60x20"] = -beta_of(ret, vixr, 60) * vix_move20.reindex(px.index)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
copper_r = px["COPPER"].pct_change()
xau_r = px["XAU"].pct_change()
gc = (px["XAU"] / px["COPPER"]).pct_change()
ndx_r = px["NDX"].pct_change()
spx_r = px["SPX"].pct_change()
cn10y_r = px["CN10Y"].pct_change()
us10y_r = px["US10Y"].pct_change()

C["copper_beta_45d"] = beta_of(ret, copper_r, 45)
C["copper_beta_60d"] = beta_of(ret, copper_r, 60)
C["gc_ratio_beta_60d"] = beta_of(ret, gc, 60)
bvix60 = -beta_of(ret, vixr, 60)
bc_cn10y = beta_of(ret, px["CN10Y"].pct_change(), 60)
C["bvix_x_cn10y"] = bvix60 * np.sign(bc_cn10y.replace(0, np.nan))
C["kurt_60d"] = ret.rolling(60, min_periods=mp(60)).kurt()
rng = (hi - lo) / px.replace(0, np.nan)
C["range_vol_ratio_20d"] = rm(rng, 20) / rs(ret, 20).replace(0, np.nan)
C["volume_trend_20x60"] = rm(vol, 20) / rm(vol, 60).replace(0, np.nan)
rollmax20 = px.rolling(20, min_periods=mp(20)).max()
rollmax120 = px.rolling(120, min_periods=mp(120)).max()
dd20 = (px / rollmax20 - 1.0)
dd120 = (px / rollmax120 - 1.0)
C["dd_ratio_20x120"] = dd20 / dd120.replace(0, np.nan)
C["ndxspx_beta_60d"] = beta_of(ret, ndx_r - spx_r, 60)
C["cnus10y_beta_60d"] = beta_of(ret, cn10y_r - us10y_r, 60)
C["upday_frac_60d"] = (ret > 0).astype(float).rolling(60, min_periods=mp(60)).mean()

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
print(f"\n{'name':<22}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'warm':>12s} {'2024+':>12s} {'2025+':>12s} {'2026+':>12s}", flush=True)
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
                     "coverage_dates_ge8": cov_d8, "signal": f}
    ok = abs(m) >= IC_TH and abs(icir) >= ICIR_TH and n >= MIN_IC_DATES
    rstr = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in rec.items() if v and k != "full")
    w = rec.get("warm")
    wstr = f"{w[0]}/{w[1]}" if w else "-"
    flag = "  <== PASS" if ok else ""
    print(f"{name:<22}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {wstr:>12s} {rstr}{flag}", flush=True)

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
with open("scripts/miner_3_20270114_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s; results saved to scripts/miner_3_20270114_screen_results.json", flush=True)
