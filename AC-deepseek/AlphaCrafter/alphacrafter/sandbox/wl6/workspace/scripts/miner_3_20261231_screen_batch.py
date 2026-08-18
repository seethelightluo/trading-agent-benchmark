"""
miner_3 batch screen 2026-12-31 cycle.

Motivation: last live block (2026-12-03..12-17) lost -4.61% driven by
commodity/crypto selloff hurting the beta_vix_60d_neg overweight. Re-check
factor directions and screen a NEW candidate family with low expected
correlation to the 8-factor library:

  - eff_ratio_60d     : Kaufman efficiency ratio |c/c[-60]-1| / sum(|r|,60) (trend strength)
  - trend_r2_60d      : R^2 of log-price OLS fit over 60d (trend quality)
  - autocorr_10d      : lag-1 return autocorrelation over 10d (persistence vs reversal)
  - obv_slope_20d     : On-Balance-Volume 20d slope / volume (volume-confirmed trend)
  - crypto_beta_60d   : beta of asset returns to BTC returns (risk-appetite linkage)
  - bond_beta_60d     : beta to US10Y return (risk-off hedge linkage, negated for defensive)
  - gold_beta_60d     : beta to XAU return (safe-haven linkage)
  - roll_sharpe_60d   : 60d return / 60d vol (risk-adjusted trend)
  - stoch_pos_60d     : (close-min60)/(max60-min60) (overbought/oversold position)
  - maxdd_60d         : rolling 60d max drawdown (crash sensitivity)
  - vix_gate_mom10    : 10d momentum gated by VIX below 120d median (calm-regime trend)
  - up_down_ratio_20d : mean(up)/mean(|down|) over 20d (return asymmetry)

Gate (H=10, 15-instrument universe): |IC|>=0.0070, |ICIR|>=0.0840, n>=250.
Data visible through 2026-12-30. Pure research; no backtest/step/account mutation.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2026-12-30"
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
    closes, vols = {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    vol = pd.DataFrame(vols)
    return px, vol


px, vol = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()
us10y = px["US10Y"].astype(float)
us10y_r = us10y.pct_change()
xau = px["XAU"].astype(float)
xau_r = xau.pct_change()
btc = px["BTC"].astype(float)
btc_r = btc.pct_change()
eth = px["ETH"].astype(float)


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


# ---------------- library signals (8 persisted factors, recomputed) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
vix_move20 = (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = -beta_of(ret, vixr, 60) * vix_move20.reindex(px.index)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
abs_ret = ret.abs()
C["eff_ratio_60d"] = (px / px.shift(60) - 1.0).abs() / abs_ret.rolling(60, min_periods=mp(60)).sum().replace(0, np.nan)

logp = np.log(px.replace(0, np.nan))


def trend_r2(series, w=60):
    out = pd.Series(np.nan, index=series.index)
    for i in range(w - 1, len(series)):
        seg = series.iloc[i - w + 1:i + 1].dropna()
        if len(seg) < max(20, w // 2):
            continue
        x = np.arange(len(seg))
        y = seg.values
        if np.ptp(y) == 0 or np.all(~np.isfinite(y)):
            continue
        c = np.corrcoef(x, y)[0, 1]
        out.iloc[i] = c * c
    return out


C["trend_r2_60d"] = logp.apply(lambda s: trend_r2(s, 60))

r1 = ret.shift(1)
C["autocorr_10d"] = ret.rolling(10, min_periods=mp(10)).corr(r1)

obv = (np.sign(ret) * vol).fillna(0).cumsum()
C["obv_slope_20d"] = (obv - obv.shift(20)) / vol.rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)

C["crypto_beta_60d"] = beta_of(ret, btc_r, 60)
C["bond_beta_60d"] = -beta_of(ret, us10y_r, 60)   # negated: high = bonds-rally hedge
C["gold_beta_60d"] = beta_of(ret, xau_r, 60)

C["roll_sharpe_60d"] = (px / px.shift(60) - 1.0) / rs(ret, 60).replace(0, np.nan)

min60 = px.rolling(60, min_periods=mp(60)).min()
max60 = px.rolling(60, min_periods=mp(60)).max()
C["stoch_pos_60d"] = (px - min60) / (max60 - min60).replace(0, np.nan)

rollmax = px.rolling(60, min_periods=mp(60)).max()
C["maxdd_60d"] = (px / rollmax - 1.0)

vix_med120 = vix.rolling(120, min_periods=mp(120)).median()
gate_calm = (vix < vix_med120).astype(float)
C["vix_gate_mom10"] = (px.shift(5) / px.shift(15) - 1.0) * gate_calm.reindex(px.index)

up = ret.where(ret > 0, 0.0)
dn = ret.where(ret < 0, 0.0).abs()
C["up_down_ratio_20d"] = up.rolling(20, min_periods=mp(20)).mean() / dn.rolling(20, min_periods=mp(20)).mean().replace(0, np.nan)

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
    flag = "  <== PASS" if ok else ""
    print(f"{name:<22}{m:>8.4f}{icir:>8.4f}{hit:>6.3f}{n:>6d}  {lc:>7.3f}  {wstr:>12s} {rstr}{flag}", flush=True)

print("\n=== GATE PASSERS (|IC|>=0.0070 & |ICIR|>=0.0840 & n>=250, full window) ===", flush=True)
gate_pass = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES:
        gate_pass.append(name)
        print(f"  PASS {name:<22} ic={r['ic']:.4f} icir={r['icir']:.4f} n={r['n']} librho={r['lib_corr']}", flush=True)

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
    print(f"  {name:<22} decay={dec} turn10={r['turnover_10d_rank']} cov={r['coverage_asset_days']}/{r['coverage_dates_ge8']}", flush=True)


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
with open("scripts/miner_3_20261231_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s; results saved to scripts/miner_3_20261231_screen_results.json", flush=True)
