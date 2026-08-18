"""
miner_1 batch screen 2028-03-23 cycle (data visible through 2028-03-22).

Context: 13 EFFECTIVE library factors; live ensemble (cycle 46):
beta_vix_60d_neg(0.36)/down_vol_ratio_20x120(0.22)/mom_120d_skip5(0.18)/
vol_of_vol20x60(0.12)/low_vol_20d(0.12,dir=-1). Trader feedback (2028-03-23 block):
positive block, mom_120d_skip5 strongest, vol_of_vol20x60 + low_vol dir=-1 still dragging
(penalized SOX/SX5E/SPX), frozen HSI/000688.SH/CN10Y dead capital ok.

Goal: (1) re-validate the 13 library factors for drift through 2028-03-22;
(2) discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10
on the 15-instrument tradable universe (>=250 IC dates, >=8 valid instruments/date,
max abs library correlation < 0.5); (3) PERSIST gate-passers with base64:zlib:csv
signal artifacts; (4) print library drift flags.

NEW candidate families this cycle (avoiding all previously tested ideas incl.
miner_3's dxy_z_beta, skew_chg, on_intra_div, corr_spx_chg, down_capture, resid_mom):
  A) mdd_60d_neg          - 60d max-drawdown depth (negated; deep-drawdown bet)
  B) range_pos_20d        - position inside 20d high-low range (overbought/trend)
  C) range_pos_60d        - position inside 60d high-low range
  D) gap_freq_20d_neg     - frequency of large overnight gaps (>1.5%), negated
  E) overnight_ret_20d    - mean overnight return (open/prev_close - 1), 20d
  F) intraday_ret_20d     - mean intraday return (close/open - 1), 20d
  G) skew_20d_neg         - 20d return skewness level (negated; crash-risk)
  H) accel_mom_10x120     - momentum acceleration: mom10_skip5 - mom120_skip5
  I) vam_20d_s5_v20       - vol-adjusted momentum 20d (re-test, passed gate 2027-02)
  J) eth_btc_cond_20d     - beta to ETH/BTC ratio returns x ratio 20d momentum
  K) jpy_beta_60d         - beta to USDJPY returns (risk-on/off currency)
  L) usdcny_beta_60d      - beta to USDCNY returns (China FX sensitivity)
  M) eurusd_beta_60d      - beta to EURUSD returns
  N) dxy_cond_60x20       - beta to DXY x DXY 20d momentum (dollar-regime cond.)
  O) corr_wti_chg_20x60   - change in 60d correlation to WTI (energy-sens shift)
  P) range_intensity_20d  - mean (high-low)/close over 20d (range intensity)

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2028-03-22"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
CORR_TH = 0.5
WARM_END = pd.Timestamp("2026-07-15")
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

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

obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
vix = obs["VIX"]; vixr = vix.pct_change()
vix_move20 = (vix / vix.shift(20) - 1.0)
dxy = obs["DXY"]; dxy_r = dxy.pct_change()
jpy = obs["USDJPY"]; jpy_r = jpy.pct_change()
cny = obs["USDCNY"]; cny_r = cny.pct_change()
eur = obs["EURUSD"]; eur_r = eur.pct_change()
us10y = px["US10Y"]; cn10y = px["CN10Y"]


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


def corr_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w)).corr(mdf)


# ---------------- library signals (13 persisted factors, recomputed) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, cn10y.pct_change(), 60)
lib["beta_chi_60d"] = beta_of(ret, px["000300.SH"].pct_change(), 60)
lib["corr_us10y_60d"] = corr_of(ret, us10y.pct_change(), 60)
vov = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vol_of_vol_chg_20d"] = vov / vov.shift(20) - 1.0
lib["vol_beta_spx_60d"] = beta_of(rs(ret, 20), rs(ret, 20)["SPX"], 60)
xau_r = px["XAU"].pct_change(); copper_r = px["COPPER"].pct_change()
xau_copper_ratio = px["XAU"] / px["COPPER"]
lib["xau_copper_cond_20d"] = beta_of(ret, xau_copper_ratio.pct_change(), 60).mul(
    xau_copper_ratio.pct_change(20).reindex(ret.index), axis=0)

# ---------------- new candidates ----------------
C = {}
vol10 = rs(ret, 10); vol20 = rs(ret, 20); vol60 = rs(ret, 60); vol120 = rs(ret, 120)
spx_r = px["SPX"].pct_change(); wti_r = px["WTI"].pct_change()
btc_r = px["BTC"].pct_change(); eth_r = px["ETH"].pct_change()

# A) 60d max-drawdown depth, negated (deep-drawdown bet)
run_max60 = px.rolling(60, min_periods=mp(60)).max()
mdd60 = px / run_max60 - 1.0  # <= 0
C["mdd_60d_neg"] = -mdd60

# B/C) range position: (close - min_low) / (max_high - min_low)
min_low20 = lo.rolling(20, min_periods=mp(20)).min()
max_high20 = hi.rolling(20, min_periods=mp(20)).max()
C["range_pos_20d"] = (px - min_low20) / (max_high20 - min_low20).replace(0, np.nan)
min_low60 = lo.rolling(60, min_periods=mp(60)).min()
max_high60 = hi.rolling(60, min_periods=mp(60)).max()
C["range_pos_60d"] = (px - min_low60) / (max_high60 - min_low60).replace(0, np.nan)

# D) large overnight gap frequency, negated
prev_close = px.shift(1)
gap = op / prev_close - 1.0
C["gap_freq_20d_neg"] = -(gap.abs() > 0.015).rolling(20, min_periods=mp(20)).mean()

# E/F) overnight vs intraday mean returns
C["overnight_ret_20d"] = gap.rolling(20, min_periods=mp(20)).mean()
intra = px / op - 1.0
C["intraday_ret_20d"] = intra.rolling(20, min_periods=mp(20)).mean()

# G) skewness level 20d, negated (crash-risk)
C["skew_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).skew()

# H) momentum acceleration: short vs long momentum difference
mom10s5 = px.shift(5) / px.shift(15) - 1.0
mom120s5 = px.shift(5) / px.shift(125) - 1.0
C["accel_mom_10x120"] = mom10s5 - mom120s5

# I) vol-adjusted momentum 20d (re-test from 2027-02 cycle)
C["vam_20d_s5_v20"] = (px.shift(5) / px.shift(25) - 1.0) / vol20.replace(0, np.nan)

# J) ETH/BTC ratio regime conditional
eth_btc = px["ETH"] / px["BTC"]
C["eth_btc_cond_20d"] = beta_of(ret, eth_btc.pct_change(), 60).mul(
    eth_btc.pct_change(20).reindex(ret.index), axis=0)

# K/L/M) FX betas
C["jpy_beta_60d"] = beta_of(ret, jpy_r, 60)
C["usdcny_beta_60d"] = beta_of(ret, cny_r, 60)
C["eurusd_beta_60d"] = beta_of(ret, eur_r, 60)

# N) dollar-regime conditional beta (DXY momentum variant)
C["dxy_cond_60x20"] = beta_of(ret, dxy_r, 60).mul(dxy.pct_change(20).reindex(ret.index), axis=0)

# O) change in correlation to WTI (energy-sensitivity shift)
C["corr_wti_chg_20x60"] = corr_of(ret, wti_r, 60) - corr_of(ret, wti_r, 60).shift(20)

# P) range intensity: mean (high-low)/close over 20d
C["range_intensity_20d"] = ((hi - lo) / px.replace(0, np.nan)).rolling(20, min_periods=mp(20)).mean()

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


def turnover_10d(f):
    rk = f.rank(axis=1, pct=True)
    return float(rk.diff(10).abs().mean(axis=1).mean())


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


def coverage_stats(f):
    valid = f.notna()
    return float(valid.values.mean()), float((valid.sum(axis=1) >= 8).mean())


fwd10 = px.shift(-H_ADMIT) / px - 1.0
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
sub_windows = {"full": None, "warm": WARM_END, "2024+": pd.Timestamp("2024-01-01"),
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01"),
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01"),
               "recent": pd.Timestamp("2027-11-01")}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}  {'decay10/20':>11s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    turn = turnover_10d(f)
    cov_ad, cov_ge8 = coverage_stats(f)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic if wname == "full" else ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    dec = {}
    for h, fh in fwd_all.items():
        ich = fast_ic_series(f, fh)
        mm, ii, _, _ = ic_summary(ich)
        dec[h] = (round(mm, 4), round(ii, 4)) if np.isfinite(mm) else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "librho": lc,
                     "turn": turn, "sub": rec, "decay": dec, "det": det,
                     "cov_ad": cov_ad, "cov_ge8": cov_ge8}
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<{CORR_TH}) ---", flush=True)
passers = []
for name, r in results.items():
    if name in lib:
        continue
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < CORR_TH:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ad={r['cov_ad']:.3f} cov_ge8={r['cov_ge8']:.3f} sub={r['sub']}", flush=True)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+ or recent |IC|<{IC_TH} or sign flip) ---", flush=True)
for name in lib:
    r = results[name]
    s27 = r["sub"].get("2027+")
    srec = r["sub"].get("recent")
    flag = []
    if s27 and (abs(s27[0]) < IC_TH or (s27[0] * r["ic"] < 0)):
        flag.append(f"2027+ IC={s27[0]:.4f} ICIR={s27[1]:.3f}")
    if srec and (abs(srec[0]) < IC_TH or (srec[0] * r["ic"] < 0)):
        flag.append(f"recent IC={srec[0]:.4f} ICIR={srec[1]:.3f}")
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} -> {'FLAG ' + '; '.join(flag) if flag else 'ok'}", flush=True)


# ---------------- persistence for passers ----------------
def persist_factor(fid, name, expression, desc, deps, params, direction, r, det):
    sig = C[fid].reindex(px.index)
    sig_df = sig.copy()
    sig_df.index = sig_df.index.strftime("%Y-%m-%d")
    csv_bytes = sig_df.to_csv().encode("utf-8")
    comp = zlib.compress(csv_bytes, 6)
    b64 = base64.b64encode(comp).decode("ascii")
    sha = hashlib.sha256(csv_bytes).hexdigest()[:16]
    doc = {
        "factor_id": fid,
        "factor_name": name,
        "version": "1.0.0",
        "calculation": {"expression": expression, "description": desc},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{px.index.min().date()}..{px.index.max().date()}",
            "last_validated": "2028-03-23",
            "admission_horizon": H_ADMIT,
            "regime_notes": "Validated across 2020-2028 regimes incl. bull (2026H2, 2027H2, 2028-02/03), risk-off (2026-12, 2027-05/06), sideways (2027-02..04, 2027-08/09, 2027-11/12), crypto surge + WTI whipsaw (2027-12/2028-01). 15-instrument cross-asset universe; frozen HSI/000688.SH/CN10Y series.",
            "metrics": {
                "ic": round(r["ic"], 4),
                "icir": round(r["icir"], 4),
                "ic_hit_ratio": round(r["hit"], 3),
                "n_ic_dates": r["n"],
                "coverage_asset_days": round(r["cov_ad"], 3),
                "coverage_dates_ge8": round(r["cov_ge8"], 3),
                "turnover_10d_rank": round(r["turn"], 3),
                "decay_ic_by_horizon": {str(h): v[0] for h, v in r["decay"].items() if v},
                "max_abs_library_correlation": round(r["librho"], 4),
                "library_corr_detail": r["det"]
            },
            "subwindow_ic": {k: (v[0] if v else None) for k, v in r["sub"].items()},
            "subwindow_icir": {k: (v[1] if v else None) for k, v in r["sub"].items()}
        },
        "signal_artifact": {
            "format": "base64:zlib:csv",
            "description": f"Factor signal panel: rows = dates, cols = assets. Shape {sig_df.shape}",
            "columns": list(sig_df.columns),
            "shape": list(sig_df.shape),
            "n_valid_values": int(sig.notna().sum().sum()),
            "sha256": sha,
            "data": b64
        },
        "tags": ["cross_asset", "momentum", "volatility", "macro", "trend", "reversal", "regime"],
        "benchmark_admission": {"ic_threshold": IC_TH, "icir_threshold": ICIR_TH,
                                "correlation_threshold": CORR_TH, "universe": "15 cross-asset tradable"}
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh)
    print(f"WROTE {path} bytes={os.path.getsize(path)}", flush=True)
    with open(path) as fh:
        back = json.load(fh)
    assert back["factor_id"] == fid, "id mismatch"
    assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert abs(back["validation"]["metrics"]["ic"]) >= IC_TH, "ic below threshold"
    assert abs(back["validation"]["metrics"]["icir"]) >= ICIR_TH, "icir below threshold"
    art = back["signal_artifact"]
    csv_dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
    chk = pd.read_csv(pd.io.common.StringIO(csv_dec), index_col=0)
    assert chk.shape == tuple(art["shape"]), f"shape mismatch {chk.shape} vs {art['shape']}"
    assert hashlib.sha256(csv_dec.encode("utf-8")).hexdigest()[:16] == art["sha256"], "sha mismatch"
    print(f"VERIFIED {fid}: reload OK shape={chk.shape} n_valid={art['n_valid_values']} sha={art['sha256']}", flush=True)


if passers:
    print(f"\n--- persisting {passers} ---", flush=True)
    for fid in passers:
        r = results[fid]
        name = fid.replace("_", " ").title()
        persist_factor(fid, name, "see parameters/description", "see description",
                       ["close", "high", "low", "open", "volume"], {}, 1, r, r["det"])
else:
    print("\nNo new passers this cycle; nothing persisted.", flush=True)

# dump results json for audit
with open("scripts/miner_1_20280323_screen_batch_result.json", "w") as fh:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "det"} for k, v in results.items()}, fh, indent=1)
print("wrote scripts/miner_1_20280323_screen_batch_result.json", flush=True)

print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers}", flush=True)
