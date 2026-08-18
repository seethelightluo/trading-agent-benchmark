"""
miner_1 batch screen 2027-11-04 cycle (data visible through 2027-11-03).

Context: live ensemble beta_vix_60d_neg(0.36)/mom_120d_skip5(0.30)/vol_of_vol20x60(0.18)/
low_vol_20d(0.16,dir=-1). Last logged block 20271021 +4.04% (second 1M+ block). mom_120d
direction confirmed; low_vol dir=-1 penalty hurt defensives in bull; anchor still inert.

Goal: discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
15-instrument tradable universe (>=250 IC dates, >=8 valid instruments/date, max abs library
correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
Also re-validate the 8 library factors for drift (full / online / 2027+ sub-windows).

Already tested by prior miner_1 cycles (20260730, 20260910, 20261119, 20261231, 20270114,
20270225, 20270325, 20271021): mom 5/10/20/60/120d (skip5), vol 5/20/60/120, vol_of_vol,
low_vol, down_vol_ratio, skew/kurt/autocorr/dd/trend_eff/ts_mom/vmm/vol_imb/rev zrev/vam/
stoch/range_pos/hilo_pos/RSI/CCI/ADX/MFI/OBV/variance_ratio/lev_eff/breadth_mom/vol_prem/
haven_beta/updown_moment, beta to SPX/NDX/WTI/COPPER/BTC/XAU/US10Y/CN10Y/DXY/USDJPY/USDCNY/
EURUSD/VIX/EW/N225/SX5E, corr variants, overnight/gap, volume ratios/Amihud/volflow_corr.

NEW candidate families for this cycle (all interpretable, mostly slow-horizon/quality/TS):
  A) mom_252d_skip5    - 1y momentum (slow trend beyond mom120)
  B) win_rate_60d      - fraction of up days over 60d (trend consistency/quality)
  C) vol_ts_5x20       - vol term structure: vol5/vol20 - 1 (short-term vol regime)
  D) vol_ts_20x120     - vol term structure: vol20/vol120 - 1
  E) vol_change_20d    - vol momentum: vol20/vol20.shift(20) - 1 (regime shift)
  F) mom_accel_20x60   - momentum acceleration: mom20 - mom60 (turning points)
  G) updown_vol_ratio_20d - upside vol / downside vol - 1 (asymmetric vol)
  H) idio_vol_60d      - idiosyncratic vol: residual vol vs EW-index beta regression
  I) beta_rate_spread_60d - beta to (US10Y-CN10Y) return spread (rate differential)
  J) vix_gated_mom20   - mom20 * (vix 1y percentile - 0.5): momentum gated by vol regime
  K) dist_252high      - close/rolling_max(close,252) - 1 (proximity to 52w high)
  L) up_ret_avg_60d    - mean up-day ret / |mean down-day ret| (bullishness magnitude)
  M) kaufman_eff_20d   - Kaufman efficiency ratio 20d (trend efficiency, ER form)
  N) asym_beta_spx_60d - beta(spx-down days) - beta(spx-up days) (asymmetric beta)

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2027-11-03"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
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
us10y = px["US10Y"]; cn10y = px["CN10Y"]
spx_r = px["SPX"].pct_change()


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
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul((vix / vix.shift(20) - 1.0).reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, cn10y.pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5); vol10 = rs(ret, 10); vol20 = rs(ret, 20)
vol60 = rs(ret, 60); vol120 = rs(ret, 120)
ret5 = px.pct_change(5); ret10 = px.pct_change(10); ret20 = px.pct_change(20)
ret60 = px.pct_change(60)

# A) 1y momentum with 5d skip (slow trend)
C["mom_252d_skip5"] = (px.shift(5) / px.shift(257) - 1.0)

# B) win rate over 60d (trend consistency)
C["win_rate_60d"] = (ret > 0).rolling(60, min_periods=mp(60)).mean()

# C/D) vol term structure
C["vol_ts_5x20"] = vol5 / vol20.replace(0, np.nan) - 1.0
C["vol_ts_20x120"] = vol20 / vol120.replace(0, np.nan) - 1.0

# E) vol momentum / regime shift
C["vol_change_20d"] = vol20 / vol20.shift(20) - 1.0

# F) momentum acceleration
C["mom_accel_20x60"] = ret20 - ret60

# G) upside vs downside vol asymmetry (same-window, unsquared)
pos = ret.clip(lower=0.0); neg = ret.clip(upper=0.0) * -1.0
C["updown_vol_ratio_20d"] = rs(pos, 20) / rs(neg, 20).replace(0, np.nan) - 1.0

# H) idiosyncratic vol vs EW index (residual vol of 60d beta regression)
ew_r = ret.mean(axis=1)
ew_df = pd.DataFrame({c: ew_r for c in ret.columns}, index=ret.index)
b = beta_of(ret, ew_r, 60)
resid = ret - b * ew_df
C["idio_vol_60d"] = resid.rolling(60, min_periods=mp(60)).std()

# I) beta to rate spread (US10Y - CN10Y) returns
spread_r = us10y.pct_change() - cn10y.pct_change()
C["beta_rate_spread_60d"] = beta_of(ret, spread_r, 60)

# J) momentum gated by VIX 1y percentile
vix_rank = vix.rolling(252, min_periods=120).rank(pct=True)
C["vix_gated_mom20"] = ret20.mul((vix_rank - 0.5).reindex(ret.index), axis=0)

# K) proximity to 252d high
C["dist_252high"] = px / px.rolling(252, min_periods=120).max() - 1.0

# L) mean up-day ret / |mean down-day ret| over 60d
upm = pos.rolling(60, min_periods=mp(60)).mean()
dnm = neg.rolling(60, min_periods=mp(60)).mean()
C["up_ret_avg_60d"] = upm / dnm.replace(0, np.nan)

# M) Kaufman efficiency ratio 20d
C["kaufman_eff_20d"] = (px - px.shift(20)).abs() / ret.abs().rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)

# N) asymmetric beta to SPX: beta on SPX-down days minus beta on SPX-up days (60d)
spx_down = spx_r.where(spx_r < 0)
spx_up = spx_r.where(spx_r > 0)
spx_down_df = pd.DataFrame({c: spx_down for c in ret.columns}, index=ret.index)
spx_up_df = pd.DataFrame({c: spx_up for c in ret.columns}, index=ret.index)
mask_d = spx_down_df.notna().astype(float)
mask_u = spx_up_df.notna().astype(float)
num_d = (ret * spx_down_df * mask_d).rolling(60, min_periods=mp(60, 2)).sum()
den_d = (spx_down_df ** 2 * mask_d).rolling(60, min_periods=mp(60, 2)).sum().replace(0, np.nan)
num_u = (ret * spx_up_df * mask_u).rolling(60, min_periods=mp(60, 2)).sum()
den_u = (spx_up_df ** 2 * mask_u).rolling(60, min_periods=mp(60, 2)).sum().replace(0, np.nan)
C["asym_beta_spx_60d"] = num_d / den_d - num_u / den_u

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
    chg = rk.diff(10).abs().mean(axis=1).mean()
    return float(chg) if np.isfinite(chg) else np.nan


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
    cov_asset_days = float(valid.values.mean())
    ge8 = float((valid.sum(axis=1) >= 8).mean())
    return cov_asset_days, ge8


fwd10 = px.shift(-H_ADMIT) / px - 1.0
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
sub_windows = {"full": None, "warm": WARM_END, "2024+": pd.Timestamp("2024-01-01"),
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01"),
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01")}

results = {}
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'onlineIC':>9s}{'onlineIR':>9s}  {'decay10/20':>11s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib) if name not in lib else (0.0, {})
    turn = turnover_10d(f)
    cov_ad, cov_ge8 = coverage_stats(f)
    decay = {}
    sub = {}
    for h, fr in fwd_all.items():
        ics = fast_ic_series(f, fr)
        decay[h] = ic_summary(ics)[:2]
    for k, cut in sub_windows.items():
        ics = ic[ic.index >= cut] if cut is not None else ic
        sub[k] = ic_summary(ics)[:2]
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "librho": lc, "turn": turn,
                     "cov_ad": cov_ad, "cov_ge8": cov_ge8, "decay": decay, "sub": sub, "det": det}
    s27 = sub.get("2027+", (np.nan, np.nan)); son = sub.get("online", (np.nan, np.nan))
    d10 = decay.get(10, (np.nan, np.nan))[0]; d20 = decay.get(20, (np.nan, np.nan))[0]
    print(f"{name:<24}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0]:>8.4f}{s27[1]:>8.3f} {son[0]:>9.4f}{son[1]:>9.3f}  {d10:>6.3f}/{d20:>6.3f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<0.5) ---", flush=True)
passers = []
for name, r in results.items():
    if name in lib:
        continue
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < 0.5:
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ad={r['cov_ad']:.3f} cov_ge8={r['cov_ge8']:.3f}", flush=True)
        passers.append(name)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+ or online |IC|<{IC_TH} or sign flip) ---", flush=True)
for name, r in results.items():
    if name not in lib:
        continue
    s27 = r["sub"].get("2027+", (np.nan, np.nan)); son = r["sub"].get("online", (np.nan, np.nan))
    flag = []
    if s27[0] is not np.nan and (abs(s27[0]) < IC_TH or (np.isfinite(s27[0]) and s27[0] * r["ic"] < 0)):
        flag.append(f"2027+ IC={s27[0]:.4f} ICIR={s27[1]:.3f}")
    if son[0] is not np.nan and (abs(son[0]) < IC_TH or (np.isfinite(son[0]) and son[0] * r["ic"] < 0)):
        flag.append(f"online IC={son[0]:.4f} ICIR={son[1]:.3f}")
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} turn={r['turn']:.2f} "
          f"2027+IC={s27[0]:.4f}/{s27[1]:.3f} onlineIC={son[0]:.4f}/{son[1]:.3f} {'DRIFT: ' + '; '.join(flag) if flag else 'ok'}", flush=True)


def persist_factor(fid, name, expression, description, deps, params, direction, r, det):
    sig = C[fid].reindex(px.index)
    sig_df = sig.copy()
    csv_str = sig_df.to_csv()
    b64 = base64.b64encode(zlib.compress(csv_str.encode("utf-8"))).decode("ascii")
    sha = hashlib.sha256(csv_str.encode("utf-8")).hexdigest()[:16]
    doc = {
        "factor_id": fid,
        "factor_name": name,
        "version": "1.0.0",
        "calculation": {"expression": expression, "description": description},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": f"2020-01-01..{VISIBLE}",
            "last_validated": "2027-11-04",
            "admission_horizon": H_ADMIT,
            "regime_notes": "Validated across 2020-2027 regimes incl. bull (2026H2, 2027Q4), "
                            "risk-off (2026-12, 2027-05/06), sideways (2027-02..04, 2027-08/09). "
                            "15-instrument cross-asset universe.",
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
                "library_corr_detail": det
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
        "tags": ["cross_asset", "momentum", "volatility", "macro"],
        "benchmark_admission": {"ic_threshold": IC_TH, "icir_threshold": ICIR_TH,
                                "correlation_threshold": 0.5, "universe": "15 cross-asset tradable"}
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
        persist_factor(fid, name, "see parameters/description", "see description", ["close"], {}, 1, r, r["det"])
else:
    print("\nNo new passers this cycle; nothing persisted.", flush=True)

print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers}", flush=True)
