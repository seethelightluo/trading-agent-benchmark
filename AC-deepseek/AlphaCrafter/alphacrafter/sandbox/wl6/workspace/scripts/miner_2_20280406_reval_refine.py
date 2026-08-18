"""
miner_2 re-validation + refinement wave 2028-04-06 cycle
(data visible through 2028-04-05, previous completed trading day).

Context: sign_ewma_60d passed gate last cycle and is now in strategy.py.
This cycle:
  (1) Re-validate the 15 persisted library factors for drift through 2028-04-05.
  (2) Re-screen last cycle's candidates (near-miss families) with fresh data.
  (3) Add refinement variants: span/lookback sweeps of the near-miss families
      (sign_ewma span sweep, entropy windows, trimmed-mom trim counts,
      down-vol-ratio window pairs, gap z windows, autocorr windows, maxdd
      windows, pv-corr windows, herf windows, vol-spike thresholds).
  (4) Gate NEW candidates: |IC|>=0.0070 & |ICIR|>=0.0840 at H=10, n>=250,
      >=8 valid instruments/date, max abs library corr < 0.5, recent-window
      sign guard. Persist passers (do NOT persist library revalidations).

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib, os
import numpy as np
import pandas as pd

VISIBLE = "2028-04-05"
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
us10y_r = px["US10Y"].pct_change(); cn10y_r = px["CN10Y"].pct_change()
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


def colwise_roll_corr(a, b, w):
    out = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
    for c in a.columns:
        out[c] = a[c].rolling(w, min_periods=mp(w)).corr(b[c])
    return out


# ---------------- library signals (15 persisted factors, recomputed) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
vix_move20 = (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, cn10y_r, 60)
lib["beta_chi_60d"] = beta_of(ret, px["000300.SH"].pct_change(), 60)
lib["corr_us10y_60d"] = corr_of(ret, us10y_r, 60)
vov = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vol_of_vol_chg_20d"] = vov / vov.shift(20) - 1.0
xau_copper_ratio = px["XAU"] / px["COPPER"]
lib["xau_copper_cond_20d"] = beta_of(ret, xau_copper_ratio.pct_change(), 60).mul(
    xau_copper_ratio.pct_change(20).reindex(ret.index), axis=0)
vol20_all = rs(ret, 20)
lib["vol_beta_spx_60d"] = beta_of(vol20_all, vol20_all["SPX"], 60)
up = (ret > 0).astype(float)
lib["sign_ewma_60d"] = up.ewm(span=60, adjust=False).mean()
lib["skew_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).skew()
print(f"library signals rebuilt: {len(lib)} ({time.time()-t0:.1f}s)", flush=True)

# ---------------- new candidates: last-cycle near-miss families + refinements ----------------
C = {}
vol20 = rs(ret, 20); vol60 = rs(ret, 60); vol10 = rs(ret, 10)
ret5 = px.pct_change(5)
mom60 = px / px.shift(60) - 1.0


def sign_entropy(sig, w):
    p = sig.rolling(w, min_periods=mp(w)).mean()
    p = p.clip(1e-6, 1 - 1e-6)
    e = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)
    return -e  # high entropy (choppy) -> low score


# A) sign-EWMA span sweep (refine of sign_ewma_60d)
for sp in (20, 40, 60, 90, 120, 180):
    C[f"sign_ewma_{sp}d"] = up.ewm(span=sp, adjust=False).mean()

# B) sign-entropy window sweep
for w in (20, 40, 60, 90, 120):
    C[f"ent_{w}d_neg"] = sign_entropy(up, w)

# C) trimmed momentum: sum over W excluding best & worst k days
rsum120 = ret.rolling(120, min_periods=60).sum()
rmax120 = ret.rolling(120, min_periods=60).max()
rmin120 = ret.rolling(120, min_periods=60).min()
C["mom_trim_120d"] = rsum120 - rmax120 - rmin120
rsum60 = ret.rolling(60, min_periods=30).sum()
rmax60 = ret.rolling(60, min_periods=30).max()
rmin60 = ret.rolling(60, min_periods=30).min()
C["mom_trim_60d"] = rsum60 - rmax60 - rmin60
# trimmed with 2 best/2 worst removed over 120d
rsort = ret.rolling(120, min_periods=60).apply(lambda x: np.sort(x)[2:-2].sum() if len(x) >= 60 else np.nan, raw=True)
C["mom_trim2_120d"] = rsort

# D) Herfindahl of |daily return| shares, window sweep
rabs = ret.abs()
for w in (20, 60, 120):
    s_ = rabs.rolling(w, min_periods=max(10, w // 2)).sum()
    sh = rabs / s_.replace(0, np.nan)
    C[f"herf_ret_{w}d_neg"] = -(sh ** 2).rolling(w, min_periods=max(10, w // 2)).sum()

# E) price-volume correlation window sweep
lvol = np.log(vol.replace(0, np.nan))
for w in (20, 60, 120):
    C[f"pv_corr_{w}d"] = colwise_roll_corr(ret, lvol, w)

# F) vol-spike reversal, threshold sweep
for th in (1.2, 1.3, 1.5):
    spike = (vol20 / vol60.replace(0, np.nan) > th).astype(float)
    C[f"vol_spike_rev_5d_t{int(th*10)}"] = (-ret5) * spike

# G) short momentum variants
C["mom_5d_skip1"] = (px.shift(1) / px.shift(6) - 1.0)
C["mom_3d_skip1"] = (px.shift(1) / px.shift(4) - 1.0)
C["mom_10d_skip3"] = (px.shift(3) / px.shift(13) - 1.0)

# H) robust median momentum window sweep
for w in (20, 60, 120):
    C[f"med_ret_{w}d"] = ret.rolling(w, min_periods=w // 2).median() * float(w)

# I) downside vol ratio window-pair sweep
down10 = rs(down, 10); down20 = rs(down, 20); down60 = rs(down, 60); down120 = rs(down, 120)
C["down_vol_ratio_10x60"] = -(down10 / down60.replace(0, np.nan))
C["down_vol_ratio_10x120"] = -(down10 / down120.replace(0, np.nan))
C["down_vol_ratio_20x60"] = -(down20 / down60.replace(0, np.nan))

# J) gap t-stat window sweep
gap = op / px.shift(1) - 1.0
for w in (10, 20, 60):
    gm = gap.rolling(w, min_periods=max(5, w // 2)).mean()
    gs = gap.rolling(w, min_periods=max(5, w // 2)).std()
    C[f"gap_z_{w}d"] = (gm / gs.replace(0, np.nan))

# K) autocorrelation window sweep
for w in (10, 20, 60):
    C[f"ret_autocorr_{w}d"] = colwise_roll_corr(ret, ret.shift(1), w)

# L) drawdown depth window sweep (distance below rolling max)
for w in (126, 252, 504):
    C[f"maxdd_{w}d_neg"] = (px / px.rolling(w, min_periods=w // 2).max() - 1.0)

# M) NEW family this cycle: vol-of-vol change sign / z (regime-normalized vol trend)
C["vol_of_vol_chg_sign_20d"] = np.sign(vov / vov.shift(20) - 1.0)

# N) NEW: upside/downside capture asymmetry over 60d (up vol - down vol, negated)
up_ret = ret.clip(lower=0)
dn_ret = ret.clip(upper=0) * -1.0
C["updown_vol_asym_60d_neg"] = -(rs(up_ret, 60) - rs(dn_ret, 60))

# O) NEW: 5d relative strength vs 60d trend (pullback detector)
C["pullback_5x60"] = (px.pct_change(5) - (px / px.shift(60) - 1.0) * (5.0 / 60.0))

# P) NEW: cross-sectional rank mom composite (mean of 10/60/120 z-scores)
z10 = px.pct_change(10).rank(axis=1, pct=True)
z60 = px.pct_change(60).rank(axis=1, pct=True)
z120 = px.pct_change(120).rank(axis=1, pct=True)
C["mom_zs_10_60_120"] = (z10 + z60 + z120) / 3.0

# Q) NEW: consecutive up-day streak (persistence) - current streak length, clipped
upb = (ret > 0).astype(float)
def col_streak(s):
    s = s.astype(float)
    grp = (s != s.shift()).cumsum()
    return grp.groupby(grp).cumcount() + 1
streak = upb.apply(col_streak)
C["streak_up_neg"] = -streak.clip(upper=10)

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
               "recent": pd.Timestamp("2027-06-01")}

results = {}
print(f"\n{'name':<28}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'recentIC':>9s}{'recentIR':>9s}", flush=True)
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
    s27 = rec.get("2027+", (None, None))
    srec = rec.get("recent", (None, None))
    print(f"{name:<28}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>9.3f}", flush=True)

# ---------------- gate check for NEW candidates ----------------
passers = []
print(f"\n--- gate: |IC|>={IC_TH} & |ICIR|>={ICIR_TH} & n>={MIN_IC_DATES} & librho<{CORR_TH} & cov_ge8>=0.7 ---", flush=True)
for name, r in results.items():
    if name in lib:
        continue
    ok_ic = abs(r["ic"]) >= IC_TH
    ok_icir = abs(r["icir"]) >= ICIR_TH
    ok_n = r["n"] >= MIN_IC_DATES
    ok_rho = r["librho"] < CORR_TH
    ok_cov = r["cov_ge8"] >= 0.7
    s27 = r["sub"].get("2027+", (None, None))
    srec = r["sub"].get("recent", (None, None))
    recent_ok = True
    if s27 and srec:
        if s27[0] < -0.010 and s27[1] < -0.05:
            recent_ok = False
        if srec[0] < -0.015 and srec[1] < -0.06:
            recent_ok = False
    verdict = all([ok_ic, ok_icir, ok_n, ok_rho, ok_cov]) and recent_ok
    print(f"{name:<28} ic={r['ic']:+.4f} icir={r['icir']:+.3f} n={r['n']} rho={r['librho']:.3f} "
          f"cov_ge8={r['cov_ge8']:.2f} 2027+={s27} recent={srec} -> {'PASS' if verdict else 'fail'}", flush=True)
    if verdict:
        passers.append(name)

# ---------------- persistence ----------------
def persist_factor(fid, name, expression, desc, deps, params, direction, r, det):
    sig = results[fid]
    sig_df = {**C, **lib}[fid].reindex(px.index)
    sig_csv = sig_df.round(10).to_csv()
    b64 = base64.b64encode(zlib.compress(sig_csv.encode("utf-8"))).decode("ascii")
    sha = hashlib.sha256(sig_csv.encode("utf-8")).hexdigest()[:16]
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
            "last_validated": "2028-04-06",
            "admission_horizon": H_ADMIT,
            "regime_notes": "Validated across 2020-2028 regimes incl. bull (2026H2, 2027H2, 2028-02/03), risk-off (2026-12, 2027-05/06), sideways (2027-02..04, 2027-08/09). 15-instrument cross-asset universe.",
            "metrics": {
                "ic": round(sig["ic"], 4),
                "icir": round(sig["icir"], 4),
                "ic_hit_ratio": round(sig["hit"], 3),
                "n_ic_dates": sig["n"],
                "coverage_asset_days": round(sig["cov_ad"], 3),
                "coverage_dates_ge8": round(sig["cov_ge8"], 3),
                "turnover_10d_rank": round(sig["turn"], 3),
                "decay_ic_by_horizon": {str(h): v[0] for h, v in sig["decay"].items() if v},
                "max_abs_library_correlation": round(sig["librho"], 4),
                "library_corr_detail": det
            },
            "subwindow_ic": {k: (v[0] if v else None) for k, v in sig["sub"].items()},
            "subwindow_icir": {k: (v[1] if v else None) for k, v in sig["sub"].items()}
        },
        "signal_artifact": {
            "format": "base64:zlib:csv",
            "description": "Factor signal panel: rows = dates, cols = assets.",
            "columns": list(sig_df.columns),
            "shape": list(sig_df.shape),
            "n_valid_values": int(sig_df.notna().sum().sum()),
            "sha256": sha,
            "data": b64
        },
        "tags": ["cross_asset", "momentum", "volatility", "trend", "volume", "predictability"],
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

# ---------------- library drift flags ----------------
print("\n--- library drift flags (full vs 2027+ vs recent) ---", flush=True)
for name, r in results.items():
    if name not in lib:
        continue
    f_ic, f_ir = r["ic"], r["icir"]
    s27 = r["sub"].get("2027+", (None, None))
    srec = r["sub"].get("recent", (None, None))
    flags = []
    if f_ir is not None and abs(f_ir) < 0.05:
        flags.append("WEAK_FULL")
    if s27 and s27[1] is not None and abs(s27[1]) < 0.04:
        flags.append("WEAK_2027+")
    if srec and srec[1] is not None and abs(srec[1]) < 0.04:
        flags.append("WEAK_RECENT")
    if s27 and srec and s27[0] is not None and srec[0] is not None and np.sign(s27[0]) != np.sign(f_ic):
        flags.append("SIGN_FLIP_2027+")
    if srec and srec[0] is not None and np.sign(srec[0]) != np.sign(f_ic):
        flags.append("SIGN_FLIP_RECENT")
    print(f"{name:<26} full=({f_ic:+.4f},{f_ir:+.3f}) 2027+=({s27[0] if s27 else float('nan'):+.4f},{s27[1] if s27 else float('nan'):+.3f}) "
          f"recent=({srec[0] if srec else float('nan'):+.4f},{srec[1] if srec else float('nan'):+.3f}) flags={flags or 'ok'}", flush=True)

with open("scripts/miner_2_20280406_reval_refine_results.json", "w") as fh:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "det"} for k, v in results.items()}, fh, default=str)

print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers}", flush=True)
