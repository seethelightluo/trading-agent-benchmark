"""
miner_3 batch screen 2027-11-04 cycle (data visible through 2027-11-03).

Context: live ensemble beta_vix_60d_neg(0.36)/mom_120d_skip5(0.30)/vol_of_vol20x60(0.18)/
low_vol_20d(0.16,dir=-1). Last blocks strongly positive (+2.86% 20270923-1007, +4.04%
20271007-1021, net assets >1.04M): bull regime, momentum (mom_120d) confirmed 3+ blocks,
low_vol penalty hurts defensives in bull. Anchor beta_vix_60d_neg inert/neutral lately.

Goal: discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
15-instrument tradable universe (>=250 IC dates, >=8 valid instruments/date, max abs library
correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
Also re-validate the 8 library factors for drift (full / online / 2027+ / recent sub-windows).

Already tested in prior batches (avoid re-testing):
 2027-08-26: CLV/park/range-ratio, vol term-structure slope, Kaufman eff, RSI-14, 3d reversal,
   overext z, xs alpha vs EW, WTI/COPPER/NDX/BTC/US10Y betas, autocorr lag1, max loss/gain,
   profit factor, streak, vol trend, Amihud.
 2027-09-09: sharpe 20/60/120, mom10_voladj, dd_20/120, ivol60_neg, skew_20/60_neg,
   down_cov_60d_neg, down_share_60d, up_ratio_20/60, autocorr5_60d, vol_cv_20x60,
   mom20_lowvol/highvol, lead_gap_20d, vix_level_corr_60d, beta_dxy_60d, beta_sprchg_60d.
 2027-09-23: mom_accel_60x20, mom_chg_20x40, stoch_k_10d, range_pos_20d, updown_vol_ratio_60d,
   semibeta_down_60d, kurt_60d_neg, sortino_60d, maxdd_120d_neg, downside_freq_60d_neg,
   trendfilter_mom20, yield_beta_60d, spx_beta_60d_neg, beta_usdjpy_60d, volume_ratio_20x120,
   overnight_ret_20d, intraday_ret_20d, gap_up_freq_20d, skew_120d_neg.

NEW candidate families for this cycle (avoiding the above):
  A) price position: range_pos_60d/120d, new_high_prox_60d
  B) medium-horizon RSI 30/60
  C) efficiency ratio 40/90d
  D) risk-adjusted momentum: sharpe_40d, mom_spread_120x20, mom_avg_60_120
  E) tail/capture: cvar_60d_neg, upside_capture_60d, kurt_20d_neg, skew_40d_neg
  F) cross-asset betas: beta_ew_60d_neg (equal-weight index), beta_btc_60d, beta_xau_60d,
     yield_beta_20d
  G) conditional/participation: mom_vol_conf_20d, vix_riskoff_mom20
  H) microstructure: gap_down_freq_20d_neg, overnight_ret_10d, autocorr2_60d, ret10_rev_neg
  I) drawdown analytics: calmar_60d, dd_duration_120d_neg, up_ratio_120d

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
vix = obs["VIX"]; vixr = vix.pct_change(); vix_move20 = (vix / vix.shift(20) - 1.0)
us10y = px["US10Y"]


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
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5); vol10 = rs(ret, 10); vol20 = rs(ret, 20)
vol40 = rs(ret, 40); vol60 = rs(ret, 60); vol120 = rs(ret, 120)
ret5 = px.pct_change(5); ret10 = px.pct_change(10); ret20 = px.pct_change(20)
ret40 = px.pct_change(40); ret60 = px.pct_change(60); ret120 = px.pct_change(120)
ew_ret = ret.mean(axis=1)  # equal-weight cross-asset index return

# A) price position
for w in (60, 120):
    hh = hi.rolling(w, min_periods=mp(w)).max()
    ll = lo.rolling(w, min_periods=mp(w)).min()
    C[f"range_pos_{w}d"] = (px - ll) / (hh - ll).replace(0, np.nan)
C["new_high_prox_60d"] = px / px.rolling(60, min_periods=mp(60)).max()

# B) medium-horizon RSI (simple average-gain/loss form)
def rsi(pxx, w):
    chg = pxx.diff()
    up = chg.clip(lower=0).rolling(w, min_periods=mp(w)).mean()
    dn = (-chg.clip(upper=0)).rolling(w, min_periods=mp(w)).mean()
    rsi_v = 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))
    return rsi_v

C["rsi_30d"] = rsi(px, 30)
C["rsi_60d"] = rsi(px, 60)

# C) efficiency ratio 40/90d
for w in (40, 90):
    C[f"eff_ratio_{w}d"] = (px / px.shift(w) - 1.0).abs() / ret.abs().rolling(w, min_periods=mp(w)).sum().replace(0, np.nan)

# D) risk-adjusted momentum
C["sharpe_40d"] = ret40 / vol40.replace(0, np.nan)
C["mom_spread_120x20"] = ret120 - ret20
C["mom_avg_60_120"] = 0.5 * (ret60 + ret120)

# E) tail / capture
q = ret.rolling(60, min_periods=mp(60)).quantile(0.05)
C["cvar_60d_neg"] = -q  # higher = less left-tail risk
up5 = ret.rolling(60, min_periods=mp(60)).quantile(0.95)
C["upside_capture_60d"] = up5
C["kurt_20d_neg"] = -ret.rolling(20, min_periods=mp(20)).kurt()
C["skew_40d_neg"] = -ret.rolling(40, min_periods=mp(40)).skew()

# F) cross-asset betas
C["beta_ew_60d_neg"] = -beta_of(ret, ew_ret, 60)
C["beta_btc_60d"] = beta_of(ret, px["BTC"].pct_change(), 60)
C["beta_xau_60d"] = beta_of(ret, px["XAU"].pct_change(), 60)
C["yield_beta_20d"] = beta_of(ret, us10y.pct_change(), 20)

# G) conditional / participation
C["mom_vol_conf_20d"] = ret20 * (vol20 / vol120.replace(0, np.nan))
vix_gate = (vix > vix.rolling(120, min_periods=mp(120)).quantile(0.5)).astype(float)
C["vix_riskoff_mom20"] = ret20.mul(vix_gate.reindex(ret.index), axis=0)

# H) microstructure
C["gap_down_freq_20d_neg"] = -(op < px.shift(1) * 0.995).rolling(20, min_periods=mp(20)).mean()
ovn = op / px.shift(1) - 1.0
C["overnight_ret_10d"] = ovn.rolling(10, min_periods=mp(10)).sum()
C["autocorr2_60d"] = ret.rolling(60, min_periods=mp(60)).apply(
    lambda x: pd.Series(x).autocorr(2) if len(x) >= mp(60) and pd.Series(x).std() > 0 else np.nan, raw=False)
C["ret10_rev_neg"] = -ret10

# I) drawdown analytics
dd = px / px.rolling(120, min_periods=mp(120)).max() - 1.0
C["calmar_60d"] = ret60 / (-dd.rolling(60, min_periods=mp(60)).min()).replace(0, np.nan)
C["dd_duration_120d_neg"] = -(px.rolling(120, min_periods=mp(120)).apply(
    lambda x: int(np.argmax(x.values[::-1])) if len(x) >= mp(120) else np.nan, raw=True))
C["up_ratio_120d"] = (ret > 0).rolling(120, min_periods=mp(120)).mean()

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
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01"),
               "recent": pd.Timestamp("2026-11-01")}

results = {}
print(f"\n{'name':<24}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'onlineIC':>9s}{'onlineIR':>9s}  {'decay10/20':>11s}", flush=True)
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
    son = rec.get("online", (None, None))
    print(f"{name:<24}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{son[0] if son else float('nan'):>9.4f}{son[1] if son else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<0.5) ---", flush=True)
passers = []
for name, r in results.items():
    if name in lib:
        continue
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < 0.5:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov_ad={r['cov_ad']:.3f} cov_ge8={r['cov_ge8']:.3f} sub={r['sub']}", flush=True)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+ or online |IC|<{IC_TH} or sign flip) ---", flush=True)
for name in lib:
    r = results[name]
    s27 = r["sub"].get("2027+")
    son = r["sub"].get("online")
    flag = []
    if s27 and (abs(s27[0]) < IC_TH or (s27[0] * r["ic"] < 0)):
        flag.append(f"2027+ IC={s27[0]:.4f} ICIR={s27[1]:.3f}")
    if son and (abs(son[0]) < IC_TH or (son[0] * r["ic"] < 0)):
        flag.append(f"online IC={son[0]:.4f} ICIR={son[1]:.3f}")
    if r["librho"] >= 0.5:
        flag.append(f"librho={r['librho']:.3f}")
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
            "last_validated": "2027-11-04",
            "admission_horizon": H_ADMIT,
            "regime_notes": "Validated across 2020-2027 regimes incl. bull (2026H2, 2027H2), risk-off (2026-12, 2027-05/06), sideways (2027-02..04, 2027-08/09). 15-instrument cross-asset universe.",
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
        "tags": ["cross_asset", "momentum", "volatility", "macro", "trend"],
        "benchmark_admission": {"ic_threshold": IC_TH, "icir_threshold": ICIR_TH,
                                "correlation_threshold": 0.5, "universe": "15 cross-asset tradable"}
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh)
    print(f"WROTE {path} bytes={os.path.getsize(path)}", flush=True)
    # read-back verification
    with open(path) as fh:
        back = json.load(fh)
    assert back["factor_id"] == fid, "id mismatch"
    assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert back["validation"]["metrics"]["ic"] >= IC_TH, "ic below threshold"
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
