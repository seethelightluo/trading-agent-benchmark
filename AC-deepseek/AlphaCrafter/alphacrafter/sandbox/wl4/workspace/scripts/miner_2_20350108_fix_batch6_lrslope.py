"""miner_2 2035-01-08 FIX: batch6 re-run with VECTORIZED lr_slope_60d.

Part 1: revalidate 3 effective library factors for drift (full window + recent 2Y).
Part 2: screen NEW candidates (batch 6) incl. fixed C3 lr_slope_60d.
Data visible through previous completed trading day only (no lookahead).
Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
Uses a VECTORIZED rank-IC (Pearson-on-ranks) to keep runtime low.
"""
import sys, warnings, json, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    coverage_metrics, turnover_rank, decay_profile, library_signals, max_library_corr,
)

t_start = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print(f"data through: {END} | n_dates: {len(closes)} | n_assets: {closes.shape[1]}", flush=True)


def fast_rank_ic_series(factor_panel, fwd, min_valid=8):
    """Vectorized daily Spearman IC = Pearson corr of cross-sectional ranks."""
    idx = factor_panel.index.intersection(fwd.index)
    f = factor_panel.loc[idx]
    r = fwd.loc[idx]
    fr = f.rank(axis=1, method="average")
    rr = r.rank(axis=1, method="average")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1)
    x = fr.where(valid)
    y = rr.where(valid)
    sx = x.sum(axis=1)
    sy = y.sum(axis=1)
    sxx = (x * x).sum(axis=1)
    syy = (y * y).sum(axis=1)
    sxy = (x * y).sum(axis=1)
    denom = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    ic = (n * sxy - sx * sy) / denom.replace(0, np.nan)
    ok = (n >= min_valid) & denom.notna() & np.isfinite(ic)
    return pd.Series(ic[ok].values, index=idx[ok], name="ic")


def summarize_ic(ic_series, expected_sign=1):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(sd, 4)}


def full_eval_fast(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20), min_valid=8,
                   expected_sign=1, library=None, admission_horizon=10):
    fwd = forward_returns(closes, admission_horizon)
    ics = fast_rank_ic_series(factor_panel, fwd, min_valid)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(factor_panel, min_valid=min_valid))
    m["turnover_10d_rank"] = turnover_rank(factor_panel, 10)
    m["decay_ic_by_horizon"] = {}
    for h in horizons:
        fw = forward_returns(closes, h)
        ih = fast_rank_ic_series(factor_panel, fw, min_valid)
        m["decay_ic_by_horizon"][str(h)] = round(float(ih.mean()), 4) if len(ih) else None
    if library is not None:
        corr, key = max_library_corr(factor_panel, library)
        m["max_abs_library_correlation"] = corr
        m["max_corr_factor"] = key
    return m, ics


def eval_factor(name, sig, expected_sign, window=None, library=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval_fast(s, c, (1, 2, 3, 5, 10, 20), 8, expected_sign,
                            library=library, admission_horizon=10)
    m["admission_gate"] = {"ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
                           "ic_pass": abs(m["ic"]) >= 0.0070,
                           "icir_pass": abs(m["icir"]) >= 0.0840}
    gate = m["admission_gate"]
    ok = gate["ic_pass"] and gate["icir_pass"]
    print(f"=== {name} (dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={'PASS' if ok else 'FAIL'}", flush=True)
    return m, ics


# ---------- library reference signals (3 effective factors) ----------
vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
usdjpy = panels["USDJPY"]["close"].astype(float) if "USDJPY" in panels else None
usdcny = panels["USDCNY"]["close"].astype(float) if "USDCNY" in panels else None
us10y = panels["US10Y"]["close"].astype(float) if "US10Y" in panels else None
cn10y = panels["CN10Y"]["close"].astype(float) if "CN10Y" in panels else None
eurusd = panels["EURUSD"]["close"].astype(float) if "EURUSD" in panels else None

lib_sigs = library_signals(panels, closes, rets, vix)
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
lib_sigs["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_down[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
lib_sigs["dn_mkt_beta_60d"] = pd.DataFrame(beta_down, index=rets.index)
if cn10y is not None:
    cn10y_ret = cn10y.pct_change()
    beta_cn = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
        beta_cn[a] = z["a"].rolling(60, min_periods=40).cov(z["r"]) / z["r"].rolling(60, min_periods=40).var()
    lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)
eff_lib = {k: lib_sigs[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
print("effective library reference signals:", list(eff_lib.keys()), flush=True)

print("=" * 70)
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, library=eff_lib)
print("--- RECENT 2Y drift (2032-12-23..END) ---", flush=True)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, ("2032-12-23", END), library=eff_lib)

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 6, fixed C3)")
print("=" * 70)
results = {}

# C1: mom60_skip10_voladj - 60d momentum skipping last 10d, vol-adjusted (skip variant)
mom60s = closes.shift(10) / closes.shift(70) - 1.0
vol60 = rets.rolling(60).std()
sig_c1 = mom60s / vol60
m, _ = eval_factor("mom60_skip10_voladj", sig_c1, 1, library=eff_lib); results["mom60_skip10_voladj"] = m

# C2: eff_ratio_20d - Kaufman efficiency ratio (trend efficiency): |net| / sum(|ret|)
sig_c2 = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
m, _ = eval_factor("eff_ratio_20d", sig_c2, 1, library=eff_lib); results["eff_ratio_20d"] = m

# C3: lr_slope_60d - VECTORIZED normalized linear-regression trend slope (t-stat of log-price trend)
def _lr_slope_vec(px, win=60):
    """Rolling t-stat of log-price OLS trend. Vectorized via rolling sums.

    For window ending at row t (0-based), local x = i-(t-win+1) for i in [t-win+1, t].
    Sxy = sum((x-xm)*y) = sum(i*y) - c_t*sum(y) with c_t = t-(win-1)/2.
    tstat = Sxy * sqrt(win-2) / sqrt(Sxx * SSE), SSE = Syy - Sxy^2/Sxx.
    """
    n = len(px)
    idx = np.arange(n, dtype=float)
    logp = np.log(px)
    sum_y = logp.rolling(win, min_periods=win).sum()
    sum_y2 = (logp ** 2).rolling(win, min_periods=win).sum()
    sum_iy = (logp * idx).rolling(win, min_periods=win).sum()
    c_t = pd.Series(idx - (win - 1) / 2.0, index=px.index)
    sxy = sum_iy - c_t * sum_y
    sxx = win * (win ** 2 - 1) / 12.0
    syy = sum_y2 - sum_y ** 2 / win
    sse = (syy - sxy ** 2 / sxx).clip(lower=0.0)
    tstat = sxy * np.sqrt(win - 2) / np.sqrt(sxx * sse)
    return tstat.replace([np.inf, -np.inf], np.nan)


sig_c3 = _lr_slope_vec(closes)
m, _ = eval_factor("lr_slope_60d", sig_c3, 1, library=eff_lib); results["lr_slope_60d"] = m

# C4: up_down_capture_20d - avg up-day ret / avg |down-day| ret over 20d (trend quality)
up = rets.where(rets > 0)
dn = rets.where(rets < 0)
sig_c4 = up.rolling(20).mean() / dn.rolling(20).mean().abs().replace(0, np.nan)
m, _ = eval_factor("up_down_capture_20d", sig_c4, 1, library=eff_lib); results["up_down_capture_20d"] = m

# C5: mom_consistency_60d - fraction of positive days over 60d (trend breadth)
sig_c5 = (rets > 0).rolling(60).mean()
m, _ = eval_factor("mom_consistency_60d", sig_c5, 1, library=eff_lib); results["mom_consistency_60d"] = m

# C6: stoch_k_14 - stochastic oscillator %K (14d) - mean reversion at extremes
ll14 = closes.rolling(14).min()
hh14 = closes.rolling(14).max()
sig_c6 = (closes - ll14) / (hh14 - ll14).replace(0, np.nan)
m, _ = eval_factor("stoch_k_14", sig_c6, -1, library=eff_lib); results["stoch_k_14"] = m

# C7: atr_pct_20d - ATR20 / close (intraday-range vol)
hi_p = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
lo_p = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
tr = pd.concat([(hi_p - lo_p), (hi_p - closes.shift(1)).abs(), (lo_p - closes.shift(1)).abs()]).groupby(level=1).max()
atr20 = tr.rolling(20).mean()
sig_c7 = atr20 / closes
m, _ = eval_factor("atr_pct_20d", sig_c7, -1, library=eff_lib); results["atr_pct_20d"] = m

# C8: vix_regime_mom_20x60 - momentum acceleration scaled DOWN in high-VIX regime (conditional)
if vix is not None:
    vix_med = vix.rolling(252).median()
    regime = vix / vix_med
    scale = regime.where(regime > 1.0, 1.0)  # >1 in high-VIX
    sig_c8 = ((mom20 - mom60) / vol20) / scale
    m, _ = eval_factor("vix_regime_mom_20x60", sig_c8, 1, library=eff_lib); results["vix_regime_mom_20x60"] = m

# C9: wti_beta_60d - beta to WTI returns (commodity-cycle exposure)
if "WTI" in rets.columns:
    wti_ret = rets["WTI"]
    beta_wti = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), wti_ret.rename("w")], axis=1).dropna()
        beta_wti[a] = z["a"].rolling(60, min_periods=40).cov(z["w"]) / z["w"].rolling(60, min_periods=40).var()
    sig_c9 = pd.DataFrame(beta_wti, index=rets.index)
    m, _ = eval_factor("wti_beta_60d", sig_c9, 1, library=eff_lib); results["wti_beta_60d"] = m

# C10: xau_beta_60d - beta to XAU returns (safe-haven exposure)
if "XAU" in rets.columns:
    xau_ret = rets["XAU"]
    beta_xau = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), xau_ret.rename("g")], axis=1).dropna()
        beta_xau[a] = z["a"].rolling(60, min_periods=40).cov(z["g"]) / z["g"].rolling(60, min_periods=40).var()
    sig_c10 = pd.DataFrame(beta_xau, index=rets.index)
    m, _ = eval_factor("xau_beta_60d", sig_c10, 1, library=eff_lib); results["xau_beta_60d"] = m

# C11: us10y_beta_60d - beta to US10Y returns (US rate sensitivity, distinct from CN10Y)
if us10y is not None:
    us10y_ret = us10y.pct_change()
    beta_us = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), us10y_ret.rename("u")], axis=1).dropna()
        beta_us[a] = z["a"].rolling(60, min_periods=40).cov(z["u"]) / z["u"].rolling(60, min_periods=40).var()
    sig_c11 = pd.DataFrame(beta_us, index=rets.index)
    m, _ = eval_factor("us10y_beta_60d", sig_c11, -1, library=eff_lib); results["us10y_beta_60d"] = m

# C12: drawdown_recovery_10d - close/rolling_min(close,10)-1 (short-term oversold bounce)
sig_c12 = closes / closes.rolling(10).min() - 1.0
m, _ = eval_factor("drawdown_recovery_10d", sig_c12, 1, library=eff_lib); results["drawdown_recovery_10d"] = m

# C13: semivar_ratio_20d - downside semivariance / total variance (return asymmetry)
neg_ret = rets.where(rets < 0, 0.0)
down_var = (neg_ret ** 2).rolling(20).mean()
tot_var = (rets ** 2).rolling(20).mean()
sig_c13 = down_var / tot_var.replace(0, np.nan)
m, _ = eval_factor("semivar_ratio_20d", sig_c13, -1, library=eff_lib); results["semivar_ratio_20d"] = m

# C14: eurusd_beta_60d - beta to EURUSD returns (global risk-on/off)
if eurusd is not None:
    eurusd_ret = eurusd.pct_change()
    beta_eur = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), eurusd_ret.rename("e")], axis=1).dropna()
        beta_eur[a] = z["a"].rolling(60, min_periods=40).cov(z["e"]) / z["e"].rolling(60, min_periods=40).var()
    sig_c14 = pd.DataFrame(beta_eur, index=rets.index)
    m, _ = eval_factor("eurusd_beta_60d", sig_c14, 1, library=eff_lib); results["eurusd_beta_60d"] = m

print("=" * 70)
print("RECENT 2Y WINDOW CHECK for candidates that PASS full-window gate")
print("=" * 70)
cand_defs = {"mom60_skip10_voladj": sig_c1, "eff_ratio_20d": sig_c2, "lr_slope_60d": sig_c3,
             "up_down_capture_20d": sig_c4, "mom_consistency_60d": sig_c5, "stoch_k_14": sig_c6,
             "atr_pct_20d": sig_c7, "drawdown_recovery_10d": sig_c12, "semivar_ratio_20d": sig_c13}
if vix is not None:
    cand_defs["vix_regime_mom_20x60"] = sig_c8
if "WTI" in rets.columns:
    cand_defs["wti_beta_60d"] = sig_c9
if "XAU" in rets.columns:
    cand_defs["xau_beta_60d"] = sig_c10
if us10y is not None:
    cand_defs["us10y_beta_60d"] = sig_c11
if eurusd is not None:
    cand_defs["eurusd_beta_60d"] = sig_c14
signed = {"stoch_k_14": -1, "atr_pct_20d": -1, "semivar_ratio_20d": -1, "us10y_beta_60d": -1}
for nm, mm in results.items():
    g = mm["admission_gate"]
    if g["ic_pass"] and g["icir_pass"]:
        sd = signed.get(nm, 1)
        m2, _ = eval_factor(nm + "_RECENT2Y", cand_defs[nm], sd, ("2032-12-23", END), library=eff_lib)
        results[nm + "_recent2y"] = m2

print("=" * 70)
print("SUMMARY")
print("=" * 70)
for nm, mm in results.items():
    g = mm["admission_gate"]
    ok = "PASS" if (g["ic_pass"] and g["icir_pass"]) else "fail"
    print(f"{nm:28s} ic={mm['ic']:+.4f} icir={mm['icir']:+.4f} hit={mm['ic_hit_ratio']:.2f} "
          f"n={mm['n_ic_dates']:5d} cov8={mm['coverage_dates_ge8']:.2f} turn={mm['turnover_10d_rank']} "
          f"maxcorr={mm.get('max_abs_library_correlation')} -> {ok}", flush=True)
print("elapsed_s:", round(time.time() - t_start, 1), flush=True)
print("DONE", flush=True)
