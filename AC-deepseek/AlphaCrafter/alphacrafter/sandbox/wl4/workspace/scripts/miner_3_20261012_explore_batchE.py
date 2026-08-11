"""miner_3 batch E screen (2026-10-12) - OPTIMIZED vectorized rank-IC.

A) drift re-validation of 4 active library factors
B) batch D candidates (results were lost)
C) new batch E candidates

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns if "volume" in panels[a].columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840
mkt = rets.mean(axis=1)

# precompute forward returns
FWDS = {h: forward_returns(closes, h) for h in HORIZONS}

def rank_ic_vec(factor_panel, fwd):
    """Vectorized per-date Spearman IC (Pearson of ranks)."""
    fr = factor_panel.rank(axis=1)
    rr = fwd.rank(axis=1)
    dates, ics = [], []
    fv = fr.values
    rv = rr.values
    for i, dt in enumerate(fr.index):
        f = fv[i]
        r = rv[i]
        m = ~(np.isnan(f) | np.isnan(r))
        if m.sum() < MIN_VALID:
            continue
        fs = f[m]; rs = r[m]
        if fs.std() < 1e-14 or rs.std() < 1e-14:
            continue
        ic = np.corrcoef(fs, rs)[0, 1]
        if not np.isnan(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")

def summarize_ic(ic_series, expected_sign=1):
    if len(ic_series) == 0:
        return {"ic": np.nan, "icir": np.nan, "ic_hit_ratio": np.nan, "n_ic_dates": 0, "ic_std": np.nan}
    ic = ic_series.mean()
    std = ic_series.std(ddof=1)
    icir = ic / std if std > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {"ic": round(float(ic), 4), "icir": round(float(icir), 4),
            "ic_hit_ratio": round(float(hit), 3), "n_ic_dates": int(len(ic_series)),
            "ic_std": round(float(std), 4)}

def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)

def per_asset(func):
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes.index)

# ---------------- library signals (for corr audit) ----------------
print("building library signals...", flush=True)
lib = {}
lib["vol_price_corr_20"] = pd.DataFrame(
    {a: rets[a].rolling(20).corr(vol_panel[a]) if a in vol_panel.columns else pd.Series(np.nan, index=rets.index)
     for a in closes.columns}, index=rets.index)
eur_ret = panels["EURUSD"]["close"].pct_change()
lib["eurusd_beta_60d"] = rolling_beta(rets, eur_ret, 60, 40)
cn_ret = panels["CN10Y"]["close"].pct_change()
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn_ret, 60, 40)
dn = mkt.where(mkt < 0, 0.0)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, dn, 60, 40)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
vix = panels["VIX"]["close"]
vix_ret = vix.pct_change()
lib["vix_beta_cond_60x20"] = -rolling_beta(rets, vix_ret, 60, 40) * (vix / vix.shift(20) - 1.0)
usdcny_ret = panels["USDCNY"]["close"].pct_change()
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny_ret, 60, 40)
def rsi14(s, win=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(win).mean()
    dn_ = (-d.clip(upper=0)).rolling(win).mean()
    return 100 - 100 / (1 + up / dn_.replace(0, np.nan))
lib["rsi_14"] = per_asset(rsi14)

# pre-stack library values once
lib_stack = {}
for name, lsig in lib.items():
    st = lsig.stack().dropna()
    if len(st):
        lib_stack[name] = st
print(f"library signals: {list(lib.keys())}", flush=True)

def max_lib_corr(cand):
    st = cand.stack().dropna()
    best, best_key = 0.0, None
    for name, ls in lib_stack.items():
        both = pd.concat([st.rename("cand"), ls.rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key

def evaluate(tag, panel, expected_sign=1):
    rec = {"tag": tag}
    try:
        ics = rank_ic_vec(panel, FWDS[H_ADM])
        m = summarize_ic(ics, expected_sign)
        m.update(coverage_metrics(panel, min_valid=MIN_VALID))
        m["turnover_10d_rank"] = turnover_rank(panel, 10)
        m["decay_ic_by_horizon"] = {}
        for h in HORIZONS:
            ics_h = rank_ic_vec(panel, FWDS[h])
            m["decay_ic_by_horizon"][str(h)] = round(float(ics_h.mean()), 4) if len(ics_h) else None
        corr, key = max_lib_corr(panel)
        m["max_abs_library_correlation"] = corr
        m["max_corr_factor"] = key
        m["admitted"] = bool(abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR)
        rec.update(m)
        print(f"[{tag}] IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.2f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.2f} geo8={m['coverage_dates_ge8']:.2f} "
              f"turn={m['turnover_10d_rank']} libcorr={corr}({key}) admit={m['admitted']}")
        print(f"     decay={m['decay_ic_by_horizon']}", flush=True)
    except Exception as e:
        rec["error"] = repr(e)
        print(f"[{tag}] ERROR {e!r}", flush=True)
    return rec

results = {}

# ---- A) active library drift ----
print("\n=== A. ACTIVE LIBRARY RE-VALIDATION ===", flush=True)
for name, sig in lib.items():
    if name in ("vol_price_corr_20", "eurusd_beta_60d", "rate_beta_cn10y_60d", "dn_mkt_beta_60d"):
        exp = -1 if name in ("eurusd_beta_60d", "rate_beta_cn10y_60d") else 1
        results["active_" + name] = evaluate("active " + name, sig, exp)

# ---- B) batch D candidates ----
print("\n=== B. BATCH D CANDIDATES ===", flush=True)
candsD = {}
candsD["rel_mom_60d"] = (closes.shift(5) / closes.shift(65) - 1.0) - (mkt.shift(5) / mkt.shift(65) - 1.0)
candsD["skew_20d"] = rets.rolling(20).skew()
candsD["drawdown_60d"] = closes / closes.rolling(60).max() - 1.0
vol20 = rets.rolling(20).std()
candsD["vol_adj_mom_20x20"] = (closes / closes.shift(20) - 1.0) / vol20
up = mkt.where(mkt > 0, 0.0)
up_beta = rolling_beta(rets, up, 60, 40)
dn_beta = rolling_beta(rets, dn, 60, 40)
candsD["capture_ratio_60d"] = up_beta / dn_beta.replace(0, np.nan)
def autocorr5(s):
    r = s.pct_change()
    return r.rolling(20).apply(lambda x: np.corrcoef(x[:-5], x[5:])[0, 1] if len(x) >= 12 else np.nan, raw=True)
candsD["autocorr_5d_20w"] = per_asset(autocorr5)
candsD["trend_consistency_20d"] = (rets > 0).rolling(20).mean()
obv = (np.sign(rets) * vol_panel).fillna(0.0).cumsum()
candsD["obv_slope_20d"] = obv - obv.shift(20)
candsD["vol_ratio_5x60"] = rets.rolling(5).std() / rets.rolling(60).std()
def range_pos(s, win=20):
    hi = s.rolling(win).max(); lo = s.rolling(win).min()
    return (s - lo) / (hi - lo).replace(0, np.nan)
candsD["range_pos_20d"] = per_asset(lambda s: range_pos(s, 20))
vol60 = rets.rolling(60).std()
candsD["mom60_vol60"] = (closes.shift(5) / closes.shift(65) - 1.0) / vol60
candsD["downside_vol_20d"] = rets.where(rets < 0, 0.0).rolling(20).std()
for name, panel in candsD.items():
    results["D_" + name] = evaluate("D " + name, panel, 1)

# ---- C) batch E candidates ----
print("\n=== C. BATCH E NEW CANDIDATES ===", flush=True)
candsE = {}
vwap20 = (closes * vol_panel).rolling(20).sum() / vol_panel.rolling(20).sum().replace(0, np.nan)
candsE["vwap_break_20d"] = (closes - vwap20) / (vol20 * closes)
def body_pos(s, win=20):
    hi = s.rolling(2).max(); lo = s.rolling(2).min()
    pos = (s - lo) / (hi - lo).replace(0, np.nan)
    return pos.rolling(win).mean()
candsE["body_pos_20d"] = per_asset(lambda s: body_pos(s, 20))
candsE["trend_gap_5x60"] = (closes / closes.shift(6) - 1.0) - (closes.shift(5) / closes.shift(65) - 1.0)
us10y_ret = panels["US10Y"]["close"].pct_change()
candsE["rate_beta_us10y_60d"] = rolling_beta(rets, us10y_ret, 60, 40)
cu_ret = panels["COPPER"]["close"].pct_change()
candsE["copper_beta_60d"] = rolling_beta(rets, cu_ret, 60, 40)
btc_ret = panels["BTC"]["close"].pct_change()
candsE["btc_beta_60d"] = rolling_beta(rets, btc_ret, 60, 40)
mom20 = closes / closes.shift(20) - 1.0
candsE["rel_strength_z20"] = (mom20 - mom20.mean(axis=1)) / mom20.std(axis=1).replace(0, np.nan)
candsE["vol_surge_5x60"] = vol_panel.rolling(5).mean() / vol_panel.rolling(60).mean().replace(0, np.nan)
def ulcer(s, win=60):
    dd = s / s.rolling(win, min_periods=40).max() - 1.0
    return np.sqrt((dd ** 2).rolling(win).mean())
candsE["ulcer_60d"] = per_asset(lambda s: ulcer(s, 60))
intraday = (closes / opens - 1.0)
candsE["intraday_rev_10d"] = -intraday.rolling(10).mean()
candsE["vol_term_20x60"] = vol20 / vol60
candsE["mom5_vol20"] = (closes / closes.shift(6) - 1.0) / vol20
for name, panel in candsE.items():
    results["E_" + name] = evaluate("E " + name, panel, 1)

print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
with open("scripts/_miner3_batchE_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("saved scripts/_miner3_batchE_results.json", flush=True)
