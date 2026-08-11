"""miner_3 batch F screen (2026-12-07) - vectorized rank-IC on data through 2026-12-04.

A) drift re-validation of 4 active library factors
B) new batch F candidates (volatility shape, trend quality, volume/liquidity,
   cross-asset betas, intraday shape)

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe).
Robustness: full-period + sub-period split + recent window excluding frozen HSI/ETH.
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
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840
mkt = rets.mean(axis=1)

FWDS = {h: forward_returns(closes, h) for h in HORIZONS}

def rank_ic_vec(factor_panel, fwd):
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

# ---------------- library signals (active + evicted, for corr audit) ----------------
print("building library signals...", flush=True)
lib = {}
lib["vol_price_corr_20"] = pd.DataFrame(
    {a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)
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
lib["rsi_14"] = pd.DataFrame({a: rsi14(closes[a]) for a in closes.columns}).reindex(closes.index)

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

def evaluate(tag, panel, expected_sign=1, extra=None):
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
        # sub-period stability: first 60% vs last 40% of IC dates
        if len(ics) >= 60:
            n = len(ics)
            first = ics.iloc[:int(n*0.6)]
            last = ics.iloc[int(n*0.6):]
            m["ic_first60"] = round(float(first.mean()), 4)
            m["ic_last40"] = round(float(last.mean()), 4)
            m["icir_last40"] = round(float(last.mean()/last.std(ddof=1)), 4) if last.std(ddof=1) > 0 else 0.0
        else:
            m["ic_first60"] = None
            m["ic_last40"] = None
            m["icir_last40"] = None
        # recent window (last 60 trading days) excluding frozen HSI/ETH
        recent_panel = panel.tail(60).drop(columns=["HSI", "ETH"], errors="ignore")
        recent_fwd = FWDS[H_ADM].tail(60).drop(columns=["HSI", "ETH"], errors="ignore")
        ics_r = rank_ic_vec(recent_panel, recent_fwd)
        m["ic_recent60_excl_frozen"] = round(float(ics_r.mean()), 4) if len(ics_r) >= 8 else None
        m["n_recent60"] = int(len(ics_r))
        rec.update(m)
        print(f"[{tag}] IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']} "
              f"cov={m['coverage_asset_days']:.2f} geo8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']} "
              f"libcorr={corr}({key}) first60={m['ic_first60']} last40={m['ic_last40']}({m['icir_last40']}) "
              f"recent60xf={m['ic_recent60_excl_frozen']}(n={m['n_recent60']}) admit={m['admitted']}")
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
        results[f"active_{name}"] = evaluate(f"active_{name}", sig, expected_sign=exp)

# ---- B) batch F candidates ----
print("\n=== B. BATCH F CANDIDATES ===", flush=True)
cands = {}

# volatility / risk shape
cands["F_skew_20d"] = rets.rolling(20).skew()
cands["F_kurt_20d"] = rets.rolling(20).kurt()
cands["F_var_ratio_20x60"] = rets.rolling(20).var() / rets.rolling(60).var() - 1.0
cands["F_vol_term_20x60"] = rets.rolling(20).std() / rets.rolling(60).std() - 1.0
cands["F_dn_vol_share_20d"] = rets.clip(upper=0).rolling(20).std() / rets.rolling(20).std()
cands["F_updown_ratio_20d"] = rets.clip(lower=0).rolling(20).std() / rets.clip(upper=0).rolling(20).std().replace(0, np.nan)
cands["F_max5d_60d"] = rets.rolling(5).sum().rolling(60).max()
cands["F_min5d_60d"] = rets.rolling(5).sum().rolling(60).min()
cands["F_autocorr1_20d"] = rets.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False)

# trend / mean-reversion
cands["F_zprice_60d"] = (closes - closes.rolling(60).mean()) / closes.rolling(60).std()
cands["F_dist_high_250d"] = closes / closes.rolling(250).max() - 1.0
cands["F_eff_ratio_60d"] = (closes - closes.shift(60)).abs() / rets.abs().rolling(60).sum()
cands["F_trend_cons_20d"] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
cands["F_mom60_voladj"] = (closes / closes.shift(60) - 1.0) / rets.rolling(60).std()

# volume / liquidity
cands["F_amihud_20d"] = (rets.abs() / vol_panel).rolling(20).mean()
cands["F_vol_zscore_20d"] = vol_panel / vol_panel.rolling(20).mean() - 1.0
cands["F_vol_price_corr_60d"] = pd.DataFrame(
    {a: rets[a].rolling(60).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)
obv = (np.sign(rets) * vol_panel).fillna(0.0)
cands["F_obv_slope_20d"] = obv.rolling(20).mean() / vol_panel.rolling(20).mean()

# cross-asset betas
usdjpy_ret = panels["USDJPY"]["close"].pct_change()
cands["F_usdjpy_beta_60d"] = rolling_beta(rets, usdjpy_ret, 60, 40)
dxy_ret = panels["DXY"]["close"].pct_change()
cands["F_dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60, 40)
xau_ret = rets["XAU"]
cands["F_xau_beta_60d"] = rolling_beta(rets, xau_ret, 60, 40)
wti_ret = rets["WTI"]
cands["F_wti_beta_60d"] = rolling_beta(rets, wti_ret, 60, 40)
cop_ret = rets["COPPER"]
cands["F_copper_beta_60d"] = rolling_beta(rets, cop_ret, 60, 40)
btc_ret = rets["BTC"]
cands["F_btc_beta_60d"] = rolling_beta(rets, btc_ret, 60, 40)
ndx_ret = rets["NDX"]
cands["F_ndx_beta_60d"] = rolling_beta(rets, ndx_ret, 60, 40)

# intraday shape
body = (closes - opens).abs() / (highs - lows).replace(0, np.nan)
cands["F_body_ratio_20d"] = body.rolling(20).mean()
upper = (highs - pd.concat([opens, closes], axis=1).max(axis=1)) / closes
lower = (pd.concat([opens, closes], axis=1).min(axis=1) - lows) / closes
cands["F_upper_shadow_20d"] = upper.rolling(20).mean()
cands["F_lower_shadow_20d"] = lower.rolling(20).mean()
gap = opens / closes.shift(1) - 1.0
cands["F_gap_20d"] = gap.rolling(20).mean()

for tag, panel in cands.items():
    results[tag] = evaluate(tag, panel)

with open("scripts/_miner3_batchF_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\nDONE {time.time()-t0:.1f}s -> scripts/_miner3_batchF_results.json", flush=True)
