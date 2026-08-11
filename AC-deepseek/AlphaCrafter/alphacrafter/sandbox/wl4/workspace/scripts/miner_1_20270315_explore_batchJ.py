"""miner_1 2027-03-15: batch J exploration + library drift re-validation.

A) drift re-validation of 4 active library factors (full + recent 500/250)
B) new batch J candidates (fresh, interpretable, low overlap with prior batches):
   - risk-adjusted momentum: risk_adj_mom_60d, ma_ratio_10x50
   - vol/risk-shape: skew_60d, maxdd_60d, downside_vol_ratio_60d, vol_ratio_10x60
   - intraday structure: range_pos_20d, gap_overnight_share_20d
   - cross-asset beta/regime: corr_us10y_60d, btc_beta_60d, wti_beta_60d, xau_beta_60d
   - relative value: rel_strength_20d, obv_trend_20d, crypto_mom_20d
   - longer vol-price corr window: vol_price_corr_60d (correlation probe)

Gate (h=10): |IC| >= 0.0070 AND |ICIR| >= 0.0840, n_ic_dates >= 200,
max_abs_library_correlation < 0.5 (vs 4 active library signals).
Data through previous completed trading day (2027-03-12), no lookahead.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, coverage_metrics, turnover_rank,
                                 max_library_corr, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = ret_panel(panels)
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
H_ADM = 10
MIN_VALID = 8
MIN_IC_DATES = 200
GATE_IC, GATE_ICIR = 0.0070, 0.0840
print(f"panels={len(panels)} closes={closes.shape} "
      f"{closes.index[0].date()}..{closes.index[-1].date()} load={time.time()-t0:.1f}s", flush=True)
LAST = closes.index.max()
print("last completed trading day:", LAST.date(), flush=True)


def rank_ic_series_vec(factor_panel, fwd, min_valid=MIN_VALID):
    fr = factor_panel.rank(axis=1, method="average", na_option="keep")
    rr = fwd.rank(axis=1, method="average", na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1).astype(float)
    f = fr.where(valid)
    r = rr.where(valid)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = fm.mul(rm).sum(axis=1, min_count=1)
    den = np.sqrt(fm.pow(2).sum(axis=1, min_count=1) * rm.pow(2).sum(axis=1, min_count=1))
    ic = num.div(den.replace(0, np.nan))
    ic = ic.where((n >= min_valid) & (n >= 2) & (den > 1e-14))
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def summarize(ic_series):
    ic = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    return {"ic": round(ic, 4), "icir": round(ic / std, 4) if std > 0 else 0.0,
            "ic_hit_ratio": round(float((np.sign(ic_series) != 0).mean()), 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(std, 4)}


def rolling_beta_vec(asset_ret, driver_ret, win=60, min_obs=40):
    m = driver_ret
    m_mean = m.rolling(win, min_periods=min_obs).mean()
    a_mean = asset_ret.rolling(win, min_periods=min_obs).mean()
    cov_am = asset_ret.mul(m, axis=0).rolling(win, min_periods=min_obs).mean() - a_mean * m_mean
    var_m = m.rolling(win, min_periods=min_obs).var()
    return cov_am.div(var_m.replace(0, np.nan))


def rolling_corr_vec(a, b, win=60, min_obs=40):
    am = a.rolling(win, min_periods=min_obs).mean()
    bm = b.rolling(win, min_periods=min_obs).mean()
    cov = a.mul(b, axis=0).rolling(win, min_periods=min_obs).mean() - am * bm
    va = a.rolling(win, min_periods=min_obs).var()
    vb = b.rolling(win, min_periods=min_obs).var()
    return cov.div(np.sqrt(va.mul(vb)).replace(0, np.nan))


# ---------------- library signals (recompute from definitions) ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0)
LIBRARY = {
    "vol_price_corr_20": pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index),
    "dn_mkt_beta_60d": rolling_beta_vec(rets, dn, 60, 40),
    "eurusd_beta_60d": rolling_beta_vec(rets, panels["EURUSD"]["close"].pct_change(), 60, 40),
    "rate_beta_cn10y_60d": rolling_beta_vec(rets, rets["CN10Y"], 60, 40),
}

# ---------------- new candidates ----------------
C = {}
C["risk_adj_mom_60d"] = (closes / closes.shift(60) - 1.0).div(rets.rolling(60).std().replace(0, np.nan))
C["ma_ratio_10x50"] = closes.rolling(10).mean() / closes.rolling(50).mean() - 1.0
C["skew_60d"] = rets.rolling(60).skew()
C["maxdd_60d"] = 1.0 - closes / closes.rolling(60).max()
C["downside_vol_ratio_60d"] = (rets.where(rets < 0, 0.0).rolling(60).std()
                               .div(rets.rolling(60).std().replace(0, np.nan)))
C["vol_ratio_10x60"] = rets.rolling(10).std().div(rets.rolling(60).std().replace(0, np.nan))
C["range_pos_20d"] = ((closes - lows) / (highs - lows).replace(0, np.nan)).rolling(20).mean()
prev_close = closes.shift(1)
overnight = (opens - prev_close).abs()
intraday = (closes - opens).abs()
C["gap_overnight_share_20d"] = (overnight / (overnight + intraday).replace(0, np.nan)).rolling(20).mean()
C["corr_us10y_60d"] = rolling_corr_vec(rets, rets["US10Y"], 60, 40)
C["btc_beta_60d"] = rolling_beta_vec(rets, rets["BTC"], 60, 40)
C["wti_beta_60d"] = rolling_beta_vec(rets, rets["WTI"], 60, 40)
C["xau_beta_60d"] = rolling_beta_vec(rets, rets["XAU"], 60, 40)
C["rel_strength_20d"] = (closes / closes.shift(20) - 1.0).sub(
    (closes / closes.shift(20) - 1.0).median(axis=1), axis=0)
obv = np.sign(rets).mul(vol_panel).cumsum()
C["obv_trend_20d"] = (obv - obv.shift(20)).div(vol_panel.rolling(60).mean().replace(0, np.nan))
C["crypto_mom_20d"] = pd.DataFrame({a: rets["BTC"].rolling(20).sum() for a in closes.columns}, index=rets.index)
C["vol_price_corr_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)

fwd = forward_returns(closes, H_ADM)
print("---- A) LIBRARY DRIFT RE-VALIDATION (h=10) ----", flush=True)
lib_results = {}
for name, sig in LIBRARY.items():
    ic_full = rank_ic_series_vec(sig, fwd)
    s_full = summarize(ic_full)
    s_recent = summarize(ic_full[ic_full.index >= ic_full.index[-1] - pd.Timedelta(days=500)])
    s_recent250 = summarize(ic_full[ic_full.index >= ic_full.index[-1] - pd.Timedelta(days=250)])
    lib_results[name] = s_full
    print(f"{name:22s} full IC={s_full['ic']:+.4f} ICIR={s_full['icir']:+.3f} n={s_full['n_ic_dates']} | "
          f"recent500 IC={s_recent['ic']:+.4f} ICIR={s_recent['icir']:+.3f} n={s_recent['n_ic_dates']} | "
          f"recent250 IC={s_recent250['ic']:+.4f} ICIR={s_recent250['icir']:+.3f} n={s_recent250['n_ic_dates']}", flush=True)

print("\n---- B) BATCH J CANDIDATES (h=10) ----", flush=True)
results = {}
for name, sig in C.items():
    ic_full = rank_ic_series_vec(sig, fwd)
    if len(ic_full) < MIN_IC_DATES:
        print(f"{name:24s} SKIP n_ic_dates={len(ic_full)}", flush=True)
        continue
    s = summarize(ic_full)
    s_r = summarize(ic_full[ic_full.index >= ic_full.index[-1] - pd.Timedelta(days=500)])
    cov = coverage_metrics(sig)
    turn = turnover_rank(sig, step=10)
    rho, rho_key = max_library_corr(sig, LIBRARY)
    results[name] = {**s, "recent500_ic": s_r["ic"], "recent500_icir": s_r["icir"],
                     "coverage": cov, "turnover_10d_rank": turn,
                     "max_abs_library_correlation": rho, "max_corr_factor": rho_key}
    flag = ""
    if abs(s["ic"]) >= GATE_IC and abs(s["icir"]) >= GATE_ICIR and rho < 0.5:
        flag = "  <== PASSES GATE"
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} | "
          f"r500 IC={s_r['ic']:+.4f} ICIR={s_r['icir']:+.3f} | rho={rho:.3f}({rho_key}) | "
          f"turn={turn:.2f} cov={cov['coverage_asset_days']:.2f}/{cov['coverage_dates_ge8']:.2f}{flag}", flush=True)

print("\n---- C) DECAY PROFILE (mean IC by horizon) for passers & near-miss ----", flush=True)
for name, s in results.items():
    if (abs(s["ic"]) >= GATE_IC and abs(s["icir"]) >= GATE_ICIR) or abs(s["ic"]) >= 0.02:
        dec = {}
        for h in (1, 2, 3, 5, 10, 20):
            ic = rank_ic_series_vec(C[name], forward_returns(closes, h))
            dec[h] = round(float(ic.mean()), 4) if len(ic) else None
        print(f"{name:24s} decay=" + " ".join(f"h{h}:{v}" for h, v in dec.items()), flush=True)

with open("scripts/miner_1_20270315_batchJ_results.json", "w") as f:
    json.dump({"last_date": str(LAST.date()), "lib": lib_results, "candidates": results}, f, indent=1, default=str)
print("\nsaved scripts/miner_1_20270315_batchJ_results.json", flush=True)
print(f"total runtime {time.time()-t0:.1f}s", flush=True)
