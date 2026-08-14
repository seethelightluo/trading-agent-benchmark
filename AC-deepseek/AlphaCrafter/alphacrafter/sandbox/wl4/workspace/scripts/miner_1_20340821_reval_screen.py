"""miner_1 (2034-08-21): (A) re-validate the 3 currently EFFECTIVE library factors
through the latest visible trading day; (B) screen ~22 novel candidate factors
across fresh families (skew/return-shape, capture asymmetry, streak/win-rate,
60d trend quality, sector/commodity/currency beta tilts, overnight/intraday
momentum, vol term-structure at short end, tail ratios).

Admission gate (h=10, 15-asset cross-section, min_valid=8):
    |IC| >= 0.0070 and |ICIR| >= 0.0840
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

log("loading panels...")
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt = rets.mean(axis=1)
log(f"closes {closes.shape} {closes.index.min().date()} -> {closes.index.max().date()}")

highs = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lows = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
opens = pd.concat({a: panels[a]["open"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()

mom5 = closes / closes.shift(5) - 1.0
mom10 = closes / closes.shift(10) - 1.0
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol10 = rets.rolling(10).std(ddof=0)
vol20 = rets.rolling(20).std(ddof=0)
vol60 = rets.rolling(60).std(ddof=0)

# ---------------- Part A: library re-validation ----------------
dn = np.minimum(mkt, 0.0)
f2 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                       / dn.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
f3 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                       / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)
f1 = (mom20 - mom60) / vol20

lib_factors = {
    "vol_adj_mom_accel_20x60": f1,
    "dn_mkt_beta_60d": f2,
    "rate_beta_cn10y_60d": f3,
}
fwd10 = forward_returns(closes, 10)
ADM_IC, ADM_ICIR = 0.0070, 0.0840

print("=" * 100)
print("PART A: LIBRARY RE-VALIDATION (h=10) through", closes.index.max().date())
print("=" * 100)
reval = {}
for name, fp in lib_factors.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    full = summarize_ic(ics, expected_sign=1)
    rec = {}
    for label, start in [("since2022", "2022-01-01"), ("since2024", "2024-01-01"),
                         ("since2024-08", "2024-08-01"), ("last500d", None)]:
        sub = ics[ics.index >= pd.Timestamp(start)] if start else ics.iloc[-500:]
        rec[label] = summarize_ic(sub, expected_sign=1)
    dec = decay_profile(fp, closes)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    p_full = abs(full["ic"]) >= ADM_IC and abs(full["icir"]) >= ADM_ICIR
    p_rec = any(abs(v["ic"]) >= ADM_IC and abs(v["icir"]) >= ADM_ICIR for v in rec.values())
    reval[name] = {"full": full, "recent": rec, "decay": dec, "cov": cov,
                   "turn": turn, "pass_full": p_full, "pass_recent": p_rec}
    print(f"FACTOR {name}")
    print(f"  FULL: IC={full['ic']:+.4f} ICIR={full['icir']:+.3f} hit={full['ic_hit_ratio']:.2f} n={full['n_ic_dates']} -> {'PASS' if p_full else 'FAIL'}")
    for k, v in rec.items():
        print(f"  {k:12s}: IC={v['ic']:+.4f} ICIR={v['icir']:+.3f} hit={v['ic_hit_ratio']:.2f} n={v['n_ic_dates']:5d} -> {'PASS' if (abs(v['ic'])>=ADM_IC and abs(v['icir'])>=ADM_ICIR) else 'fail'}")
    print(f"  decay: { {k: round(v,4) for k,v in dec.items()} }  covA={cov['coverage_asset_days']} covD={cov['coverage_dates_ge8']} turn={turn}")

# ---------------- Part B: novel candidates ----------------
C = {}

# --- return-shape / asymmetry family ---
C["skew_60d"] = rets.rolling(60, min_periods=40).skew()
C["win_rate_60d"] = (rets > 0).rolling(60, min_periods=40).mean()
up = rets.where(rets > 0, np.nan)
dn_ret = rets.where(rets < 0, np.nan)
C["up_capture_60d"] = up.rolling(60, min_periods=40).mean() / (dn_ret.rolling(60, min_periods=40).mean().abs() + 1e-12)
def max_streak_pos(s, win=60):
    out = pd.Series(np.nan, index=s.index)
    pos = (s > 0).astype(int)
    for i in range(win - 1, len(s)):
        out.iloc[i] = _longest_run(pos.iloc[i - win + 1:i + 1].values)
    return out
def _longest_run(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
C["max_up_streak_60d"] = pd.DataFrame({a: max_streak_pos(rets[a]) for a in rets.columns}, index=rets.index)
q95 = rets.rolling(60, min_periods=40).quantile(0.95)
q05 = rets.rolling(60, min_periods=40).quantile(0.05)
C["tail_ratio_60d"] = q95 / (q05.abs() + 1e-12)
C["max_daily_ret_20d"] = rets.rolling(20).max()

# --- trend quality / consistency ---
def trend_r2(r, win=60):
    x = np.arange(win)
    def _r2(y):
        if np.isnan(y).any():
            return np.nan
        b = np.polyfit(x, y, 1)
        pred = np.polyval(b, x)
        ss_res = np.nansum((y - pred) ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r.rolling(win).apply(_r2, raw=True)
C["trend_r2_60d"] = trend_r2(closes, 60)
C["sharpe_60d"] = mom60 / (vol60 + 1e-12)

# --- cross-asset beta tilts (novel references) ---
def rolling_beta(a, b, win=60, minp=40):
    return (a.rolling(win, min_periods=minp).cov(b)
            / b.rolling(win, min_periods=minp).var())
ndx_ret = rets["NDX"]
xau_ret = rets["XAU"]
wti_ret = rets["WTI"]
btc_ret = rets["BTC"]
C["beta_ndx_60d"] = pd.DataFrame({a: rolling_beta(rets[a], ndx_ret) for a in rets.columns}, index=rets.index)
C["beta_xau_60d"] = pd.DataFrame({a: rolling_beta(rets[a], xau_ret) for a in rets.columns}, index=rets.index)
C["btc_mom_tilt_60d"] = pd.DataFrame({a: rolling_beta(rets[a], btc_ret) for a in rets.columns}, index=rets.index) * (closes["BTC"] / closes["BTC"].shift(60) - 1.0)
C["wti_mom_tilt_60d"] = pd.DataFrame({a: rolling_beta(rets[a], wti_ret) for a in rets.columns}, index=rets.index) * (closes["WTI"] / closes["WTI"].shift(60) - 1.0)

# --- currency / rate macro tilts ---
dxy = panels["DXY"]["close"].astype(float)
dxy_ret = dxy.pct_change()
C["dxy_mom_beta_tilt"] = -pd.DataFrame({a: rolling_beta(rets[a], dxy_ret) for a in rets.columns}, index=rets.index) * (dxy / dxy.shift(20) - 1.0)
us10y_ret = rets["US10Y"]
spread = closes["US10Y"] - closes["CN10Y"]
spread_chg = spread.diff()
C["spread_beta_uscn_60d"] = pd.DataFrame({a: rolling_beta(rets[a], spread_chg) for a in rets.columns}, index=rets.index)
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
vix_level_ratio = vix / vix.rolling(60).mean()
C["vix_level_beta_tilt"] = -pd.DataFrame({a: rolling_beta(rets[a], vix_ret) for a in rets.columns}, index=rets.index) * vix_level_ratio

# --- short-horizon momentum / microstructure ---
C["roll_mom_5_10"] = (mom5 - mom10) / (vol10 + 1e-12)
C["vol_ratio_5_20"] = vol10 / vol20
prev_close = closes.shift(1)
C["overnight_gap_20d"] = ((opens - prev_close) / prev_close).rolling(20).mean()
C["intraday_mom_20d"] = ((closes - opens) / opens).rolling(20).mean()
C["atr_ratio_10_60"] = (highs - lows).rolling(10).mean() / ((highs - lows).rolling(60).mean() + 1e-12)

# --- volume/participation ---
C["volume_trend_20_60"] = vol_panel.rolling(20).mean() / (vol_panel.rolling(60).mean() + 1e-12)
C["ret_vol_corr_20d"] = pd.DataFrame(
    {a: rets[a].rolling(20).corr(vol_panel[a].pct_change()) for a in rets.columns},
    index=rets.index)

log("building library signals for correlation check...")
library = library_signals(panels, closes, rets, vix)
library["vol_adj_mom_accel_20x60"] = f1
library["dn_mkt_beta_60d"] = f2
library["rate_beta_cn10y_60d"] = f3

print("=" * 100)
print("PART B: NOVEL CANDIDATE SCREEN (h=10, full history)")
print("=" * 100)
hdr = (f"{'candidate':26s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covA':>5s} "
       f"{'covD':>5s} {'turn':>5s} {'libCorr':>7s} {'d5':>7s} {'d10':>7s} {'d20':>7s} {'pass':>4s}")
print(hdr)
print("-" * 120)

results = {}
for name, fp in C.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    if len(ics) == 0:
        print(f"{name:26s} no IC dates")
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    dec = decay_profile(fp, closes)
    corr, key = max_library_corr(fp, library)
    p = abs(m["ic"]) >= ADM_IC and abs(m["icir"]) >= ADM_ICIR
    results[name] = {"metrics": m, "coverage": cov, "turnover": turn,
                     "decay": dec, "corr": corr, "corr_key": key, "pass": p}
    print(f"{name:26s} {m['ic']:8.4f} {m['icir']:7.3f} {m['ic_hit_ratio']:5.2f} "
          f"{m['n_ic_dates']:5d} {cov['coverage_asset_days']:5.2f} {cov['coverage_dates_ge8']:5.2f} "
          f"{turn:5.2f} {corr:7.3f} {dec.get('5', float('nan')):7.4f} {dec.get('10', float('nan')):7.4f} "
          f"{dec.get('20', float('nan')):7.4f} {'YES' if p else ''}")

print("-" * 120)
print("PASSING CANDIDATES (full history):")
for name, r in results.items():
    if r["pass"]:
        print(f"  {name}: IC={r['metrics']['ic']:.4f} ICIR={r['metrics']['icir']:.3f} "
              f"hit={r['metrics']['ic_hit_ratio']:.2f} n={r['metrics']['n_ic_dates']} "
              f"libCorr={r['corr']:.3f}({r['corr_key']}) decay10={r['decay'].get('10')}")

print("=" * 100)
print("RECENT-WINDOW ROBUSTNESS for |IC|>=0.005 full-history candidates")
print("=" * 100)
for name, r in results.items():
    if abs(r["metrics"]["ic"]) < 0.005:
        continue
    fp = C[name].replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    for label, start in [("since2024", "2024-01-01"), ("since2024-08", "2024-08-01")]:
        sub = ics[ics.index >= pd.Timestamp(start)]
        if len(sub) < 20:
            continue
        rm = summarize_ic(sub, expected_sign=1)
        rp = abs(rm["ic"]) >= ADM_IC and abs(rm["icir"]) >= ADM_ICIR
        print(f"{name:26s} {label:12s} IC={rm['ic']:+.4f} ICIR={rm['icir']:+.3f} "
              f"hit={rm['ic_hit_ratio']:.2f} n={rm['n_ic_dates']:4d} pass={'YES' if rp else ''}")

log("done")
