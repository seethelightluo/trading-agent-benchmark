"""miner_1 (2034-09-04) PART B: novel candidate screen with FAST vectorized rank-IC.

Avoids the timeout that killed the 2034-08-21 screen:
  * pre-ranks factor + fwd panels once, then per-date numpy corrcoef
  * trend_r2 computed via rolling corr with time index (vectorized)
  * 22+ candidates across return-shape, trend-quality, cross-asset beta,
    macro tilts, short-horizon microstructure, volume families

Admission gate (h=10, 15-asset cross-section, min_valid=8):
    |IC| >= 0.0070 and |ICIR| >= 0.0840
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, library_signals,
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
vol5 = rets.rolling(5).std(ddof=0)
vol10 = rets.rolling(10).std(ddof=0)
vol20 = rets.rolling(20).std(ddof=0)
vol60 = rets.rolling(60).std(ddof=0)

def fast_rank_ic(fp, fwd, min_valid=8):
    """Spearman IC via pre-ranked panels + numpy per-date corrcoef (fast)."""
    fr = fp.rank(axis=1)
    rr = fwd.rank(axis=1)
    dates, ics = [], []
    for dt in fp.index:
        if dt not in rr.index:
            continue
        f = fr.loc[dt].values
        r = rr.loc[dt].values
        mask = ~(np.isnan(f) | np.isnan(r))
        if mask.sum() < min_valid:
            continue
        fv, rv = f[mask], r[mask]
        if np.std(fv) < 1e-14 or np.std(rv) < 1e-14:
            continue
        ic = np.corrcoef(fv, rv)[0, 1]
        if not np.isnan(ic):
            dates.append(dt); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")

def fast_summarize(ics, expected_sign=1):
    if len(ics) == 0:
        return {"ic": 0.0, "icir": 0.0, "ic_hit_ratio": 0.0, "n_ic_dates": 0, "ic_std": 0.0}
    sd = ics.std(ddof=1)
    return {"ic": round(float(ics.mean()), 4),
            "icir": round(float(ics.mean() / sd if sd > 0 else 0.0), 4),
            "ic_hit_ratio": round(float((np.sign(ics) == expected_sign).mean()), 3),
            "n_ic_dates": int(len(ics)),
            "ic_std": round(float(sd), 4)}

ADM_IC, ADM_ICIR = 0.0070, 0.0840
fwd10 = forward_returns(closes, 10)
fwd10r = fwd10.rank(axis=1)

log("building candidates...")
C = {}

# --- return-shape / asymmetry family ---
C["skew_60d"] = rets.rolling(60, min_periods=40).skew()
C["win_rate_60d"] = (rets > 0).rolling(60, min_periods=40).mean()
up = rets.where(rets > 0, np.nan)
dn_ret = rets.where(rets < 0, np.nan)
C["up_capture_60d"] = up.rolling(60, min_periods=40).mean() / (dn_ret.rolling(60, min_periods=40).mean().abs() + 1e-12)
q95 = rets.rolling(60, min_periods=40).quantile(0.95)
q05 = rets.rolling(60, min_periods=40).quantile(0.05)
C["tail_ratio_60d"] = q95 / (q05.abs() + 1e-12)
C["max_daily_ret_20d"] = rets.rolling(20).max()

# --- trend quality / consistency ---
t_idx = np.arange(len(closes))
C["trend_r2_60d"] = closes.rolling(60, min_periods=40).corr(pd.Series(t_idx, index=closes.index)) ** 2
C["sharpe_60d"] = mom60 / (vol60 + 1e-12)

# --- cross-asset beta tilts ---
def rolling_beta(a, b, win=60, minp=40):
    return (a.rolling(win, min_periods=minp).cov(b) / b.rolling(win, min_periods=minp).var())
ndx_ret, xau_ret, wti_ret, btc_ret = rets["NDX"], rets["XAU"], rets["WTI"], rets["BTC"]
C["beta_ndx_60d"] = pd.DataFrame({a: rolling_beta(rets[a], ndx_ret) for a in rets.columns}, index=rets.index)
C["beta_xau_60d"] = pd.DataFrame({a: rolling_beta(rets[a], xau_ret) for a in rets.columns}, index=rets.index)
C["btc_mom_tilt_60d"] = pd.DataFrame({a: rolling_beta(rets[a], btc_ret) for a in rets.columns}, index=rets.index) * (closes["BTC"] / closes["BTC"].shift(60) - 1.0)
C["wti_mom_tilt_60d"] = pd.DataFrame({a: rolling_beta(rets[a], wti_ret) for a in rets.columns}, index=rets.index) * (closes["WTI"] / closes["WTI"].shift(60) - 1.0)

# --- currency / rate / vol macro tilts ---
dxy = panels["DXY"]["close"].astype(float); dxy_ret = dxy.pct_change()
C["dxy_mom_beta_tilt"] = -pd.DataFrame({a: rolling_beta(rets[a], dxy_ret) for a in rets.columns}, index=rets.index) * (dxy / dxy.shift(20) - 1.0)
us10y_ret = rets["US10Y"]
spread = closes["US10Y"] - closes["CN10Y"]; spread_chg = spread.diff()
C["spread_beta_uscn_60d"] = pd.DataFrame({a: rolling_beta(rets[a], spread_chg) for a in rets.columns}, index=rets.index)
vix = panels["VIX"]["close"].astype(float); vix_ret = vix.pct_change()
vix_level_ratio = vix / vix.rolling(60).mean()
C["vix_level_beta_tilt"] = -pd.DataFrame({a: rolling_beta(rets[a], vix_ret) for a in rets.columns}, index=rets.index) * vix_level_ratio
usd_cn = panels["USDCNY"]["close"].astype(float); usdcny_ret = usd_cn.pct_change()
C["usdcny_beta_tilt"] = pd.DataFrame({a: rolling_beta(rets[a], usdcny_ret) for a in rets.columns}, index=rets.index) * (usd_cn / usd_cn.shift(20) - 1.0)

# --- short-horizon momentum / reversal / microstructure ---
C["roll_mom_5_10"] = (mom5 - mom10) / (vol10 + 1e-12)
C["vol_ratio_5_20"] = vol5 / (vol20 + 1e-12)
prev_close = closes.shift(1)
C["overnight_gap_20d"] = ((opens - prev_close) / prev_close).rolling(20).mean()
C["intraday_mom_20d"] = ((closes - opens) / opens).rolling(20).mean()
C["atr_ratio_10_60"] = (highs - lows).rolling(10).mean() / ((highs - lows).rolling(60).mean() + 1e-12)
C["rev_10d"] = -mom10 / (vol10 + 1e-12)          # short-term reversal (regime flip candidate)
C["rev_20d"] = -mom20 / (vol20 + 1e-12)          # 1-month reversal
C["mom20_voladj"] = mom20 / (vol20 + 1e-12)      # momentum control

# --- volume/participation ---
C["volume_trend_20_60"] = vol_panel.rolling(20).mean() / (vol_panel.rolling(60).mean() + 1e-12)
C["ret_vol_corr_20d"] = pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a].pct_change()) for a in rets.columns}, index=rets.index)

log(f"{len(C)} candidates built")

log("building library signals for correlation check...")
library = library_signals(panels, closes, rets, vix)
f1 = (mom20 - mom60) / vol20
dn = np.minimum(mkt, 0.0)
f2 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn) / dn.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
f3 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret) / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)
library["vol_adj_mom_accel_20x60"] = f1
library["dn_mkt_beta_60d"] = f2
library["rate_beta_cn10y_60d"] = f3

print("=" * 130)
print(f"PART B: NOVEL CANDIDATE SCREEN (h=10) through {closes.index.max().date()}  |  gate |IC|>=0.0070 & |ICIR|>=0.0840")
print("=" * 130)
hdr = (f"{'candidate':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covD':>5s} {'turn':>5s} "
       f"{'libCorr':>7s} {'d5':>7s} {'d10':>7s} {'d20':>7s} {'IC_24':>8s} {'IC_24A':>8s} {'IC_250':>8s} {'pass':>4s}")
print(hdr)
print("-" * 130)

results = {}
for name, fp in C.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = fast_rank_ic(fp, fwd10, min_valid=8)
    if len(ics) == 0:
        print(f"{name:24s} no IC dates")
        continue
    m = fast_summarize(ics, expected_sign=1)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    # decay at 5/10/20 via fast_rank_ic
    dec = {}
    for h in (5, 10, 20):
        fwdh = forward_returns(closes, h)
        ih = fast_rank_ic(fp, fwdh, min_valid=8)
        dec[h] = round(float(ih.mean()), 4) if len(ih) else float("nan")
    corr, key = max_library_corr(fp, library)
    # recent windows
    wins = {}
    for label, start in [("24", "2024-01-01"), ("24A", "2024-08-01"), ("250", None)]:
        sub = ics[ics.index >= pd.Timestamp(start)] if start else ics.iloc[-250:]
        wins[label] = fast_summarize(sub, expected_sign=1)
    p = abs(m["ic"]) >= ADM_IC and abs(m["icir"]) >= ADM_ICIR
    results[name] = {"metrics": m, "coverage": cov, "turnover": turn, "decay": dec,
                     "corr": corr, "corr_key": key, "pass": p, "windows": wins}
    w24 = wins["24"]["ic"]; w24a = wins["24A"]["ic"]; w250 = wins["250"]["ic"]
    print(f"{name:24s} {m['ic']:8.4f} {m['icir']:7.3f} {m['ic_hit_ratio']:5.2f} {m['n_ic_dates']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:5.2f} {corr:7.3f} "
          f"{dec[5]:7.4f} {dec[10]:7.4f} {dec[20]:7.4f} "
          f"{w24['ic']:8.4f} {w24a['ic']:8.4f} {w250['ic']:8.4f} {'YES' if p else ''}")

print("-" * 130)
print("PASSING (full-history gate):")
for name, r in results.items():
    if r["pass"]:
        print(f"  {name}: IC={r['metrics']['ic']:.4f} ICIR={r['metrics']['icir']:.3f} hit={r['metrics']['ic_hit_ratio']:.2f} "
              f"n={r['metrics']['n_ic_dates']} libCorr={r['corr']:.3f}({r['corr_key']}) "
              f"IC24={r['windows']['24']['ic']:.4f} IC250={r['windows']['250']['ic']:.4f}")
print("=" * 130)
print("RECENT-WINDOW ROBUSTNESS (|IC|>=0.005 full-history candidates):")
for name, r in results.items():
    if abs(r["metrics"]["ic"]) < 0.005:
        continue
    for lbl in ("24", "24A", "250"):
        w = r["windows"][lbl]
        ok = abs(w["ic"]) >= ADM_IC and abs(w["icir"]) >= ADM_ICIR
        print(f"  {name:24s} {lbl:4s} IC={w['ic']:+.4f} ICIR={w['icir']:+.3f} hit={w['ic_hit_ratio']:.2f} n={w['n_ic_dates']:4d} {'PASS' if ok else ''}")

json.dump({k: {"metrics": v["metrics"], "coverage": v["coverage"], "turnover": v["turnover"],
               "decay": v["decay"], "corr": v["corr"], "corr_key": v["corr_key"],
               "pass": v["pass"], "windows": v["windows"]} for k, v in results.items()},
          open("scripts/_miner1_20340904_revalB.json", "w"), indent=1, default=str)
log(f"done -> scripts/_miner1_20340904_revalB.json  ({time.time()-t0:.0f}s total)")
