"""miner_3 screen v4 (2026-07-30 cycle): fresh factor families.

Goal: find NEW factors passing |IC|>=0.007 and |ICIR|>=0.084 at horizon 10 on the
15-instrument cross-asset universe, with low correlation to the 4 former library
signals (mom_10d_skip5, mom_120d_skip5, vol_of_vol20x60, vix_beta_cond_60x20).

Families NOT heavily covered in earlier screens:
  - candle structure (close location value, body/shadow ratios)
  - oscillator (RSI, fast stochastic)
  - return autocorrelation (trend persistence)
  - drawdown depth / recovery distance
  - volume flow imbalance (up-day vs down-day volume)
  - overnight vs intraday return decomposition
  - cross-asset bond-linkage betas (US10Y / CN10Y / spread)
  - cross-sectionally demeaned relative momentum
  - momentum deceleration / acceleration
  - upside/downside capture asymmetry

Data: API panel truncated at visible_through = 2026-07-29 (no future data).
"""
import sys, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_validation_lib import (TRADABLE, MIN_INSTR, load_panel, load_macro,
                                   align_fwd_returns, rank_ic_series, ic_analysis,
                                   library_corr)

VISIBLE = "2026-07-29"
panel = load_panel(max_date=VISIBLE)
ret = panel.pct_change()
print(f"panel: {panel.shape} assets={panel.shape[1]} dates={len(panel)} "
      f"through {panel.index.max().date()}", flush=True)

# OHLC panel from CSV for candle factors (truncated at VISIBLE)
import json
from pathlib import Path
STOCK_DIR = Path("../persistent/stock_data")

def load_ohlc(sym):
    df = pd.read_csv(STOCK_DIR / f"{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date").sort_index()
    return df

ohlc = {s: load_ohlc(s) for s in TRADABLE}
hi = pd.DataFrame({s: ohlc[s]["high"] for s in TRADABLE}).reindex(panel.index)
lo = pd.DataFrame({s: ohlc[s]["low"] for s in TRADABLE}).reindex(panel.index)
op = pd.DataFrame({s: ohlc[s]["open"] for s in TRADABLE}).reindex(panel.index)
cl = panel
vol_p = pd.DataFrame({s: (ohlc[s]["volume"] if "volume" in ohlc[s] else pd.Series(np.nan, index=ohlc[s].index))
                      for s in TRADABLE}).reindex(panel.index)

rng = (hi - lo).replace(0, np.nan)
clv = (cl - lo) / rng
body = (cl - op) / rng
shadow_up = (hi - np.maximum(op, cl)) / rng
shadow_dn = (np.minimum(op, cl) - lo) / rng

def roll_std(x, w, mp=None):
    return x.rolling(w, min_periods=mp or max(10, w // 2)).std()

def roll_mean(x, w, mp=None):
    return x.rolling(w, min_periods=mp or max(10, w // 2)).mean()

def beta_of(a, m, w):
    return a.rolling(w, min_periods=max(20, w // 2)).cov(m) / \
        m.rolling(w, min_periods=max(20, w // 2)).var()

# ---------------- macro series ----------------
vix = load_macro("VIX", max_date=VISIBLE)
dxy = load_macro("DXY", max_date=VISIBLE)
us10y_r = cl["US10Y"].pct_change()
cn10y_r = cl["CN10Y"].pct_change()
spread = cl["US10Y"] - cl["CN10Y"]
spread_r = spread.pct_change()

C = {}
# --- candle structure ---
C["clv_20d"] = clv.rolling(20, min_periods=10).mean()
C["body_ratio_20d"] = body.rolling(20, min_periods=10).mean()
C["shadow_up_20d"] = shadow_up.rolling(20, min_periods=10).mean()
C["shadow_dn_20d"] = shadow_dn.rolling(20, min_periods=10).mean()
C["shadow_asym_20d"] = (shadow_up.rolling(20, min_periods=10).mean() /
                        shadow_dn.rolling(20, min_periods=10).mean().replace(0, np.nan))
C["clv_change_10x60"] = clv.rolling(10, min_periods=5).mean() - clv.rolling(60, min_periods=30).mean()
# --- oscillators ---
def rsi(n):
    up = ret.clip(lower=0).rolling(n).mean()
    dn = (-ret.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
C["rsi_14"] = rsi(14)
C["rsi_21"] = rsi(21)
lo_n = lo.rolling(14, min_periods=7).min()
hi_n = hi.rolling(14, min_periods=7).max()
C["stoch_k_14"] = ((cl - lo_n) / (hi_n - lo_n).replace(0, np.nan)) * 100
# --- autocorrelation / persistence ---
def autocorr_lag1(w):
    mu = ret.rolling(w, min_periods=max(10, w // 2)).mean()
    num = ((ret - mu) * (ret.shift(1) - mu.shift(1))).rolling(w, min_periods=max(10, w // 2)).sum()
    den = (ret ** 2).rolling(w, min_periods=max(10, w // 2)).sum()
    return num / den.replace(0, np.nan)
C["autocorr_1_20d"] = autocorr_lag1(20)
C["autocorr_1_60d"] = autocorr_lag1(60)
# --- drawdown / recovery ---
C["dd_120d"] = cl / cl.rolling(120, min_periods=60).max() - 1.0
C["recovery_120d"] = cl / cl.rolling(120, min_periods=60).min() - 1.0
C["dd_ratio_20x120"] = (cl / cl.rolling(20, min_periods=10).max() - 1.0) - \
                       (cl / cl.rolling(120, min_periods=60).max() - 1.0)
# --- volume flow imbalance ---
upday = (ret > 0).astype(float)
up_vol = (vol_p * upday).rolling(20, min_periods=10).sum()
dn_vol = (vol_p * (1 - upday)).rolling(20, min_periods=10).sum()
C["vol_imbalance_20d"] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)
C["vol_imbalance_60d"] = ((vol_p * upday).rolling(60, min_periods=30).sum() -
                          (vol_p * (1 - upday)).rolling(60, min_periods=30).sum()) / \
    (vol_p.rolling(60, min_periods=30).sum().replace(0, np.nan))
# --- overnight vs intraday ---
gap = op / cl.shift(1) - 1.0
intra = cl / op - 1.0
C["overnight_ret_20d"] = gap.rolling(20, min_periods=10).mean()
C["gap_trend_20d"] = gap.rolling(60, min_periods=30).mean()
C["overnight_share_20d"] = (gap.rolling(20, min_periods=10).mean() /
                            (gap + intra).rolling(20, min_periods=10).mean().abs().replace(0, np.nan))
# --- bond linkage ---
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)
C["beta_cn10y_60d"] = beta_of(ret, cn10y_r, 60)
C["corr_us10y_60d"] = ret.rolling(60, min_periods=30).corr(us10y_r)
C["beta_spread_60d"] = beta_of(ret, spread_r, 60)
# --- relative (cross-sectionally demeaned) momentum ---
mom20 = cl.pct_change(20)
mom60 = cl.pct_change(60)
C["rel_mom_20d"] = mom20 - mom20.mean(axis=1)
C["rel_mom_60d"] = mom60 - mom60.mean(axis=1)
# --- momentum deceleration ---
C["mom_accel_20x60"] = (cl.pct_change(20) - cl.pct_change(60)).rank(axis=1, pct=True) * 0 + \
                       (cl.pct_change(20) - cl.pct_change(60))
# --- upside/downside capture ---
maxr = ret.clip(lower=0).rolling(20, min_periods=10).sum()
minr = (-ret.clip(upper=0)).rolling(20, min_periods=10).sum()
C["upside_capture_20d"] = maxr / minr.replace(0, np.nan)
# --- risk-adjusted reversal ---
vol5 = roll_std(ret, 5)
C["zrev_5d_20"] = -(cl / cl.shift(5) - 1.0) / vol5
# --- vol regime z-score ---
vol20 = roll_std(ret, 20)
C["vol_z_20x120"] = (vol20 - vol20.rolling(120, min_periods=60).mean()) / \
    vol20.rolling(120, min_periods=60).std().replace(0, np.nan)
# --- market synchronicity ---
ewr = ret.mean(axis=1)
C["corr_ew_20d"] = ret.rolling(20, min_periods=10).corr(ewr)

# ---------------- library signals (former effective) for correlation audit ----------------
lib = {}
lib["mom_10d_skip5"] = cl.shift(5) / cl.shift(15) - 1.0
lib["mom_120d_skip5"] = cl.shift(5) / cl.shift(125) - 1.0
lib["vol_of_vol20x60"] = roll_std(roll_std(ret, 20), 60)
vixr = vix.pct_change()
lib["vix_beta_cond_60x20"] = -beta_of(ret, vixr, 60) * (vix / vix.shift(20) - 1.0)

print(f"{'factor':<26}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}{'cov':>6}{'turn':>7}{'librho':>8}  gate")
results = {}
for name, f in C.items():
    f = f.reindex(panel.index)
    r = ic_analysis(f, panel, horizon=10, label=name)
    results[name] = r
    lc = library_corr(f, lib)
    ok = abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084
    print(f"{name:<26}{r['ic']:>8.4f}{r['icir']:>8.4f}{r['ic_hit_ratio']:>7.3f}"
          f"{r['n_ic_dates']:>6d}{r['coverage_asset_days']:>6.2f}{r['turnover_10d_rank']:>7.2f}"
          f"{lc:>8.3f}  {'PASS' if ok else ''}", flush=True)

print("\n=== PASSED (gate) sorted by |ic|*|icir| ===")
passed = {n: r for n, r in results.items()
          if abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084 and r["n_ic_dates"] >= 200}
for n, r in sorted(passed.items(), key=lambda kv: -abs(kv[1]["ic"]) * abs(kv[1]["icir"])):
    lc = library_corr(C[n], lib)
    print(f"  {n:<26} ic={r['ic']:+.4f} icir={r['icir']:.4f} |ic*icir|={abs(r['ic']*r['icir']):.5f} "
          f"hit={r['ic_hit_ratio']:.3f} n={r['n_ic_dates']} cov={r['coverage_asset_days']:.2f} "
          f"turn={r['turnover_10d_rank']:.2f} librho={lc:.3f}")

import pickle
with open("scripts/miner_3_20260730_screen4.pkl", "wb") as fh:
    pickle.dump({n: C[n] for n in passed}, fh)
with open("scripts/miner_3_20260730_screen4_results.json", "w") as fh:
    json.dump({n: {k: v for k, v in r.items() if k != "decay_ic_by_horizon"} for n, r in results.items()},
              fh, indent=1, default=str)
print("\nsaved screen4 results + passing signals")
