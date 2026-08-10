"""Miner2 broad screen #2 (cycle 2): novel factor families on the 15-asset cross-section.

Motivation: prior cycle's reversal family passed IC gates but was quarantined for
missing recoverable signal artifacts. Library is now empty. This cycle screens NEW
candidate families (cross-asset beta/correlation with macro observations, gap/overnight
decomposition, multi-horizon reversal composites, short-horizon range extremes) and
re-validates the strongest baseline (rev_1d) as calibration.

Families (per-asset time-series panels, dates x 15 symbols):
  A. Market beta (SPX) 60d, and negative market beta
  B. Dollar beta (DXY) 60d, VIX-change beta 60d, USDJPY-change beta 60d (risk proxies)
  C. Crypto beta: correlation of asset ret with BTC ret 60d
  D. Multi-horizon weighted reversal composites (1-3-5 blend)
  E. Cross-sectional reversal at 5d: -(ret5 - cs median)
  F. Short-horizon range extremes: dist from 5d high (neg), dist from 5d low (pos)
  G. Vol-scaled 5d reversal: -ret5/vol5 ; z-score of 5d return vs vol
  H. Gap-adjusted reversal: -(logret_1d - gap_1d)
  I. Overnight gap + intraday body composite
  J. Amihud 5d / volume-trend reversal
  K. Downside semideviation 20d (neg)
  L. Calibration baselines: rev_1d, nclv_1d, vol20 (expected known results)

Gates (15-name universe): |IC1| >= 0.0070 and |ICIR1| >= 0.0840.
Window: 2021-01-01 .. 2026-07-15 (1y warmup). >=8 names per date.
"""
import sys, time, os
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner1_common import SYMBOLS, load_close, MACRO, IDX_DIR
import miner2_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

# ---- macro observation panels (aligned to idx) ----
macro = {}
for m in MACRO:
    d = pd.read_csv(os.path.join(IDX_DIR, f"{m}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")["close"].reindex(idx)
    macro[m] = pd.to_numeric(d, errors="coerce").astype(float)
DXY_R = macro["DXY"].pct_change()
VIX_C = macro["VIX"].diff()
JPY_R = macro["USDJPY"].pct_change()

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)


def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    print(f"{name:28s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5,
            "ic10": ic10, "passed": passed}


cands = {}

# ---- A. market beta to SPX ----
for nd in (30, 60):
    beta = RET.rolling(nd).cov(RET["SPX"]) / (RET["SPX"].rolling(nd).var() + 1e-12)
    cands[f"beta_spx_{nd}d"] = beta
    cands[f"nbeta_spx_{nd}d"] = -beta
# correlation to SPX
for nd in (30, 60):
    cands[f"corr_spx_{nd}d"] = RET.rolling(nd).corr(RET["SPX"])

# ---- B. macro betas ----
cands["beta_dxy_60d"] = RET.rolling(60).cov(DXY_R) / (DXY_R.rolling(60).var() + 1e-12)
cands["nbeta_dxy_60d"] = -cands["beta_dxy_60d"]
cands["beta_vix_60d"] = RET.rolling(60).cov(VIX_C) / (VIX_C.rolling(60).var() + 1e-12)
cands["nbeta_vix_60d"] = -cands["beta_vix_60d"]
cands["beta_jpy_60d"] = RET.rolling(60).cov(JPY_R) / (JPY_R.rolling(60).var() + 1e-12)
cands["nbeta_jpy_60d"] = -cands["beta_jpy_60d"]

# ---- C. crypto beta (corr with BTC) ----
cands["corr_btc_60d"] = RET.rolling(60).corr(RET["BTC"])
cands["ncorr_btc_60d"] = -RET.rolling(60).corr(RET["BTC"])

# ---- D. multi-horizon reversal composites ----
r1 = -LOG
r3 = -np.log(CP / CP.shift(3))
r5 = -np.log(CP / CP.shift(5))
cands["rev_1_3_5"] = r1 + 0.5 * r3 + 0.25 * r5
cands["rev_1_5"] = r1 + 0.5 * r5
cands["rev_1d_lead"] = -LOG.shift(-1)  # mis-specified: uses future; diagnostic only

# ---- E. cross-sectional 5d reversal ----
m5 = CP / CP.shift(5) - 1.0
cands["crev_5d"] = -(m5.sub(m5.median(axis=1), axis=0))
m10 = CP / CP.shift(10) - 1.0
cands["crev_10d"] = -(m10.sub(m10.median(axis=1), axis=0))

# ---- F. short-horizon range extremes ----
cands["ndist_5d_high"] = -(CP / CP.rolling(5).max() - 1.0)
cands["dist_5d_low"] = CP / CP.rolling(5).min() - 1.0
cands["ndist_10d_high"] = -(CP / CP.rolling(10).max() - 1.0)
cands["dist_10d_low"] = CP / CP.rolling(10).min() - 1.0

# ---- G. vol-scaled 5d reversal ----
vol5 = RET.rolling(5).std()
vol20 = RET.rolling(20).std()
cands["rev5_vol5"] = -m5 / (vol5 + 1e-12)
cands["rev5_vol20"] = -m5 / (vol20 + 1e-12)
# z-score of 5d return vs own 60d vol regime
cands["rev5_z60"] = -m5 / (vol5.rolling(60).mean() + 1e-12)

# ---- H. gap-adjusted reversal ----
gap1 = OP / CP.shift(1) - 1.0
cands["gap_adj_rev"] = -(LOG - np.log1p(gap1))  # intraday-only reversal
cands["gap_rev_2d"] = -(OP / CP.shift(2) - 1.0)

# ---- I. overnight + intraday composite ----
intra = CP / OP - 1.0
cands["overnight_rev"] = -gap1
cands["oi_comp"] = -(0.5 * gap1 + 0.5 * intra)

# ---- J. Amihud / volume ----
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
illiq5 = (RET.abs() / (VOL + 1e-9)).rolling(5).mean()
cands["n_illiq_5d"] = -illiq5
cands["vol_trend_5_20"] = VOL.rolling(5).mean() / (VOL.rolling(20).mean() + 1e-9)
cands["rev1_x_voltrend"] = r1 * cands["vol_trend_5_20"]

# ---- K. downside semideviation ----
down = RET.clip(upper=0)
cands["ndownside_vol_20"] = -down.rolling(20).std()
cands["downside_ratio_20"] = -down.rolling(20).std() / (vol20 + 1e-12)

# ---- L. calibration baselines ----
cands["rev_1d"] = r1
rng1 = (HP - LP).replace(0, np.nan)
cands["nclv_1d"] = -((CP - LP) / rng1)
cands["vol_20d"] = vol20 * np.sqrt(252)

res = {}
for name, panel in cands.items():
    try:
        res[name] = run(name, panel)
    except Exception as e:
        print(f"{name}: ERROR {e}")

npass = sum(1 for r in res.values() if r["passed"])
print(f"\nscreen done in {time.time()-t0:.1f}s | {npass} passed gate / {len(res)} candidates")
print("passers:", [n for n, r in res.items() if r["passed"]])
