"""Miner2 broad screen #1: novel factor families on the 15-asset cross-section.

Families (all per-asset time-series panels, dates x 15 symbols):
  A. Lottery/extreme return effects: max/min daily return over n days (MAX effect)
  B. Trend efficiency (Kaufman ER): |P_t - P_{t-n}| / sum of |dP|
  C. Trend t-stat (TSMOM): slope of OLS fit over n days scaled by residual vol
  D. Trend acceleration: mom(60) - mom(20), mom(120) - mom(60)
  E. Relative (cross-sectional) momentum: asset ret - cross-sectional median ret
  F. Breadth: fraction of up days over n
  G. Mean-reversion oscillators: Bollinger %B(20,2), RSI(14)
  H. Vol-of-vol: std of rolling-20d vol over 120d
  I. AR(1) autocorrelation of daily returns over n
  J. Time since n-day high / low (in days)
  K. Z-score of close vs n-day MA
  L. Distance to 20d low as fraction of 20d range
  M. Baselines from prior cycles: rev_1d/2d/5d, nclv_1d, vol_20d, inv_vol_20d,
     mom_20d/60d, sharpe_60d, max_dd_20d, skew_60d (calibration)

Gates (15-name universe): |IC1| >= 0.0070 and |ICIR1| >= 0.0840.
Window: 2020-01 .. 2026-07-15 (warm-up). >=8 names per date for cross-section.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner1_common import SYMBOLS, load_close
import miner2_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]  # 1y warmup for rolling windows
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

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
    print(f"{name:26s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


cands = {}

# ---- A. lottery / extreme returns ----
for nd in (5, 20, 60):
    cands[f"max_ret_{nd}d"] = RET.rolling(nd).max()
    cands[f"min_ret_{nd}d"] = RET.rolling(nd).min()
    cands[f"nmax_ret_{nd}d"] = -RET.rolling(nd).max()

# ---- B. Kaufman efficiency ratio ----
for nd in (10, 20, 60):
    num = (CP - CP.shift(nd)).abs()
    den = LOG.abs().rolling(nd).sum()
    cands[f"kaufman_er_{nd}d"] = num / (den + 1e-12)

# ---- C. TSMOM trend t-stat (slope / residual std via polyfit) ----
def ts_slope_tstat(nd):
    cols = {}
    x = np.arange(nd, dtype=float)
    xm = x - x.mean()
    den = np.sqrt((xm ** 2).sum())
    for s in SYMBOLS:
        y = np.log(CP[s].values)
        df = pd.DataFrame({"y": y})
        out = np.full(len(df), np.nan)
        for i in range(nd - 1, len(df)):
            seg = y[i - nd + 1:i + 1]
            if not np.all(np.isfinite(seg)):
                continue
            slope = (xm * (seg - seg.mean())).sum() / (xm ** 2).sum()
            resid = seg - (seg.mean() + slope * xm)
            sse = np.sqrt((resid ** 2).mean())
            out[i] = slope / (sse + 1e-12) * np.sqrt(nd)
        cols[s] = out
    return pd.DataFrame(cols, index=CP.index)

cands["tsmom_t_60d"] = ts_slope_tstat(60)
cands["tsmom_t_120d"] = ts_slope_tstat(120)

# ---- D. trend acceleration ----
cands["accel_60_20"] = (CP / CP.shift(60) - 1.0) - (CP / CP.shift(20) - 1.0)
cands["accel_120_60"] = (CP / CP.shift(120) - 1.0) - (CP / CP.shift(60) - 1.0)
cands["accel_20_5"] = (CP / CP.shift(20) - 1.0) - (CP / CP.shift(5) - 1.0)

# ---- E. relative (cross-sectional) momentum ----
for nd in (10, 20, 60):
    m = CP / CP.shift(nd) - 1.0
    med = m.median(axis=1)
    cands[f"rel_mom_{nd}d"] = m.sub(med, axis=0)

# ---- F. breadth: up-day fraction ----
for nd in (20, 60):
    cands[f"up_frac_{nd}d"] = (RET > 0).rolling(nd).mean()

# ---- G. oscillators ----
ma20 = CP.rolling(20).mean()
sd20 = RET.rolling(20).std()
cands["boll_pctb_20"] = (CP - ma20) / (2.0 * sd20 * np.sqrt(1) + 1e-12)  # std of price ~ sd of ret
# RSI(14) Wilder-lite: average gain / average loss
def rsi(nd=14):
    up = RET.clip(lower=0).rolling(nd).mean()
    dn = (-RET.clip(upper=0)).rolling(nd).mean()
    return 100.0 - 100.0 / (1.0 + up / (dn + 1e-12))
cands["rsi_14"] = rsi(14)
cands["rsi_7"] = rsi(7)

# ---- H. vol-of-vol ----
vol20 = RET.rolling(20).std()
cands["vol_of_vol_20_120"] = vol20.rolling(120).std() / (vol20.rolling(120).mean() + 1e-12)
cands["vol_of_vol_20_60"] = vol20.rolling(60).std() / (vol20.rolling(60).mean() + 1e-12)

# ---- I. AR(1) autocorrelation ----
def ar1(nd):
    cols = {}
    for s in SYMBOLS:
        r = RET[s].values
        out = np.full(len(r), np.nan)
        for i in range(nd, len(r)):
            seg = r[i - nd:i]
            if not np.all(np.isfinite(seg)):
                continue
            a, b = seg[:-1], seg[1:]
            if a.std() == 0 or b.std() == 0:
                continue
            out[i] = np.corrcoef(a, b)[0, 1]
        cols[s] = out
    return pd.DataFrame(cols, index=CP.index)

cands["ar1_20d"] = ar1(20)
cands["ar1_60d"] = ar1(60)

# ---- J. days since n-day high / low ----
def days_since(rollmax_or_min, nd):
    cols = {}
    for s in SYMBOLS:
        c = CP[s].values
        out = np.full(len(c), np.nan)
        if rollmax_or_min == "high":
            rw = pd.Series(c).rolling(nd).max().values
            cond = c == rw
        else:
            rw = pd.Series(c).rolling(nd).min().values
            cond = c == rw
        last = -1
        for i in range(len(c)):
            if np.isnan(rw[i]):
                continue
            if cond[i]:
                last = i
            out[i] = i - last if last >= 0 else np.nan
        cols[s] = out
    return pd.DataFrame(cols, index=CP.index)

cands["days_since_high_60"] = days_since("high", 60)
cands["days_since_low_60"] = days_since("low", 60)

# ---- K. z-score vs n-day MA ----
for nd in (20, 60):
    m = CP.rolling(nd).mean()
    s = CP.rolling(nd).std()
    cands[f"z_close_ma{nd}"] = (CP - m) / (s + 1e-12)

# ---- L. distance to 20d low as fraction of 20d range ----
rng20 = CP.rolling(20).max() - CP.rolling(20).min()
cands["pos_in_20d_range"] = (CP - CP.rolling(20).min()) / (rng20 + 1e-12)
cands["npos_in_20d_range"] = -(CP - CP.rolling(20).min()) / (rng20 + 1e-12)

# ---- M. baselines / calibration ----
for nd in (1, 2, 5):
    cands[f"rev_{nd}d"] = -(CP / CP.shift(nd) - 1.0)
cands["nclv_1d"] = -((CP - LP) / (HP - LP).replace(0, np.nan))
for nd in (20, 60):
    v = RET.rolling(nd).std() * np.sqrt(252)
    cands[f"vol_{nd}d"] = v
    cands[f"inv_vol_{nd}d"] = -v
cands["mom_20d"] = CP / CP.shift(20) - 1.0
cands["mom_60d"] = CP / CP.shift(60) - 1.0
cands["sharpe_60d"] = RET.rolling(60).mean() * 252 / (RET.rolling(60).std() * np.sqrt(252) + 1e-12)
cands["max_dd_20d"] = CP.rolling(20).max() / CP - 1.0
cands["skew_60d"] = RET.rolling(60).skew()

res = []
for name, panel in cands.items():
    try:
        res.append(run(name, panel))
    except Exception as e:
        print(f"{name}: ERROR {e}")

npass = sum(r["passed"] for r in res)
print(f"\nscreen done in {time.time()-t0:.1f}s | {npass} passed gate / {len(res)} candidates")
