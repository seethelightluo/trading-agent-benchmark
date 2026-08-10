"""Miner3 novel factor family screen (2026-07-16).

Families NOT previously covered by miner1/2/3 screens:
  A. Volume / liquidity      (volume present for 9 of 15 names)
  B. Streak / consistency    (consecutive up/down days, win rate, tail days)
  C. Volatility dynamics     (vol-of-vol, |ret| autocorr, Garman-Klass, vol momentum)
  D. Trend efficiency & candle structure (Kaufman ER, ADX-like, ATR%, shadows, gaps)

Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 (daily rank IC, 15-name universe,
min 8 valid names per date). Validation window 2021-01-01..2026-07-15.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: pd.to_numeric(closes[s]["volume"], errors="coerce").reindex(idx).astype(float) for s in SYMBOLS})
VO = VO.replace(0.0, np.nan)  # zero-volume names become NaN
RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)
EPS = 1e-12


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to,
            "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


def streak_panel(flag):
    """Consecutive-True run length per column (0 where flag==0)."""
    out = pd.DataFrame(index=flag.index, columns=flag.columns, dtype=float)
    for s in flag.columns:
        ser = flag[s].astype(float)
        out[s] = ser.groupby((ser == 0).cumsum()).cumsum()
    return out


cands = {}

# ---------- A. volume / liquidity ----------
vol_mean20 = VO.rolling(20).mean()
vol_std20 = VO.rolling(20).std()
cands["vol_z_20d"] = (VO - vol_mean20) / (vol_std20 + EPS)
cands["vol_trend_5_20"] = VO.rolling(5).mean() / (vol_mean20 + EPS) - 1.0
cands["amihud_20d"] = -(RET.abs() / (VO + EPS)).rolling(20).mean()
cands["pv_corr_20d"] = RET.rolling(20).corr(VO.pct_change())
cands["vol_spike_1d"] = VO / (vol_mean20 + EPS)
cands["vol_cv_20d"] = vol_std20 / (vol_mean20 + EPS)

# ---------- B. streak / consistency ----------
up = (RET > 0).astype(float)
down = (RET < 0).astype(float)
cands["streak_up"] = streak_panel(up).clip(upper=10)
cands["streak_dn"] = -streak_panel(down).clip(upper=10)
cands["winrate_20d"] = up.rolling(20).mean()
cands["winrate_60d"] = up.rolling(60).mean()
cands["worst_day_20d"] = RET.rolling(20).min()
cands["best_day_60d"] = RET.rolling(60).max()
cands["down_days_60d"] = -down.rolling(60).sum()

# ---------- C. volatility dynamics ----------
rv10 = RET.rolling(10).std() * np.sqrt(252)
rv20 = RET.rolling(20).std() * np.sqrt(252)
rv60 = RET.rolling(60).std() * np.sqrt(252)
cands["vol_of_vol_60d"] = rv10.rolling(60).std() / (rv10.rolling(60).mean() + EPS)
cands["vol_autocorr_20d"] = RET.abs().rolling(20).corr(RET.abs().shift(1))
cands["vol_mom_20d"] = rv10 / (rv10.shift(20) + EPS) - 1.0
cands["hv_ratio_60_20"] = rv60 / (rv20 + EPS) - 1.0
# Garman-Klass
hl = np.log(HP / LP)
co = np.log(CP / OP)
cands["gk_vol_20d"] = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(20).mean() * 252)

# ---------- D. trend efficiency / candle structure ----------
cands["kaufman_er_60d"] = (CP - CP.shift(60)).abs() / (LRET.abs().rolling(60).sum() + EPS)
tr = pd.concat([HP - LP, (HP - CP.shift(1)).abs(), (LP - CP.shift(1)).abs()], axis=1)
atr = tr.max(axis=1).rolling(20).mean()
ma20 = CP.rolling(20).mean()
ma60 = CP.rolling(60).mean()
cands["adx_like_20d"] = (ma20 - ma60).abs() / (atr + EPS)
cands["atr_pct_20d"] = atr / (CP + EPS)
rng = (HP - LP) + EPS
up_sh = (HP - np.maximum(OP, CP)) / rng
dn_sh = (np.minimum(OP, CP) - LP) / rng
cands["upper_shadow_20d"] = up_sh.rolling(20).mean()
cands["lower_shadow_20d"] = dn_sh.rolling(20).mean()
cands["gap_size_5d"] = (OP / CP.shift(1) - 1.0).abs().rolling(5).mean()

res = []
for n, p in cands.items():
    res.append(run(n, p))
print(f"\nscreen {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)}/{len(res)} passed gate")

passers = [r for r in res if r["passed"]]
print("\n=== PASSERS ===")
for r in passers:
    print(r["name"], f"IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} cov={r['cov']:.3f} to={r['to']:.3f}")
