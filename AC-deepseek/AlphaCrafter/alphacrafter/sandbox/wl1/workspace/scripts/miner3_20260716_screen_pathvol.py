"""Miner3 cycle-2 novel screen: price-path shape + volume dynamics families.

Idea 1 (path shape): Kaufman Efficiency Ratio and Variance Ratio capture whether a
price path is trending (high ER, VR>1) or mean-reverting/choppy (low ER, VR<1),
independent of raw return sign. Cross-sectional rank of path shape has not been
explored in the previous cycles (reversal/CLV/vol/beta families dominated).

Idea 2 (volume dynamics): within-asset relative volume (V / MA(V)) and Amihud
illiquidity z-score are comparable across asset classes because they are
dimensionless within-asset ratios. Volume is present for 9/15 symbols.

Gates: |IC1|>=0.0070 and |ICIR1|>=0.0840 on 15-instrument universe.
Window: 2020-01-01..2026-07-15 warm-up; IC evaluated on 2021-01-01+ (1y burn-in).
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
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VP = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)


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
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


cands = {}

# ---------- A. Kaufman Efficiency Ratio (path directionality) ----------
def kaufman_er(nd):
    path = CP.diff().abs().rolling(nd).sum()
    net = (CP - CP.shift(nd)).abs()
    return net / (path + 1e-12)

for nd in (5, 10, 20, 40, 60):
    er = kaufman_er(nd)
    cands[f"er_{nd}d"] = er
    cands[f"neg_er_{nd}d"] = -er

# ---------- B. Variance Ratio (trend vs mean-reversion tendency) ----------
for nd in (5, 10, 20):
    rnd = CP.pct_change(nd)              # n-day simple return
    vr = rnd.rolling(nd).var() / (nd * RET.rolling(nd).var() + 1e-12)
    cands[f"vr_{nd}d"] = vr              # >1 trending, <1 mean-reverting
    cands[f"inv_vr_{nd}d"] = -vr         # high = mean-reverting

# ---------- C. RSI / oscillator ----------
def rsi(nd=14):
    up = RET.clip(lower=0).rolling(nd).mean()
    dn = (-RET).clip(lower=0).rolling(nd).mean()
    rs = up / (dn + 1e-12)
    return rs / (1 + rs) - 0.5           # centered RSI

cands["rsi_14d_c"] = rsi(14)
cands["rsi_7d_c"] = rsi(7)
# overbought/oversold reversal: high RSI -> fade
cands["neg_rsi_14d_c"] = -rsi(14)

# ---------- D. Bollinger / moving-average position ----------
ma20 = CP.rolling(20).mean(); sd20 = RET.rolling(20).std()
cands["bb_pos_20d"] = (CP - ma20) / (2 * sd20 + 1e-12)
cands["neg_bb_pos_20d"] = -(CP - ma20) / (2 * sd20 + 1e-12)
for fst, slw in ((10, 30), (20, 60), (50, 200)):
    maf = CP.rolling(fst).mean(); mas = CP.rolling(slw).mean()
    cands[f"ma_cross_{fst}_{slw}"] = (maf - mas) / (sd20 * np.sqrt(slw) + 1e-12)

# ---------- E. Volume dynamics (9/15 symbols) ----------
vol_ma20 = VP.rolling(20).mean()
rel_vol = VP / (vol_ma20 + 1e-12)                # within-asset volume ratio
for nd in (5, 10, 20):
    cands[f"rel_vol_{nd}d"] = VP.rolling(nd).mean() / (VP.rolling(60).mean() + 1e-12)
    cands[f"vol_trend_{nd}_60"] = VP.rolling(nd).mean() / (vol_ma20 + 1e-12)
cands["rel_vol_5_20"] = VP.rolling(5).mean() / (vol_ma20 + 1e-12)
# Amihud illiquidity (within-asset z-score over trailing 60d)
amihud = RET.abs() / (VP + 1e-12)
for nd in (10, 20):
    am_z = (amihud.rolling(nd).mean() - amihud.rolling(60).mean()) / (amihud.rolling(60).std() + 1e-12)
    cands[f"amihud_z_{nd}d"] = am_z
# volume-price divergence: rolling corr(ret, dlog vol)
dvol = np.log(VP + 1e-12).diff()
for nd in (10, 20):
    vp_corr = RET.rolling(nd).corr(dvol)
    cands[f"vp_corr_{nd}d"] = vp_corr
    cands[f"neg_vp_corr_{nd}d"] = -vp_corr

res = [run(n, p) for n, p in cands.items()]
print(f"\nscreen done {time.time()-t0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")
print("\n=== PASSED (|IC1|>=0.007, |ICIR1|>=0.084) ===")
for r in res:
    if r["passed"]:
        print(f"  {r['name']:22s} IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} "
              f"hit={r['ic1']['hit']:.3f} cov={r['cov']:.3f} to={r['to']:.3f} "
              f"IC5={r['ic5']['ic']:+.4f} IC10={r['ic10']['ic']:+.4f}")
print("\n=== top 15 by |IC1| ===")
for r in sorted(res, key=lambda r: -abs(r["ic1"]["ic"]))[:15]:
    mark = "PASS" if r["passed"] else "   "
    print(f"[{mark}] {r['name']:22s} IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} hit={r['ic1']['hit']:.3f} cov={r['cov']:.3f}")
