"""miner_3 - focused screen for novel factor families (2026-07-16).
Families: drawdown/52w-high, skewness, cross-asset spillover (BTC/SPX/XAU beta
and corr), illiquidity (Amihud), intraday range vol, and conditional reversal
(volume/volatility-conditioned short-term reversal).
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 on 1d fwd rank IC.
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
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
vol20 = RET.rolling(20).std() * np.sqrt(252)
vol60 = RET.rolling(60).std() * np.sqrt(252)
N_CELLS = len(idx) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840
fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20)}


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | {dec} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic": ics, "passed": passed}


cands = {}
# ---------- drawdown / 52w high ----------
cands["dd_20d"] = CP / CP.rolling(20).max() - 1.0
cands["dd_60d"] = CP / CP.rolling(60).max() - 1.0
cands["dd_252d"] = CP / CP.rolling(252).max() - 1.0
cands["w52_high_prox"] = 1.0 - CP / CP.rolling(252).max()          # 0 near high, 1 far below
cands["neg_dd_60_vol"] = -(CP / CP.rolling(60).max() - 1.0) / vol60
# ---------- return distribution shape ----------
def rskew(x, nd):
    mu = x.rolling(nd).mean()
    sd = x.rolling(nd).std()
    return ((x - mu) ** 3).rolling(nd).mean() / (sd ** 3 + 1e-12)
cands["skew_20d"] = rskew(RET, 20)
cands["skew_60d"] = rskew(RET, 60)
cands["max_ret_20d"] = RET.rolling(20).max()
cands["min_ret_20d"] = RET.rolling(20).min()
# ---------- intraday range volatility ----------
rng = (HP - LP) / CP
cands["inv_range_vol_20"] = -rng.rolling(20).mean()
cands["range_vol_chg_5_60"] = rng.rolling(5).mean() / rng.rolling(60).mean()
# ---------- illiquidity / volume quality ----------
cands["amihud_20d"] = -(RET.abs() / (VO + 1e-9)).rolling(20).mean()   # negated: liquid -> high
cands["dollar_vol_chg_20"] = (CP * VO).pct_change(20)
cands["vol_price_corr_60"] = RET.rolling(60).corr(VO.pct_change())
# ---------- cross-asset spillover ----------
def roll_corr(y_panel, x, win):
    out = {}
    for s in y_panel.columns:
        df = pd.concat([y_panel[s], x], axis=1).dropna()
        out[s] = df.iloc[:, 0].rolling(win).corr(df.iloc[:, 1]).reindex(idx)
    return pd.DataFrame(out)

btc_r = RET["BTC"]; spx_r = RET["SPX"]; xau_r = RET["XAU"]; hsi_r = RET["HSI"]
cands["btc_corr_20"] = roll_corr(RET, btc_r, 20)
cands["btc_beta_60"] = roll_corr(RET, btc_r, 60).mul((btc_r.rolling(60).std().to_numpy().reshape(-1,1) / (RET.rolling(60).std().to_numpy() + 1e-12)), axis=0)
cands["spx_corr_20"] = roll_corr(RET, spx_r, 20)
cands["spx_beta_60"] = roll_corr(RET, spx_r, 60).mul((spx_r.rolling(60).std().to_numpy().reshape(-1,1) / (RET.rolling(60).std().to_numpy() + 1e-12)), axis=0)
cands["xau_corr_20"] = roll_corr(RET, xau_r, 20)
cands["hsi_corr_20"] = roll_corr(RET, hsi_r, 20)
cands["neg_btc_beta_60"] = -cands["btc_beta_60"]
# ---------- conditional reversal ----------
vma60 = VO.rolling(60).mean()
cands["rev1d_hi_vol"] = -(CP / CP.shift(1) - 1.0) * (VO / vma60)
cands["rev1d_hi_volz"] = -(CP / CP.shift(1) - 1.0) * ((VO - vma60) / (VO.rolling(60).std() + 1e-9))
cands["rev1d_vol_cond"] = -(CP / CP.shift(1) - 1.0) * (vol20 / (vol20.rolling(120).mean() + 1e-12))
cands["clv1d_vol_cond"] = ((CP - LP) / (HP - LP + 1e-12)) * (vol20 / (vol20.rolling(120).mean() + 1e-12))

res = [run(n, p) for n, p in cands.items()]
print(f"\nscreen2 done {time.time()-t0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")
print("\n=== sorted by |IC1| ===")
for r in sorted(res, key=lambda r: -abs(r["ic"][1]["ic"])):
    mark = "PASS" if r["passed"] else "   "
    print(f"[{mark}] {r['name']:24s} IC1={r['ic'][1]['ic']:+.4f} ICIR1={r['ic'][1]['icir']:+.3f} hit={r['ic'][1]['hit']:.3f}")
