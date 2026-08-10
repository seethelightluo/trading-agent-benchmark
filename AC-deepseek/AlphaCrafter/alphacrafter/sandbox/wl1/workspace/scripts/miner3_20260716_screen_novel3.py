"""Miner3 novel factor screen #3 (2026-07-16).

Families NOT already covered by prior miner1/2/3 screens (momentum, reversal,
vol level, volume, OHLC location/clv, macro-beta, efficiency-ratio, streaks,
vol-of-vol, Garman-Klass, overnight/intraday split):

  A. Return-distribution shape : rolling skew, kurtosis, downside/upside
     semideviation ratio, Sortino-like quality ratio
  B. Drawdown state           : rolling max-drawdown depth (negated),
     distance-from-high, time-under-water
  C. Cross-asset beta         : rolling beta to US10Y changes, XAU, BTC,
     COPPER; rate-beta, safe-haven-beta, risk-appetite-beta
  D. Mean-reversion speed     : AR(1) of daily returns, |ret| autocorr
  E. Range-mid momentum       : momentum of (H+L)/2 and (H+L+2C)/4
  F. Gap dynamics             : 5d cumulative gap / ATR, gap-reversal
  G. Vol asymmetry            : up-day vol / down-day vol ratio

Gate (daily paper rank IC, 15-name universe, >=8 names/date):
  |IC1| >= 0.0070 and |ICIR1| >= 0.0840. Window 2020-01-06..2026-07-15.
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
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
ATR14 = ((HP - LP) + (HP - CP.shift(1)).abs() + (LP - CP.shift(1)).abs()).rolling(14).mean()

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:26s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | {dec} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic": ics, "passed": passed, "panel": panel}


cands = {}
# ---------- A. return-distribution shape ----------
for nd in (30, 60, 120):
    cands[f"skew_{nd}d"] = RET.rolling(nd).skew()
    cands[f"kurt_{nd}d"] = RET.rolling(nd).kurt()
for nd in (30, 60):
    up = RET.clip(lower=0)
    dn = (-RET).clip(lower=0)
    upv = up.rolling(nd).mean()
    dnv = dn.rolling(nd).mean()
    cands[f"updn_ratio_{nd}d"] = upv / (dnv + 1e-12)          # win/loss magnitude asymmetry
    mean = RET.rolling(nd).mean()
    dsd = RET[RET < 0].rolling(nd).std() if False else RET.rolling(nd).std()
    downside = RET.clip(upper=0).rolling(nd).std()
    cands[f"sortino_{nd}d"] = mean / (downside + 1e-12)        # quality of return
    cands[f"sharpe_{nd}d"] = mean / (RET.rolling(nd).std() + 1e-12)
# ---------- B. drawdown state ----------
for nd in (20, 60, 120):
    rollmax = CP.rolling(nd, min_periods=10).max()
    dd = CP / rollmax - 1.0
    cands[f"dd_{nd}d"] = dd                                   # depth from rolling high (neg = drawdown)
    cands[f"negdd_{nd}d"] = -dd                               # 0 in drawdown, >0 at highs
    twu = (dd < -0.02).rolling(nd).sum() / nd                 # time under water 2%
    cands[f"twu_{nd}d"] = -twu                                # less time underwater = better
cands["dist_52w_high"] = CP / CP.rolling(252, min_periods=60).max() - 1.0
# ---------- C. cross-asset beta ----------
macro_dir = "../persistent/index_data"


def load_macro(name):
    d = pd.read_csv(os.path.join(macro_dir, f"{name}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= pd.Timestamp("2026-07-15")].set_index("date")["close"].astype(float)
    return d.reindex(idx).ffill()


u10 = load_macro("US10Y") if False else closes["US10Y"]["close"].reindex(idx)   # use tradable US10Y
xau = CP["XAU"]
btc = CP["BTC"]
cop = CP["COPPER"]


def roll_beta(y_panel, x, win):
    out = {}
    for s in y_panel.columns:
        y = y_panel[s]
        df = pd.concat([y, x], axis=1).dropna()
        cov = df.iloc[:, 0].rolling(win).cov(df.iloc[:, 1])
        var = df.iloc[:, 1].rolling(win).var()
        out[s] = (cov / var).reindex(idx)
    return pd.DataFrame(out)


for win in (60, 120):
    b_y = roll_beta(RET, u10.diff(), win)     # rate beta (yield change)
    b_xau = roll_beta(RET, xau.pct_change(), win)  # safe-haven beta
    b_btc = roll_beta(RET, btc.pct_change(), win)  # risk-appetite beta
    b_cop = roll_beta(RET, cop.pct_change(), win)  # growth/industrial beta
    cands[f"beta_u10_{win}"] = b_y
    cands[f"beta_xau_{win}"] = b_xau
    cands[f"beta_btc_{win}"] = b_btc
    cands[f"beta_cop_{win}"] = b_cop
    # beta-change: contraction/expansion (positive = beta rising)
    cands[f"dbeta_u10_{win}"] = b_y - b_y.rolling(win).mean()
    cands[f"dbeta_xau_{win}"] = b_xau - b_xau.rolling(win).mean()
# ---------- D. mean-reversion speed ----------
for nd in (20, 60):
    ar1 = {}
    ac = {}
    for s in SYMBOLS:
        r = RET[s]
        a = pd.concat([r, r.shift(1)], axis=1).dropna()
        c = a.iloc[:, 0].rolling(nd).corr(a.iloc[:, 1])
        ac[s] = c.reindex(idx)
    cands[f"ar1_{nd}d"] = pd.DataFrame(ac)                  # AR(1): + = momentum, - = reversal
    cands[f"neg_ar1_{nd}d"] = -pd.DataFrame(ac)
# ---------- E. range-mid momentum ----------
mid = (HP + LP) / 2
wcp = (HP + LP + 2 * CP) / 4
for nd in (10, 20, 60):
    cands[f"mom_mid_{nd}d"] = mid / mid.shift(nd) - 1.0
    cands[f"mom_wcp_{nd}d"] = wcp / wcp.shift(nd) - 1.0
# ---------- F. gap dynamics ----------
gap = OP / CP.shift(1) - 1.0
cands["gap_5d_atr"] = gap.rolling(5).sum() / (ATR14 / CP + 1e-12)
cands["neg_gap_5d_atr"] = -gap.rolling(5).sum() / (ATR14 / CP + 1e-12)
cands["gap_rev_1d"] = -gap
cands["gap_rev_5d"] = -gap.rolling(5).mean()
# ---------- G. vol asymmetry ----------
for nd in (20, 60):
    upr = RET.clip(lower=0)
    dnr = (-RET).clip(lower=0)
    ustd = upr.rolling(nd).std()
    dstd = dnr.rolling(nd).std()
    cands[f"vol_asym_{nd}d"] = ustd / (dstd + 1e-12)

res = [run(n, p) for n, p in cands.items()]
print(f"\nscreen done {time.time()-t0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")
print("\n=== PASSED ===")
for r in res:
    if r["passed"]:
        print(f"{r['name']:26s} IC1={r['ic'][1]['ic']:+.4f} ICIR1={r['ic'][1]['icir']:+.3f} cov={r['cov']:.3f} to={r['to']:.3f}")

# quick correlation vs known library-family panels (quarantined signals) for top passes
import json
def panel_corr(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if m.sum() < 50 or A[m].std() == 0 or B[m].std() == 0:
        return np.nan
    return float(np.corrcoef(A[m], B[m])[0, 1])

lib = {}
lib["rev_1d"] = -RET
h5 = HP.rolling(5).max(); l5 = LP.rolling(5).min()
lib["clv_5d"] = (CP - l5) / (h5 - l5 + 1e-12)
lib["nclv_1d"] = -(CP - LP) / (HP - LP + 1e-12)
lib["nbody_1d"] = -(CP - OP) / (HP - LP + 1e-12)
lib["id_rev_1d"] = -(CP / OP - 1.0)
print("\n=== max |corr| vs known library-family panels for passed candidates ===")
for r in res:
    if not r["passed"]:
        continue
    mx = 0.0; arg = ""
    for k, lp in lib.items():
        c = panel_corr(r["panel"], lp)
        if np.isfinite(c) and abs(c) > mx:
            mx = abs(c); arg = k
    flag = " <-- REDUNDANT(>0.5)" if mx > 0.5 else ""
    print(f"{r['name']:26s} max|corr|={mx:.3f} vs {arg}{flag}")
