"""Miner3 novel factor family screen (2026-07-16).

Families NOT covered by prior miner screens (momentum/rev/clv/vol/volume/streak/
candle/macro-beta/overnight-intraday):
  A. Return distribution shape: rolling skewness, kurtosis, downside semi-dev,
     tail quantiles, profit/loss ratio, max drawdown depth
  B. Serial dependence: rolling autocorr of returns, variance ratios
  C. Market-beta structure: rolling beta to equal-weight market, downside vs
     upside beta asymmetry, R2 (idiosyncratic share), residual vol, coskewness

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

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)
EPS = 1e-12

cands = {}

# ---------------- A. return distribution shape ----------------
for nd in (20, 60):
    mu = RET.rolling(nd).mean()
    sd = RET.rolling(nd).std()
    r3 = ((RET - mu) ** 3).rolling(nd).mean() / (sd ** 3 + EPS)
    r4 = ((RET - mu) ** 4).rolling(nd).mean() / (sd ** 4 + EPS)
    neg = RET.clip(upper=0.0)
    pos = RET.clip(lower=0.0)
    cands[f"skew_{nd}d"] = r3
    cands[f"kurt_{nd}d"] = r4
    cands[f"downside_vol_share_{nd}d"] = (neg.rolling(nd).std()) / (sd + EPS)
    cands[f"downside_semi_{nd}d"] = -((neg ** 2).rolling(nd).mean() ** 0.5) / (sd + EPS)
    cands[f"q05_{nd}d"] = RET.rolling(nd).quantile(0.05)
    cands[f"q95_{nd}d"] = RET.rolling(nd).quantile(0.95)
    cands[f"pl_ratio_{nd}d"] = pos.rolling(nd).mean() / (neg.rolling(nd).mean().abs() + EPS)
    cands[f"tail_ratio_{nd}d"] = RET.rolling(nd).quantile(0.95) / (RET.rolling(nd).quantile(0.05).abs() + EPS)
    roll_max = CP.rolling(nd).max()
    cands[f"dd_depth_{nd}d"] = (CP - roll_max) / (roll_max + EPS)   # negative = drawdown

# ---------------- B. serial dependence ----------------
def roll_autocorr(x, win):
    a = x.rolling(win).corr(x.shift(1))
    return a

for nd in (20, 60):
    cands[f"ac1_{nd}d"] = roll_autocorr(RET, nd)
    cands[f"sign_ac1_{nd}d"] = roll_autocorr(np.sign(RET), nd)
    for k in (5, 10):
        vk = RET.rolling(nd * k).apply(lambda z: z.reshape(-1, k).sum(axis=1).var(), raw=False) if False else None
    # variance ratio via overlapping sums: var(k-day ret) / (k * var(1-day ret))
    cands[f"vr5_{nd}d"] = RET.rolling(nd).apply(
        lambda z: pd.Series(z).rolling(5).sum().var() / (5.0 * pd.Series(z).var() + EPS), raw=False)
    cands[f"vr10_{nd}d"] = RET.rolling(nd).apply(
        lambda z: pd.Series(z).rolling(10).sum().var() / (10.0 * pd.Series(z).var() + EPS), raw=False)

# ---------------- C. market-beta structure ----------------
mkt = RET.mean(axis=1)  # equal-weight cross-asset market
mkt_up = (mkt > 0).astype(float)
mkt_dn = (mkt < 0).astype(float)
for win in (60, 120):
    beta = {}
    dbeta = {}
    ubeta = {}
    r2 = {}
    resid = {}
    cosk = {}
    for s in SYMBOLS:
        x = RET[s]
        df = pd.concat([x, mkt], axis=1).dropna()
        a, b = df.iloc[:, 0], df.iloc[:, 1]
        cov_ab = a.rolling(win).cov(b)
        var_b = b.rolling(win).var()
        beta[s] = (cov_ab / (var_b + EPS)).reindex(idx)
        # R2 and residual vol
        r2[s] = ((cov_ab ** 2) / ((a.rolling(win).var() * var_b) + EPS)).reindex(idx)
        # coskewness: E[(ri-mui)(rm-mum)^2] / (sigi * sigm^2)
        ma = a.rolling(win).mean(); mb = b.rolling(win).mean()
        sa = a.rolling(win).std(); sb = b.rolling(win).std()
        cosk[s] = (((a - ma) * (b - mb) ** 2).rolling(win).mean() / (sa * sb ** 2 + EPS)).reindex(idx)
        # downside/upside beta (only days in regime)
        ddf = df[df.iloc[:, 1] < 0]
        udf = df[df.iloc[:, 1] > 0]
        if len(ddf) > 20:
            db = ddf.iloc[:, 0].rolling(win, min_periods=20).cov(ddf.iloc[:, 1]) / (
                ddf.iloc[:, 1].rolling(win, min_periods=20).var() + EPS)
            dbeta[s] = db.reindex(idx)
        if len(udf) > 20:
            ub = udf.iloc[:, 0].rolling(win, min_periods=20).cov(udf.iloc[:, 1]) / (
                udf.iloc[:, 1].rolling(win, min_periods=20).var() + EPS)
            ubeta[s] = ub.reindex(idx)
    cands[f"beta_mkt_{win}d"] = pd.DataFrame(beta)
    cands[f"neg_beta_mkt_{win}d"] = -pd.DataFrame(beta)
    cands[f"r2_mkt_{win}d"] = pd.DataFrame(r2)
    cands[f"idio_vol_{win}d"] = -(RET.rolling(win).std() * np.sqrt(1 - pd.DataFrame(r2).clip(0, 1) + EPS))
    cands[f"coskew_{win}d"] = pd.DataFrame(cosk)
    cands[f"down_beta_{win}d"] = pd.DataFrame(dbeta)
    cands[f"up_beta_{win}d"] = pd.DataFrame(ubeta)
    cands[f"beta_asym_{win}d"] = pd.DataFrame(dbeta) - pd.DataFrame(ubeta)

# ---------------- run ----------------
def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to,
            "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}

res = []
for n, p in cands.items():
    res.append(run(n, p))
print(f"\nscreen {time.time()-t0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")

print("\n=== sorted by |IC1| ===")
for r in sorted(res, key=lambda r: -abs(r["ic1"]["ic"])):
    mark = "PASS" if r["passed"] else "   "
    print(f"[{mark}] {r['name']:24s} IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} "
          f"hit={r['ic1']['hit']:.3f} cov={r['cov']:.3f} to={r['to']:.3f}")
