"""miner_3 deep validation of passing reversal / CLV candidates - 2026-07-16.
Checks: year-by-year IC, VIX regime split, time drift, per-asset IC, decay,
pairwise candidate correlation (redundancy control), Pearson vs rank IC.
"""
import sys, os, time
import numpy as np
import pandas as pd
from scipy import stats as st

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
RET = CP.pct_change()
vol20 = RET.rolling(20).std() * np.sqrt(252)
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd1 = F.fwd_returns(closes, 1).reindex(idx)

panels = {
    "rev_1d": -RET,
    "rev_2d": -(CP / CP.shift(2) - 1.0),
    "rev_3d": -(CP / CP.shift(3) - 1.0),
    "rev_5d": -(CP / CP.shift(5) - 1.0),
    "rev_3d_vol": -(CP / CP.shift(3) - 1.0) / vol20,
    "clv_1d": (CP - LP) / (HP - LP + 1e-12),
    "clv_5d": (CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min() + 1e-12),
}
N_CELLS = len(idx) * len(SYMBOLS)

print("\n=== full-window metrics ===")
for name, p in panels.items():
    p = p.reindex(idx)
    cov = float(p.notna().sum().sum()) / N_CELLS
    to = F.turnover10(p)
    r = F.fast_ic(p, fwd1)
    print(f"{name:10s} cov={cov:.3f} to={to:.3f} | IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']} obs={r['n_obs']}")

print("\n=== year-by-year IC1 ===")
yr_res = {}
for name, p in panels.items():
    p = p.reindex(idx)
    row = []
    for yr in range(2021, 2027):
        m = (idx >= pd.Timestamp(f"{yr}-01-01")) & (idx <= pd.Timestamp(f"{yr}-12-31"))
        r = F.fast_ic(p.loc[idx[m]], fwd1.loc[idx[m]])
        if r["n_dates"]:
            row.append(f"{yr}:{r['ic']:+.3f}/{r['icir']:+.2f}")
        else:
            row.append(f"{yr}:n/a")
    yr_res[name] = row
    print(f"{name:10s} " + "  ".join(row))

print("\n=== VIX regime split (median) ===")
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")["close"].reindex(idx)
med = vix.median()
for name, p in panels.items():
    p = p.reindex(idx)
    rl = F.fast_ic(p.loc[idx[vix < med]], fwd1.loc[idx[vix < med]])
    rh = F.fast_ic(p.loc[idx[vix >= med]], fwd1.loc[idx[vix >= med]])
    print(f"{name:10s} lowVIX IC={rl['ic']:+.4f} ICIR={rl['icir']:+.3f} n={rl['n_dates']} | "
          f"highVIX IC={rh['ic']:+.4f} ICIR={rh['icir']:+.3f} n={rh['n_dates']}")

print("\n=== time drift: first half vs last half ===")
half_pos = len(idx) // 2
for name, p in panels.items():
    p = p.reindex(idx)
    r1 = F.fast_ic(p.loc[idx[:half_pos]], fwd1.loc[idx[:half_pos]])
    r2 = F.fast_ic(p.loc[idx[half_pos:]], fwd1.loc[idx[half_pos:]])
    print(f"{name:10s} early IC={r1['ic']:+.4f} ICIR={r1['icir']:+.3f} | late IC={r2['ic']:+.4f} ICIR={r2['icir']:+.3f}")

print("\n=== per-asset time-series correlation (factor_t, fwd_ret_{t+1}) ===")
for name, p in panels.items():
    p = p.reindex(idx)
    corrs = {}
    for s in SYMBOLS:
        x = p[s].astype(float)
        y = fwd1[s].astype(float)
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) > 60 and df.iloc[:, 0].std() > 0:
            corrs[s] = st.pearsonr(df.iloc[:, 0], df.iloc[:, 1])[0]
    c = pd.Series(corrs)
    pos = (c > 0).sum()
    print(f"{name:10s} pos={pos}/15 mean_corr={c.mean():+.4f} | "
          + " ".join(f"{s}:{v:+.2f}" for s, v in c.items()))

print("\n=== pairwise rank correlation among candidates (pooled) ===")
stk = pd.DataFrame({n: p.reindex(idx).stack() for n, p in panels.items()})
stk = stk.dropna()
for i, a in enumerate(panels):
    for b in list(panels)[i + 1:]:
        r = st.spearmanr(stk[a], stk[b])[0]
        print(f"rho({a:10s},{b:10s}) = {r:+.3f}")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
