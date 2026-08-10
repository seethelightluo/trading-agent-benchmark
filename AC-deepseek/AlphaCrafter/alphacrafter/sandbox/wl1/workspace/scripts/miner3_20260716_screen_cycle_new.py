"""miner_3 cycle screen 2026-07-16: broad candidate families on canonical grid.
Canonical grid: 2021-01-04..2026-07-15 (1172 dates) x 15 symbols.
Metrics: rank IC(1d), ICIR, hit, coverage, turnover10, decay, year splits, VIX split.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, build_returns
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
CANON = idx[(idx >= pd.Timestamp("2021-01-04"))]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} | canon {len(CANON)} dates")

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
vol5 = RET.rolling(5).std()
vol20 = RET.rolling(20).std()
vol60 = RET.rolling(60).std()

fwd1 = build_returns(closes, 1).reindex(idx)

panels = {
    # --- reversal / location (reference family, previously passes admission) ---
    "rev_1d": -RET,
    "clv_5d": (CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min() + 1e-12),
    "clv_3d": (CP - LP.rolling(3).min()) / (HP.rolling(3).max() - LP.rolling(3).min() + 1e-12),
    # --- intraday vs overnight decomposition ---
    "intraday_rev": -(CP / OP - 1.0),                      # intraday reversal
    "gap_rev": -(OP / CP.shift(1) - 1.0),                  # overnight gap reversal
    "gap_intra_diff": -(OP / CP.shift(1) - 1.0) + (CP / OP - 1.0),  # overnight minus intraday
    # --- volatility / risk family ---
    "vol20_neg": -vol20,                                   # low vol premium
    "range20_neg": -((HP.rolling(20).max() - LP.rolling(20).min()) / (CP + 1e-9)),
    "vol_ratio_5_20_neg": -(vol5 / (vol20 + 1e-12)),       # vol-of-vol reversal
    "downside_vol20_neg": -(RET.where(RET < 0).rolling(20).std()),
    "skew20_neg": -RET.rolling(20).skew(),                 # negative skew premium?
    "kurt20_neg": -RET.rolling(20).kurt(),
    # --- momentum / trend (skip recent to reduce overlap with reversal) ---
    "mom5s1": CP.shift(1) / CP.shift(6) - 1.0,
    "mom20s5": CP.shift(5) / CP.shift(25) - 1.0,
    "mom60s10": CP.shift(10) / CP.shift(70) - 1.0,
    "dd20_rev": -(CP / CP.rolling(20).max() - 1.0),        # distance from 20d high (reversal)
    "rsi14_rev": -(14 - 14 / (1 + RET.rolling(14).apply(lambda x: max(x[x > 0].mean(), 1e-12) / max(-x[x < 0].mean(), 1e-12), raw=True))),
    # --- liquidity / volume ---
    "amt_z5": (VO - VO.rolling(5).mean()) / (VO.rolling(5).std() + 1e-9),   # volume spike
    "amt_z20_neg": -((VO - VO.rolling(20).mean()) / (VO.rolling(20).std() + 1e-9)),
    "vol_adj_mom20": (CP.shift(5) / CP.shift(25) - 1.0) / (vol20 + 1e-12),
    # --- cross-asset beta ---
    "beta_spx60": RET.rolling(60).cov(RET["SPX"]) / (RET["SPX"].rolling(60).var() + 1e-12),
    "beta_ndx60": RET.rolling(60).cov(RET["NDX"]) / (RET["NDX"].rolling(60).var() + 1e-12),
    "beta_wti60": RET.rolling(60).cov(RET["WTI"]) / (RET["WTI"].rolling(60).var() + 1e-12),
}

N_CELLS = len(CANON) * len(SYMBOLS)
print(f"\n=== full-window metrics (canon {len(CANON)} dates x 15) ===")
res = {}
for name, p in panels.items():
    p = p.reindex(CANON)
    cov = float(p.notna().sum().sum()) / N_CELLS
    to = F.turnover10(p)
    r = F.fast_ic(p, fwd1.reindex(CANON))
    res[name] = (r, cov, to)
    flag = "PASS" if (abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084) else ""
    print(f"{name:18s} cov={cov:.3f} to={to:.3f} | IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} "
          f"hit={r['hit']:.3f} n={r['n_dates']} obs={r['n_obs']} {flag}")

print("\n=== decay (IC by horizon) for PASS-ish candidates ===")
for name, (r, cov, to) in res.items():
    if abs(r["ic"]) < 0.005 or abs(r["icir"]) < 0.06:
        continue
    p = panels[name].reindex(CANON)
    dec = F.fast_ic_all(p, closes, horizons=(1, 2, 3, 5, 10, 20, 30), min_names=8)
    print(f"{name:18s} " + " ".join(f"h{h}:{dec[h]['ic']:+.3f}" for h in (1, 2, 3, 5, 10, 20, 30)))

print("\n=== year-by-year IC1 (PASS-ish) ===")
for name, (r, cov, to) in res.items():
    if abs(r["ic"]) < 0.005 or abs(r["icir"]) < 0.06:
        continue
    p = panels[name].reindex(CANON)
    row = []
    for yr in range(2021, 2027):
        m = (CANON >= pd.Timestamp(f"{yr}-01-01")) & (CANON <= pd.Timestamp(f"{yr}-12-31"))
        rr = F.fast_ic(p.loc[CANON[m]], fwd1.loc[CANON[m]])
        row.append(f"{yr}:{rr['ic']:+.3f}/{rr['icir']:+.2f}" if rr["n_dates"] else f"{yr}:n/a")
    print(f"{name:18s} " + "  ".join(row))

print("\n=== VIX regime split (PASS-ish) ===")
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")["close"].reindex(CANON)
med = vix.median()
for name, (r, cov, to) in res.items():
    if abs(r["ic"]) < 0.005 or abs(r["icir"]) < 0.06:
        continue
    p = panels[name].reindex(CANON)
    rl = F.fast_ic(p.loc[CANON[vix < med]], fwd1.loc[CANON[vix < med]])
    rh = F.fast_ic(p.loc[CANON[vix >= med]], fwd1.loc[CANON[vix >= med]])
    print(f"{name:18s} lowVIX IC={rl['ic']:+.4f} ICIR={rl['icir']:+.3f} n={rl['n_dates']} | "
          f"highVIX IC={rh['ic']:+.4f} ICIR={rh['icir']:+.3f} n={rh['n_dates']}")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
