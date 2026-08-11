"""Sub-period IC stability check for xs_dev_5 and bond_beta_diff_60.

Checks IC/ICIR over disjoint 12-month windows plus the original cycle-8 eval
window (2021-01-04..2026-07-15) to compare with screen results.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel, daily_rank_ic

close = load_close_panel()
ret = close.pct_change()
lr = np.log(close / close.shift(1))

# --- xs_dev_5 ---
ret5 = close.pct_change(5)
xs_mean5 = ret5.mean(axis=1)
xs_dev5 = ret5.sub(xs_mean5, axis=0)

# --- bond beta diff ---
def roll_beta(x, m, win=60, minp=30):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        out[s] = x[s].rolling(win, min_periods=minp).cov(m) / (m.rolling(win, min_periods=minp).var() + 1e-12)
    return out

bbd = roll_beta(lr, lr["US10Y"]) - roll_beta(lr, lr["CN10Y"])

fwd1 = close.pct_change(-1)  # 1-day forward
fwd5 = close.shift(-5) / close - 1.0
fwd10 = close.shift(-10) / close - 1.0

subs = [
    ("2020-01..2020-12", "2020-01-02", "2020-12-31"),
    ("2021-01..2021-12", "2021-01-04", "2021-12-31"),
    ("2022-01..2022-12", "2022-01-03", "2022-12-30"),
    ("2023-01..2023-12", "2023-01-03", "2023-12-29"),
    ("2024-01..2024-12", "2024-01-02", "2024-12-31"),
    ("2025-01..2025-12", "2025-01-02", "2025-12-31"),
    ("2026-01..2026-11", "2026-01-02", "2026-11-26"),
    ("cycle8_eval_210104..260715", "2021-01-04", "2026-07-15"),
]

for name, fac in [("xs_dev_5", xs_dev5), ("bond_beta_diff_60", bbd)]:
    print(f"\n=== {name} ===")
    for label, s, e in subs:
        idx = fac.index[(fac.index >= s) & (fac.index <= e)]
        if len(idx) < 50:
            continue
        f = fac.loc[idx]
        for h, fwd in [(1, fwd1), (5, fwd5), (10, fwd10)]:
            ic = daily_rank_ic(f, fwd.loc[idx])
            if len(ic) < 30:
                continue
            m = ic.mean()
            sd = ic.std(ddof=1)
            icir = m / sd if sd > 0 else 0
            print(f"  {label:28s} h={h:2d} IC={m:+.5f} ICIR={icir:+.5f} n={len(ic)}")
