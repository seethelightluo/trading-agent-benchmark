"""miner1 2031-07-04: deep validation of trend-cond momentum candidates that tripped the gate.
Candidates: tcm_20x60 (h=10, negative IC), tcm_10x120 (h=5, marginal positive).
Checks: sub-period stability, HSI/CN10Y flat-artifact sensitivity, turnover, decay profile.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner1_20310704_helpers import load_panel, daily_ic, summarize, GATE_IC, GATE_ICIR

panel = load_panel()
close = panel['close']

def trend_cond_mom(close, k=10, long=60, flat_z=0.5):
    mom = close / close.shift(k) - 1.0
    ma_long = close.rolling(long).mean()
    z = (close - ma_long) / close.rolling(long).std()
    trend = np.sign(z).where(z.abs() >= flat_z, 0.0)
    return mom * trend

tcm_20x60 = trend_cond_mom(close, 20, 60, 0.5)
tcm_10x120 = trend_cond_mom(close, 10, 120, 0.5)

def fwd(close, h):
    return close.shift(-h) / close - 1.0

def run_detail(name, f, horizon, window, exclude=()):
    ff = f.loc[window[0]:window[1]]
    fr = fwd(close, horizon).loc[window[0]:window[1]]
    if exclude:
        ff = ff.drop(columns=list(exclude))
        fr = fr.drop(columns=list(exclude))
    ic_s = daily_ic(ff, fr)
    s = summarize(ic_s)
    print(f"{name} h={horizon} {window} excl={exclude or '-'}: "
          f"IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_dates']}")
    return ic_s, s

print("=" * 100)
print("A) tcm_20x60 h=10 - sub-period stability (incl + excl flat artifacts)")
subs = [('2020-06-01', '2022-12-31'), ('2023-01-01', '2025-12-31'),
        ('2026-01-01', '2028-06-30'), ('2028-07-01', '2031-07-03')]
for w in subs:
    run_detail('tcm20x60', tcm_20x60, 10, w)
    run_detail('tcm20x60', tcm_20x60, 10, w, exclude=['HSI', 'CN10Y'])

print()
print("B) tcm_10x120 h=5 - sub-period stability")
for w in subs:
    run_detail('tcm10x120', tcm_10x120, 5, w)
    run_detail('tcm10x120', tcm_10x120, 5, w, exclude=['HSI', 'CN10Y'])

print()
print("C) Decay profile tcm_20x60 (recent 2026-2031)")
for h in [1, 2, 3, 5, 8, 10, 15, 20]:
    ic_s, s = run_detail('tcm20x60', tcm_20x60, h, ('2026-01-01', '2031-07-03'))

print()
print("D) Decay profile tcm_10x120 (last3y)")
for h in [1, 2, 3, 5, 8, 10, 15, 20]:
    ic_s, s = run_detail('tcm10x120', tcm_10x120, h, ('2028-07-01', '2031-07-03'))

print()
print("E) Turnover (recent window, rank-based mean abs daily change)")
for name, f in [('tcm_20x60', tcm_20x60), ('tcm_10x120', tcm_10x120)]:
    ff = f.loc['2026-01-01':'2031-07-03']
    ranks = ff.rank(axis=1) / ff.notna().sum(axis=1)
    to = ranks.diff().abs().mean().mean()
    print(f"  {name}: turnover_rank={to:.4f}")
    # value-based turnover normalized by cross-sectional std
    z = (ff - ff.mean(axis=1)) / ff.std(axis=1)
    zto = z.diff().abs().mean().mean()
    print(f"  {name}: z_turnover={zto:.4f}")

print()
print("F) Gate check recap (|IC|>=0.007, |ICIR|>=0.084)")
print(f"  tcm_20x60 h=10 recent: |IC|={abs(-0.0383):.4f} |ICIR|={abs(-0.136):.4f} -> "
      f"{'PASS' if abs(-0.0383)>=GATE_IC and abs(-0.136)>=GATE_ICIR else 'FAIL'}")
print(f"  tcm_10x120 h=5 last3y: |IC|={abs(0.0252):.4f} |ICIR|={abs(0.087):.4f} -> "
      f"{'PASS' if abs(0.0252)>=GATE_IC and abs(0.087)>=GATE_ICIR else 'FAIL'}")
