"""miner_1 cycle 2033-11-10. Visible through 2033-11-09.
Exp01: settlement-price location factor (SETTLE_LOC).
Motivation: in a futures/futures-like cross-asset universe, where the daily
close is the exchange settlement print, the depth of supportive buying late in
a session shows up as a strong close relative to the day's traded range. Assets
that routinely settle at the top of their range are being accumulated;
consistent settlement at the bottom flags auction-driven supply.
Construction: each day, settle = (close - low)/(high - low) in [0,1]; the w-day
mean of settle captures average close location. Direction +1 = prefer assets
settling high in range. Also swept: weaker-momentum inverse, and dispersion of
location (low dispersion = consistent buyer).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from miner_1_lib import build_panel, compute_ic, coverage, turnover, decay_ic, report

closes, highs, lows, vols, rets = build_panel()
idx = closes.index
print(f"Panel: {closes.shape[0]} dates x {closes.shape[1]} assets, "
      f"{idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}", flush=True)

def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5, fwd10, fwd20 = fwd(5), fwd(10), fwd(20)

rng = (highs - lows).replace(0, np.nan)
settle = ((closes - lows) / rng).clip(0, 1)  # 1 = closed at high, 0 = closed at low

F = {}
for w in (10, 20, 40):
    sm = settle.rolling(w).mean()
    F[f'settle_loc_{w}'] = sm
    F[f'settle_weak_{w}'] = -sm
    F[f'settle_loc_cons_{w}'] = -settle.rolling(w).std()

print("\n===== SETTLEMENT-LOCATION FAMILY (full window through 2033-11-09) =====", flush=True)
results = {}
for name, fv in F.items():
    a, ok = report(name, fv, fwd5, fwd10, fwd20)
    results[name] = (a, ok, fv)

print("\n===== PASSERS: RECENT 2Y / 1Y + COVERAGE + TURNOVER + DECAY =====", flush=True)
s2 = pd.Timestamp('2031-11-10'); s1 = pd.Timestamp('2032-11-10')
for name, (a, ok, fv) in results.items():
    if ok:
        f = fv.reindex(fwd10.index)
        r2 = compute_ic(f.loc[f.index >= s2], fwd10.loc[fwd10.index >= s2])
        r1 = compute_ic(f.loc[f.index >= s1], fwd10.loc[fwd10.index >= s1])
        cov, d8 = coverage(f)
        to = turnover(f)
        dec = decay_ic(f, rets)
        print(f"\n{name}: FULL IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']} hit={a['hit']:.3f}", flush=True)
        print(f"  recent2y IC={r2 if False else None}", flush=True)
        print(f"  recent2y IC={r2['IC']:+.4f} ICIR={r2['ICIR']:+.4f} n={r2['n']} | 1y IC={r1['IC']:+.4f} ICIR={r1['ICIR']:+.4f} n={r1['n']}", flush=True)
        print(f"  cov={cov:.3f} dates_ge8={d8:.3f} turn={to:.3f} decay={ {k: round(v,3) for k,v in dec.items()} }", flush=True)
print("\nDONE", flush=True)