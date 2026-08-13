"""miner_1 2033-05-12: full-library drift re-validation on extended history.

Windows:
  WARM  : 2020-01-01..2026-07-15 (admission reference, all 15 assets)
  OOS   : 2026-07-16..2033-05-11 (online period, live assets only)
  RECENT: last 365d (live assets only)
Admission gate: |IC|>=0.007 and |ICIR|>=0.084 (h=10).
"""
import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, forward_returns, rank_ic_series, VAL_START, VAL_END, factor_to_panel
from miner1_libfuncs import FUNCS, build_refs, LIVE

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=4000)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets, last date {max_date.date()} ({time.time()-t0:.1f}s)", flush=True)
refs = build_refs(prices)

H = 10
fwd = forward_returns(prices, H)
oos_start = VAL_END + pd.Timedelta(days=1)
recent_start = max_date - pd.Timedelta(days=365)
print(f"OOS window: {oos_start.date()} .. {max_date.date()}", flush=True)
print(f"RECENT window: {recent_start.date()} .. {max_date.date()}", flush=True)


def stats(ic):
    if len(ic) < 2:
        return {'ic': float('nan'), 'icir': float('nan'), 'hit': float('nan'), 'n': int(len(ic))}
    sd = ic.std(ddof=1)
    return {'ic': float(ic.mean()), 'icir': float(ic.mean() / sd) if sd > 0 else float('nan'),
            'hit': float((ic > 0).mean()), 'n': int(ic.notna().sum())}


results = {}
for name, fn in FUNCS.items():
    t1 = time.time()
    panel = factor_to_panel(lambda df, s, fn=fn: fn(df, s, ref=refs), prices)
    warm_p = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    warm_ic = rank_ic_series(warm_p, fwd.reindex(warm_p.index), min_valid=8)
    oos_p = panel[(panel.index >= oos_start)]
    oos_ic = rank_ic_series(oos_p[LIVE], fwd.reindex(oos_p.index)[LIVE], min_valid=8)
    rec_p = panel[(panel.index >= recent_start)]
    rec_ic = rank_ic_series(rec_p[LIVE], fwd.reindex(rec_p.index)[LIVE], min_valid=8)
    results[name] = {'warm': stats(warm_ic), 'oos_live': stats(oos_ic), 'recent_live': stats(rec_ic)}
    r = results[name]
    flag = ''
    if r['recent_live']['ic'] is not None and abs(r['recent_live']['ic']) < 0.007 and r['recent_live']['n'] > 30:
        flag = ' <-- RECENT WEAK'
    print(f"{name}: warm_ic={r['warm']['ic']:+.4f}({r['warm']['n']}) "
          f"oos_ic={r['oos_live']['ic']:+.4f}({r['oos_live']['n']}) "
          f"recent_ic={r['recent_live']['ic']:+.4f} icir={r['recent_live']['icir']:+.3f} "
          f"hit={r['recent_live']['hit']:.3f} ({time.time()-t1:.1f}s){flag}", flush=True)

with open('scripts/miner_1_20330512_revalidate_lib.json', 'w') as f:
    json.dump(results, f, indent=1)
print("saved scripts/miner_1_20330512_revalidate_lib.json", flush=True)
