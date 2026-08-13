"""miner_2 2032-06-24: probe data availability and recent live window."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, VAL_END

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3400)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, range {min_date.date()}..{max_date.date()} ({time.time()-t0:.1f}s)", flush=True)
for s in WATCHLIST:
    dd = prices.get(s)
    if dd is None:
        print(f"  {s}: MISSING")
    else:
        print(f"  {s}: {len(dd)} rows, {dd.index.min().date()}..{dd.index.max().date()}, last close={dd['close'].iloc[-1]:.4f}")
print(f"VAL_END (warm-up end): {VAL_END.date()}", flush=True)
print(f"OOS days since warm-up: {(max_date - VAL_END).days}", flush=True)

# index signals
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    ix = load_index(s, prices=prices)
    if ix is None:
        print(f"  index {s}: MISSING")
    else:
        print(f"  index {s}: {len(ix)} rows, {ix.index.min().date()}..{ix.index.max().date()}, last={ix['close'].iloc[-1]:.4f}")

# quick recent regime stats: last 6 months daily returns
r6 = {}
for s, dd in prices.items():
    r = dd['close'].pct_change().tail(126)
    r6[s] = (r.mean() * 252, r.std() * np.sqrt(252), r.sum())
print("\nlast-6M ann ret / ann vol / cum ret:")
for s, (mu, sd, cum) in sorted(r6.items(), key=lambda kv: -kv[1][2]):
    print(f"  {s:10s} {mu:+.2%} {sd:.2%} {cum:+.2%}")
