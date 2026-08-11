"""miner_2 probe (2027-03-25): data coverage & canonical grid vs library artifacts."""
import sys, json, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, canonical_grid, VAL_START, VAL_END

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
maxd = max(d.index.max() for d in prices.values())
print(f"canonical grid: {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)
print(f"full data end: {maxd.date()} | days requested 3000", flush=True)
for s, df in prices.items():
    print(f"  {s}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}", flush=True)

# library artifacts grid check
n_ok = 0; n_mismatch = 0
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        shape = arr.shape
        ag = d.get('signal_artifact_grid', {})
        match = shape == (T, N)
        if match:
            n_ok += 1
        else:
            n_mismatch += 1
        print(f"lib {fid}: artifact {shape} grid {ag.get('n_dates')} {ag.get('start')}..{ag.get('end')} match_current={match}", flush=True)
    except Exception as e:
        print(f"lib load error {f}: {e}", flush=True)
print(f"artifact match: {n_ok} ok, {n_mismatch} mismatch", flush=True)

# online window
all_dates = sorted(set().union(*[set(d.index) for d in prices.values()]))
grid2 = pd.DatetimeIndex([d for d in all_dates if d > VAL_END])
print(f"online grid: {len(grid2)} dates {grid2.min().date()}..{grid2.max().date()}", flush=True)

# index data availability
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx = load_index(s, days=3000, prices=prices)
    if idx is not None:
        print(f"index {s}: {len(idx)} rows {idx.index.min().date()}..{idx.index.max().date()}", flush=True)
    else:
        print(f"index {s}: None", flush=True)
