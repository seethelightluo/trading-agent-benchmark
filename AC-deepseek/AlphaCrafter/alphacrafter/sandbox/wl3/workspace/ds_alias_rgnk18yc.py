import json, numpy as np, pandas as pd
from pathlib import Path
# Check artifact shapes
for p in sorted(Path('factors').glob('*_signal.npy'))[:5]:
    arr = np.load(p, allow_pickle=False)
    print(p.name, arr.shape)
# Check grid from current data
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, canonical_grid, VAL_START, VAL_END
prices = load_prices(days=3000)
print("prices loaded:", len(prices))
for s, df in prices.items():
    print(s, df.index.min(), df.index.max(), len(df))
    break
grid = canonical_grid(prices)
print("canonical grid len:", len(grid), grid.min(), grid.max())