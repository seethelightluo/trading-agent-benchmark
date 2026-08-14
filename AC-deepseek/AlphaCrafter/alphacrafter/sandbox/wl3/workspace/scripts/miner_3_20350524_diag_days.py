"""Find which load_prices(days=...) reproduces the dominant 2388-date library grid."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, canonical_grid, VAL_START, VAL_END
import pandas as pd

for days in [3500, 4000, 4200, 4400, 4600, 5000, 5500, 6000, 7000]:
    prices = load_prices(days=days)
    # reset cache
    import factor_common as fc
    fc._CANON_GRID = None
    grid = canonical_grid(prices)
    starts = {s: str(df.index.min().date()) for s, df in prices.items()}
    n_full = sum(1 for s, df in prices.items() if df.index.min() <= VAL_START)
    print(f"days={days:5d} grid_n={len(grid):5d} grid_start={grid.min().date()} grid_end={grid.max().date()} assets_with_full={n_full}")
