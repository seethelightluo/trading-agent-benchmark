import sys, numpy as np
sys.path.insert(0, 'scripts')
from factor_common import load_prices, canonical_grid, WATCHLIST
for days in (2200, 2500, 3000, 3500):
    prices = load_prices(days=days)
    lens = {s: len(d) for s, d in prices.items()}
    grid = canonical_grid(prices)
    print(f"days={days}: grid_len={len(grid)} min={grid.min().date()} max={grid.max().date()} | asset lens: min={min(lens.values())} max={max(lens.values())}")
    print("   BTC len:", lens.get('BTC'), "ETH len:", lens.get('ETH'), "000300 len:", lens.get('000300.SH'))
# Also check npy shapes again
from pathlib import Path
shapes = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    arr = np.load(p, allow_pickle=False)
    shapes.setdefault(arr.shape[0], []).append(p.name)
print("artifact shape-0 distribution:", {k: len(v) for k, v in shapes.items()})