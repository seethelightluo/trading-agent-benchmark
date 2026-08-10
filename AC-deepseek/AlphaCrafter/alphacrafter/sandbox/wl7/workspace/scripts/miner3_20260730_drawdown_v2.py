"""miner_3 factor idea #1 (fixed): trend-position / drawdown-from-high family.
Computed per asset on its own calendar to avoid union-calendar NaN gaps.
Factor = close/rolling_max(close,N)-1 (drawdown depth) and range position.
One idea per script: trend position vs recent range."""
import sys
sys.path.insert(0, "scripts")
from miner3_lib import validate_factor, per_asset, load_panel
import pandas as pd


def make_dd(n: int):
    def fn(panel, macro):
        return per_asset(lambda s: s / s.rolling(n).max() - 1.0)(panel, macro)
    return fn


def make_rangepos(n: int):
    def fn(panel, macro):
        def f(s):
            lo = s.rolling(n).min()
            hi = s.rolling(n).max()
            rng = (hi - lo).replace(0, pd.NA)
            return (s - lo) / rng
        return per_asset(f)(panel, macro)
    return fn


panel = load_panel()
for n in (20, 60, 120, 250):
    validate_factor(f"dd{n}", make_dd(n))
for n in (20, 60):
    validate_factor(f"rangepos{n}", make_rangepos(n))
