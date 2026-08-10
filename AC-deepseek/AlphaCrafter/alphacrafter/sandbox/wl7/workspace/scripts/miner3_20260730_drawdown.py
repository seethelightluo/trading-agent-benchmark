"""miner_3 factor idea #1: trend-position / drawdown-from-high family.
Distance of close from rolling max: close/rolling_max(close,N)-1.
Tests N in {20, 60, 120, 250} plus range position. One idea: trend position."""
import sys
sys.path.insert(0, "scripts")
from miner3_lib import validate_factor, load_panel, load_macro, WATCH
import pandas as pd


def make_dd(n: int):
    def fn(panel, macro):
        return panel / panel.rolling(n).max() - 1.0
    return fn


def make_rangepos(n: int):
    def fn(panel, macro):
        lo = panel.rolling(n).min()
        hi = panel.rolling(n).max()
        return (panel - lo) / (hi - lo)
    return fn


panel = load_panel()
for n in (20, 60, 120, 250):
    validate_factor(f"dd{n}", make_dd(n))
for n in (20, 60):
    validate_factor(f"rangepos{n}", make_rangepos(n))
