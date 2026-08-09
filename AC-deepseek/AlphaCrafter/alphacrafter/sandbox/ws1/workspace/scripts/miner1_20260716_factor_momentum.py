"""Factor family: cross-asset momentum (classic, with 5-day skip). One idea: momentum."""
import numpy as np
from miner1_20260716_lib import validate_factor, decay_table, regime_breakdown, forward_returns, daily_ic


def make_mom(lookback, skip=5):
    def fn(sym, close, volume):
        return close.shift(skip) / close.shift(skip + lookback) - 1.0
    return fn


if __name__ == '__main__':
    for lb in (10, 20, 60, 120):
        label = f'mom_{lb}d_skip5'
        panel, fac, results = validate_factor(label, make_mom(lb), min_valid=8)
        decay_table(results)
        ret = forward_returns(panel['closes'], panel['grid'], 1)
        ics = daily_ic(fac, ret, min_valid=8)
        regime_breakdown(ics, panel, label)
        print('=' * 80)
