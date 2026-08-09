"""Factor family: trend (price vs moving average), risk-adjusted trend, vol, drawdown, efficiency."""
import numpy as np
from miner1_20260716_lib import validate_factor, decay_table, forward_returns, daily_ic, regime_breakdown


def make_trend(win):
    def fn(sym, close, volume):
        ma = close.rolling(win).mean()
        return close / ma - 1.0
    return fn


def risk_adj_trend(win=20):
    def fn(sym, close, volume):
        ret = close.pct_change()
        mu = ret.rolling(win).mean()
        sd = ret.rolling(win).std()
        return (mu / sd).replace([np.inf, -np.inf], np.nan)
    return fn


def inv_vol(win=20):
    def fn(sym, close, volume):
        sd = close.pct_change().rolling(win).std()
        return (-sd).replace([np.inf, -np.inf], np.nan)
    return fn


def drawdown_dist(win=60):
    def fn(sym, close, volume):
        return 1.0 - close / close.rolling(win).max()
    return fn


def efficiency_ratio(win=60):
    def fn(sym, close, volume):
        net = (close - close.shift(win)).abs()
        path = close.diff().abs().rolling(win).sum()
        return (net / path).replace([np.inf, -np.inf], np.nan)
    return fn


def vol_of_vol(win=20, sub=60):
    def fn(sym, close, volume):
        sd = close.pct_change().rolling(win).std()
        return sd.rolling(sub).std()
    return fn


if __name__ == '__main__':
    candidates = [
        ('trend_sma20', make_trend(20)),
        ('trend_sma60', make_trend(60)),
        ('trend_sma120', make_trend(120)),
        ('risk_adj_trend20', risk_adj_trend(20)),
        ('inv_vol20', inv_vol(20)),
        ('drawdown_60d', drawdown_dist(60)),
        ('efficiency_60d', efficiency_ratio(60)),
        ('vol_of_vol', vol_of_vol()),
    ]
    for label, fn in candidates:
        panel, fac, results = validate_factor(label, fn, min_valid=8)
        decay_table(results)
        ret = forward_returns(panel['closes'], panel['grid'], 5)
        ics = daily_ic(fac, ret, min_valid=8)
        regime_breakdown(ics, panel, label)
        print('=' * 80)
