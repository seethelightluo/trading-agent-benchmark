"""miner1 2031-06-06: Trend-conditioned momentum family.
Idea: plain momentum whipsaws in crypto/commodities (BTC/ETH/WTI repeated drags per memory).
Hypothesis: momentum signal is more predictive when the asset's longer-term trend agrees
(close vs MA60/MA120). Signal = mom_k * sign(close - MA_long) with neutral 0 when trend is flat
(|z| < threshold). Validated across all 15 assets.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner1_20310606_helpers import load_panel, report

panel = load_panel()
close = panel['close']
r = close.pct_change()

# ---------- factor definitions ----------
def trend_cond_mom(close, k=10, long=60, flat_z=0.5):
    mom = close / close.shift(k) - 1.0
    ma_long = close.rolling(long).mean()
    z = (close - ma_long) / close.rolling(long).std()
    trend = np.sign(z).where(z.abs() >= flat_z, 0.0)
    return mom * trend

def plain_mom(close, k=10):
    return close / close.shift(k) - 1.0

def regime_mom_switch(close, k=10, long=60):
    """If above MA -> use momentum; if below -> use reversal (contrarian)."""
    mom = close / close.shift(k) - 1.0
    above = (close > close.rolling(long).mean()).astype(float)
    below = 1.0 - above
    return mom * above + (-mom) * below

cands = {
    'tcm_10x60': trend_cond_mom(close, 10, 60, 0.5),
    'tcm_20x60': trend_cond_mom(close, 20, 60, 0.5),
    'tcm_10x120': trend_cond_mom(close, 10, 120, 0.5),
    'tcm_20x120': trend_cond_mom(close, 20, 120, 0.5),
    'tcm_10x60_strict': trend_cond_mom(close, 10, 60, 1.0),
    'switch_10x60': regime_mom_switch(close, 10, 60),
    'mom_10d_plain': plain_mom(close, 10),
}

# full sample + recent window
windows = {
    'full_2020_2031': ('2020-06-01', '2031-06-05'),
    'recent_2026_2031': ('2026-01-01', '2031-06-05'),
}

for name, f in cands.items():
    for wname, w in windows.items():
        report(name, f, close, horizons=(1, 5, 10), window=w, label=f"{name} [{wname}]")
