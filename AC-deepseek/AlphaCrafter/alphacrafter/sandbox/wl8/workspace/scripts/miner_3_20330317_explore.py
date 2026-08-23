"""
miner_3 2033-03-17 exploration: fresh candidate factors on the 15-instrument
cross-asset tradable universe (all data through current date; no lookahead).

Motivation: library is empty (all evicted/deprecated), trader runs fallback
mom10/vix-beta/yield-beta ensemble. We need robust, low-correlation factors
with |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d admission horizon.

Candidates (interpretable, distinct from evicted library):
 A) lr2_trend_60     : R^2 of log-price linear trend over 60d (trend certainty)
 B) gap_intensity_20 : mean |open/prev_close - 1| over 20d (overnight shock)
 C) serr_ac_20       : 1-day return autocorrelation over 20d (trend persistence)
 D) close_loc_20     : mean (close-low)/(high-low) over 20d (close location)
 E) down_freq_60     : share of down days over 60d (drawdown frequency)
 F) updown_asym_20   : up-day mean ret - down-day mean ret (signed asymmetry)
 G) gk_vol_10x40     : vol-growth: Garman-Klass 10d vol / 40d vol (vol regime)
 H) maxgap_20        : max |1-day return| over 20d (tail shock recency)
"""
import sys, os
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import factor_validation_lib as fvl

CURRENT = "2033-03-17"

# ---- factor functions signature: fn(close, vol, open_, high, low, macro, **params)
# returns pandas Series aligned to close.index

def f_lr2_trend(close, vol, open_, high, low, macro, w=60):
    c = close.astype(float)
    lc = np.log(c)
    out = pd.Series(np.nan, index=c.index)
    x = np.arange(w)
    denom = (x - x.mean()) ** 2
    for i in range(w - 1, len(c)):
        y = lc.iloc[i - w + 1: i + 1].values
        if not np.all(np.isfinite(y)):
            continue
        x_mean = x.mean()
        beta =