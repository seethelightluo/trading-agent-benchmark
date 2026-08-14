"""miner_3 20340904: Factor idea #1 - intraday close-location value (CLV) 20d.

Idea: over the last 20 days, how often does each asset close near the top of its
intraday range? CLV = (close-low)/(high-low). Persistent intraday buying pressure
(closing near highs) is a demand-side signal that may predict continued strength
over the next 10 days (positive IC), while persistent closing near lows signals
supply pressure / weakness (negative forward returns).

This is distinct from evicted hl_pos_20d (close vs rolling min/max of CLOSE) and
from volume/momentum families. Uses only OHLC.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_common import (ASSETS, load_ohlc, run_full_validation,
                           load_visible_through)

END = load_visible_through()
print("visible through:", END)

ohlc = load_ohlc(END)
clv = {}
for a in ASSETS:
    df = ohlc[a]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    c = (df["close"] - df["low"]) / rng
    clv[a] = c.rolling(20).mean()
panel = pd.DataFrame(clv).sort_index()
print("panel shape:", panel.shape)

res = run_full_validation(panel, label="intraday_clv_20d", horizon=10)
