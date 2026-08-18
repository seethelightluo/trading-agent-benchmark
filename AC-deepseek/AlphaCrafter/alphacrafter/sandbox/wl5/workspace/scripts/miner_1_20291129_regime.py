# -*- coding: utf-8 -*-
"""miner_1 2029-11-29: regime snapshot - recent cross-asset moves and macro state."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20291129_common import (
    TRADABLE, MACRO, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
)

C = ohlcv_panels()['close']
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

last = C.index[-1]
print('\n=== Cross-asset returns as of %s ===' % last.date())
print('%-10s %8s %8s %8s %8s %8s %8s' % ('asset', '1d', '5d', '10d', '40d', '120d', 'YTD'))
for s in TRADABLE:
    c = C[s].dropna()
    if len(c) < 130:
        continue
    r1 = c.iloc[-1] / c.iloc[-2] - 1
    r5 = c.iloc[-1] / c.iloc[-6] - 1
    r10 = c.iloc[-1] / c.iloc[-11] - 1
    r40 = c.iloc[-1] / c.iloc[-41] - 1
    r120 = c.iloc[-1] / c.iloc[-121] - 1
    ytd = c.iloc[-1] / c.loc[c.index >= '2029-01-01'].iloc[0] - 1
    print('%-10s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%%' % (
        s, 100 * r1, 100 * r5, 100 * r10, 100 * r40, 100 * r120, 100 * ytd))

print('\n=== Macro state ===')
for m in MACRO:
    s = macro_panel(m)
    r5 = s.iloc[-1] / s.iloc[-6] - 1
    r20 = s.iloc[-1] / s.iloc[-21] - 1
    r60 = s.iloc[-1] / s.iloc[-61] - 1
    print('%-8s last=%10.3f  5d=%7.2f%%  20d=%7.2f%%  60d=%7.2f%%' % (
        m, s.iloc[-1], 100 * r5, 100 * r20, 100 * r60))

# MA positioning
print('\n=== MA positioning (close vs MA20/MA60/MA120) ===')
for s in TRADABLE:
    c = C[s].dropna()
    if len(c) < 130:
        continue
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    ma120 = c.rolling(120).mean().iloc[-1]
    px = c.iloc[-1]
    print('%-10s close=%12.4f  vsMA20=%+.2f%% vsMA60=%+.2f%% vsMA120=%+.2f%%' % (
        s, px, 100 * (px / ma20 - 1), 100 * (px / ma60 - 1), 100 * (px / ma120 - 1)))

# Rolling 20d vol snapshot
print('\n=== 20d realized vol (annualized) ===')
vol20 = R.rolling(20).std() * np.sqrt(252)
for s in TRADABLE:
    v = vol20[s].dropna()
    if len(v) < 25:
        continue
    print('%-10s vol20=%6.1f%%  (1y ago=%6.1f%%)' % (s, 100 * v.iloc[-1], 100 * v.iloc[-252] if len(v) > 252 else float('nan')))
