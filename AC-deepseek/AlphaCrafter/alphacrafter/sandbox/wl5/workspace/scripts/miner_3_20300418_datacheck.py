# -*- coding: utf-8 -*-
"""miner_3 2030-04-18: quick data availability check through previous completed trading day."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(5000)
print('Close panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))
print('Last 5 dates:', [str(d.date()) for d in C.index[-5:]])
print('NaN tail per asset:')
print(C.tail(3).isna().sum(axis=0).to_dict())

for name in ['DXY', 'VIX', 'USDJPY', 'USDCNY', 'EURUSD']:
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    last = df['date'].max()
    print('%s last date: %s  rows=%d' % (name, last.date(), len(df)))

print('Volume availability tail:')
print(V.tail(3).notna().sum(axis=1).to_dict())
