# -*- coding: utf-8 -*-
"""miner_3 2030-06-13: datacheck - verify data range, universe, macro availability."""
import sys
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(5000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))
print('Assets:', list(C.columns))

# per-asset last date
last = C.apply(lambda s: s.dropna().index.max())
print('\nLast available date per asset:')
for a in C.columns:
    print('  %-10s %s  (n=%d)' % (a, last[a].date(), C[a].notna().sum()))

# macro data
for name in ['DXY', 'VIX', 'USDJPY', 'USDCNY', 'EURUSD']:
    try:
        df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.set_index('date').sort_index()
        print('macro %-7s -> %s  n=%d' % (name, df.index.max().date(), len(df)))
    except Exception as e:
        print('macro %s ERROR %s' % (name, e))

# recent returns snapshot
print('\nLast 5 rows of close panel (transposed, tail):')
print(C.tail(5).T)
