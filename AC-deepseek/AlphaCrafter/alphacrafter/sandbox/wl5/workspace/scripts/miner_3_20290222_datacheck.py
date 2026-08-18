# -*- coding: utf-8 -*-
"""miner_3 2029-02-22: data availability check before factor exploration.
Checks close panel range, macro series range, volume coverage per asset."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Close panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))
print('Columns:', list(C.columns))
print('\nPer-asset history start (first valid close):')
for s in C.columns:
    first = C[s].first_valid_index()
    last = C[s].last_valid_index()
    n = C[s].notna().sum()
    print('  %-9s first=%s last=%s n_valid=%d' % (s, first.date(), last.date(), n))

print('\nVolume coverage (valid share of panel):')
for s in V.columns:
    n = V[s].notna().sum()
    print('  %-9s n_valid=%d share=%.3f' % (s, n, n / len(V)))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

for m in ['DXY', 'USDJPY', 'EURUSD', 'USDCNY', 'VIX']:
    s = load_macro(m)
    print('macro %-7s first=%s last=%s n_valid=%d' % (m, s.first_valid_index().date(), s.last_valid_index().date(), s.notna().sum()))

# sanity: last 5 closes
print('\nLast 5 trading days close:')
print(C.tail(5).round(4).to_string())
