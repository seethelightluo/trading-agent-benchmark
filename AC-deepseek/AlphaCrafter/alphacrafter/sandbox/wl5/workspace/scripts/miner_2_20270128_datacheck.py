# -*- coding: utf-8 -*-
"""miner_2 2027-01-28 cycle: data availability check through 2027-01-27."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lo, O = L.load_close_panel(4000)
print("Close panel shape:", C.shape)
print("Date range:", C.index.min(), "->", C.index.max())
print("\nPer-asset coverage (close):")
print(C.notna().sum())
print("\nPer-asset volume coverage (non-null):")
print(V.notna().sum())
print("\nLast 5 dates:")
print(C.index[-5:].strftime('%Y-%m-%d').tolist())

# check how many dates have >=8 valid closes in the last 250 trading days
last250 = C.tail(250)
ge8 = last250.notna().sum(axis=1) >= 8
print("\nDates with >=8 valid closes (last 250d):", int(ge8.sum()), "/ 250")

# check OHLC data availability for candidates
print("\nHigh/Low panel shapes:", H.shape, Lo.shape, O.shape)
print("NaN in H last 60d:", H.tail(60).isna().sum().sum())
