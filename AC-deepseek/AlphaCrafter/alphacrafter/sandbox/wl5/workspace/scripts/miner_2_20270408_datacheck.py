# -*- coding: utf-8 -*-
"""miner_2 2027-04-08 cycle: data availability check.
Visible data through previous completed trading day (2027-04-07).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

VIS = '2027-04-08'
C, V, H, Lw, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lw, O = C[mask], V[mask], H[mask], Lw[mask], O[mask]
print("panel shape:", C.shape)
print("date range:", C.index.min(), "->", C.index.max())
print("last 5 dates:", [str(d.date()) for d in C.index[-5:]])
print("\nper-asset valid close rows:")
print(C.notna().sum().to_string())
print("\nper-asset volume non-null ratio:")
print((V.notna().mean()).round(3).to_string())
print("\nlast close values:")
print(C.iloc[-1].round(2).to_string())
