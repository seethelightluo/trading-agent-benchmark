# -*- coding: utf-8 -*-
"""miner_3 2027-08-12: verify data panel state before factor research."""
import sys
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
print("Panel: %s -> %s | %d dates x %d assets" % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))
print("Assets:", list(C.columns))
print("\nPer-asset last 5 close rows:")
print(C.tail(5).to_string())
print("\nVolume zeros per asset (fraction):")
print((V == 0).mean().round(3).to_string())
print("\nHigh missing per asset (fraction):")
print(H.isna().mean().round(3).to_string())
print("\nMissing close per asset (fraction):")
print(C.isna().mean().round(3).to_string())
