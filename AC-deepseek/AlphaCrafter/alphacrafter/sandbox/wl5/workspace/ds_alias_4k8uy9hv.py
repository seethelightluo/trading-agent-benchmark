
import sys
sys.path.insert(0, 'scripts')
import miner3_lib as L
import pandas as pd
import numpy as np

C, V, H, Lo, O = L.load_close_panel(4000)
# Volume coverage by asset
vol_nonzero = (V > 0).sum()
vol_total = V.notna().sum()
print("Volume nonzero coverage by asset:")
for c in V.columns:
    nz = (V[c] > 0).sum()
    print(f"  {c}: {nz}/{len(V)} nonzero ({nz/len(V)*100:.0f}%)")
print("\nVolume constant (std=0 in last 60d):")
for c in V.columns:
    s = V[c].tail(60)
    print(f"  {c}: std={s.std():.4f}")
