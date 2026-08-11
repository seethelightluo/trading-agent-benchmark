import pandas as pd, numpy as np
panel = pd.read_pickle("scripts/panel_cache.pkl")
for k, v in panel.items():
    print(k, type(v), getattr(v, 'shape', None))
print()
print("close idx:", panel['close'].index.min(), panel['close'].index.max(), len(panel['close']))
print("macro cols:", panel['macro'].columns.tolist())
print(panel['macro'].tail(3))
# volume coverage per symbol
vol = panel['vol']
print("\nvolume non-null fraction per symbol:")
for s in vol.columns:
    v = vol[s]
    print(f"  {s}: {v.notna().mean():.3f} mean={v.mean():.6f}")