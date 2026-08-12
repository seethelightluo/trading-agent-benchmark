import pandas as pd, numpy as np, json, os
panel = pd.read_pickle('scripts/panel_cache_20300816.pkl')
for k, v in panel.items():
    print(k, v.shape, getattr(v.index, 'min', lambda: None)(), '->', getattr(v.index, 'max', lambda: None)())
C = panel['close']
print("\nlast 8 dates:", list(C.index[-8:].astype(str)))
print("\ncoverage last 60d notna:", C.tail(60).notna().sum().to_dict())
print("\nlast close values:")
print(C.iloc[-1].round(2).to_dict())
