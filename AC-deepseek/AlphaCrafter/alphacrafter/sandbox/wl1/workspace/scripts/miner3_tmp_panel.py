import pandas as pd, numpy as np
panel = pd.read_pickle('scripts/panel_cache.pkl')
print('panel keys:', list(panel.keys()))
for k, v in panel.items():
    print(k, v.shape, v.index.min(), v.index.max())
