import pandas as pd
panel = pd.read_pickle("scripts/panel_cache.pkl")
print(type(panel))
print(panel.keys() if hasattr(panel,'keys') else panel.columns)
macro = panel["macro"]
print(type(macro), macro.shape)
print(macro.columns.tolist()[:20] if hasattr(macro,'columns') else '')
v = macro["VIX"]
print(type(v), v.shape)
print(v.head(3))