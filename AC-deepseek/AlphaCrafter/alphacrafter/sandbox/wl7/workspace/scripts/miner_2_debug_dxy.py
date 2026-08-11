import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from miner_2_lib import load_panel, load_macro, WATCH
panel = load_panel()
rets = panel.pct_change()
mac = load_macro()
dxy = mac["DXY"]
dxyr = dxy.pct_change()

# approach 1: per-asset Series rolling cov
s = WATCH[0]
df2 = pd.concat([rets[s].rename("a"), dxyr.rename("d")], axis=1)
beta1 = df2["a"].rolling(60, min_periods=40).cov(df2["d"]) / df2["d"].rolling(60, min_periods=40).var()
print("approach1 type", type(beta1), "shape", beta1.shape, "valid", int(beta1.notna().sum()))

# approach 2: align then rolling cov with Series
dxyr_al = dxyr.reindex(rets.index)
beta2 = rets.rolling(60, min_periods=40).cov(dxyr_al)
print("approach2 type", type(beta2), "shape", beta2.shape)
