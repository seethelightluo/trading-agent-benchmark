"""Debug vix_beta_cond_60x20 NaN issue."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel

close = load_close_panel()
print(f"panel dates={close.shape[0]} assets={close.shape[1]}")

vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix.set_index("date")["close"]
cutoff = close.index.max()
vix = vix[vix.index <= cutoff]
vix = vix[~vix.index.duplicated(keep="last")]
vix = vix.reindex(close.index).ffill()
print(f"VIX rows={vix.notna().sum()}")

lr = close.pct_change()
vix_ret = vix.pct_change()
beta60 = lr.rolling(60, min_periods=30).cov(vix_ret) / vix_ret.rolling(60, min_periods=30).var()
print("beta60 type:", type(beta60), "shape:", beta60.shape)
if isinstance(beta60, pd.DataFrame):
    print("beta60 notna:", beta60.notna().sum().sum(), "of", beta60.size)
    print(beta60.tail(3))
else:
    print(beta60.tail(3))

vix_move20 = vix / vix.shift(20) - 1.0
f3 = -beta60 * vix_move20
print("f3 type:", type(f3), "shape:", f3.shape)
if isinstance(f3, pd.DataFrame):
    print("f3 notna:", f3.notna().sum().sum(), "of", f3.size)
    print(f3.tail(3))
