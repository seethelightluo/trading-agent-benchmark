"""
Debug why volume_z has 0 IC observations
"""
import sys
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)

acct = get_account_dict()
watch_list = acct.get("watch_list", [])

for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=300)
    if df is not None:
        vol = df['volume']
        print(f"{sym}: volume stats - min={vol.min()}, max={vol.max()}, mean={vol.mean():.0f}, zeros={(vol==0).sum()}, NaN={vol.isna().sum()}")
        # check the last few rows
        print(f"  last 5 volume: {vol.tail(5).values}")
    else:
        print(f"{sym}: No data")