"""Debug why close-based candidate panels yield 0 IC dates."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns, rank_ic_series)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()

p = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
print("range_pos_20d shape:", p.shape)
print("non-null count:", int(p.notna().sum().sum()))
print("per-asset non-null:")
print(p.notna().sum())
# count dates where >=8 assets valid and cross-section std > 0
valid = p.notna()
n_ge8 = int((valid.sum(axis=1) >= 8).sum())
print("dates with >=8 valid:", n_ge8)
# sample the cross-section std
std_dates = p.dropna(how="all").std(axis=1)
print("std stats:", std_dates.describe().to_dict())
# check a few dates
print(p.dropna(how="all").head(3))

fwd = forward_returns(closes, 10)
ics = rank_ic_series(p, fwd, 8)
print("rank_ic_series len:", len(ics))

# manual check on one date
dt = p.dropna(how="all").index[100]
f = p.loc[dt]; r = fwd.loc[dt]
pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
print("manual date", dt, "pair len:", len(pair), "r std:", pair["r"].std(), "f std:", pair["f"].std())
