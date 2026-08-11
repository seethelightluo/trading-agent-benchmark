"""Debug vectorized rank IC vs original on macd_12x26."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns, rank_ic_series)

panels = load_panels(days=3000)
closes = close_panel(panels)
fwd = forward_returns(closes, 10)

def per_asset(func):
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes.index)

panel = per_asset(lambda s: (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()) / s)
panel = panel.reindex(closes.index)

fr = panel.rank(axis=1, method="average")
rr = fwd.rank(axis=1, method="average")
print("fr shape", fr.shape, "rr shape", rr.shape)
print("fr index == rr index:", fr.index.equals(rr.index))
count = (fr.notna() & rr.notna()).astype(float)
n = count.sum(axis=1)
print("n describe:", n.describe().to_dict())
valid_n = (n >= 8)
print("dates n>=8:", int(valid_n.sum()))

fm = fr.fillna(0.0) - (fr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
rm = rr.fillna(0.0) - (rr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
fm = fm.where(count > 0)
rm = rm.where(count > 0)
num = (fm * rm).sum(axis=1)
den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
print("num non-null:", int(num.notna().sum()), "den>0:", int((den > 1e-14).sum()))
ic = (num / den.replace(0, np.nan)).where((n >= 8) & (den > 1e-14))
print("vec ic non-null:", int(ic.notna().sum()))

# original
ics_orig = rank_ic_series(panel, fwd, 8)
print("orig ic dates:", len(ics_orig))
if len(ics_orig) > 0:
    print("orig first ic:", ics_orig.iloc[0])
    dt = ics_orig.index[0]
    print("vec at same date:", ic.loc[dt] if dt in ic.index else "MISSING")
