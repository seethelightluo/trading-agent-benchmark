"""Debug rank_ic_fast coverage discrepancy vs library rank_ic_series."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel, forward_returns,
                                 rank_ic_series, library_signals, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes shape:", closes.shape, "dtypes:", closes.dtypes.unique())
fwd10 = forward_returns(closes, 10)

lib0 = library_signals(panels, closes, rets)
mom = lib0["mom_10d_skip5"]
skew = rets.rolling(20).skew()
print("mom notna total:", int(mom.notna().sum().sum()), "of", mom.size)
print("skew notna total:", int(skew.notna().sum().sum()), "of", skew.size)
print("skew notna per col:", skew.notna().sum().to_dict())
print("fwd10 notna per col:", fwd10.notna().sum().to_dict())

# valid pair counts
def valid_pair_count(f, r, min_valid=8):
    fv = f.to_numpy(dtype=float)
    rv = r.to_numpy(dtype=float)
    cnt = 0
    n_dates = 0
    for i in range(len(f.index)):
        m = ~(np.isnan(fv[i]) | np.isnan(rv[i]))
        if m.sum() >= min_valid:
            n_dates += 1
        cnt += int(m.sum())
    return cnt, n_dates

print("mom/fwd10 valid pairs:", valid_pair_count(mom, fwd10))
print("skew/fwd10 valid pairs:", valid_pair_count(skew, fwd10))

# library rank_ic_series on same inputs
ics_lib = rank_ic_series(mom, fwd10, 8)
print("library rank_ic_series mom: n =", len(ics_lib), "mean IC =", ics_lib.mean())

# what does rank() produce for the first valid row?
print("\nmom head tail valid rows sample:")
m_valid_rows = mom.notna().sum(axis=1)
print(m_valid_rows.describe())
print("fwd10 valid rows describe:")
print(fwd10.notna().sum(axis=1).describe())
